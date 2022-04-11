# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""RMA agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from six.moves import range
from six.moves import zip
import sonnet as snt
import tensorflow.compat.v1 as tf
import trfl

from tvt import losses
from tvt import memory as memory_module
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest

PolicyOutputs = collections.namedtuple(
    'PolicyOutputs', ['policy', 'action', 'baseline'])

StepOutput = collections.namedtuple(
    'StepOutput', ['action', 'baseline', 'read_info'])

AgentState = collections.namedtuple(
    'AgentState', ['core_state', 'prev_action'])

Observation = collections.namedtuple(
    'Observation', ['image', 'last_action', 'last_reward'])

RNNStateNoMem = collections.namedtuple(
    'RNNStateNoMem', ['controller_outputs', 'h_controller'])

RNNState = collections.namedtuple(
    'RNNState',
    list(RNNStateNoMem._fields) + ['memory', 'mem_reads', 'h_mem_writer'])

CoreOutputs = collections.namedtuple(
    'CoreOutputs', ['action', 'policy', 'baseline', 'z', 'read_info'])


def rnn_inputs_to_static_rnn_inputs(inputs):
  """Converts time major tensors to timestep lists."""
  # Inputs to core build method are expected to be a tensor or tuple of tensors.
  if isinstance(inputs, tuple):
    num_timesteps = inputs[0].shape.as_list()[0]
    converted_inputs = [tf.unstack(input_, num_timesteps) for input_ in inputs]
    return list(zip(*converted_inputs))
  else:
    return tf.unstack(inputs)


def static_rnn_outputs_to_core_outputs(outputs):
  """Convert from length T list of nests to nest of tensors with first dim T."""
  list_of_flats = [nest.flatten(n) for n in outputs]
  new_outputs = list()
  for i in range(len(list_of_flats[0])):
    new_outputs.append(tf.stack([flat_nest[i] for flat_nest in list_of_flats]))
  return nest.pack_sequence_as(structure=outputs[0], flat_sequence=new_outputs)


def unroll(core, initial_state, inputs, dtype=tf.float32):
  """Perform a static unroll of the core."""
  static_rnn_inputs = rnn_inputs_to_static_rnn_inputs(inputs)
  static_outputs, _ = tf.nn.static_rnn(
      core,
      inputs=static_rnn_inputs,
      initial_state=initial_state,
      dtype=dtype)
  core_outputs = static_rnn_outputs_to_core_outputs(static_outputs)
  return core_outputs


class ImageEncoderDecoder(snt.AbstractModule):
  """Image Encoder/Decoder module."""

  def __init__(
      self,
      image_code_size,
      name='image_encoder_decoder'):
    """Initialize the image encoder/decoder."""
    super(ImageEncoderDecoder, self).__init__(name=name)

    # This is set by a call to `encode`. `decode` will fail before this is set.
    self._convnet_output_shape = None

    with self._enter_variable_scope():
      self._convnet = snt.nets.ConvNet2D(
          output_channels=(16, 32),
          kernel_shapes=(3, 3),
          strides=(1, 1),
          paddings=('SAME',))
      self._post_convnet_layer = snt.Linear(image_code_size, name='final_layer')

  @snt.reuse_variables
  def encode(self, image):
    """Encode the image observation."""
    convnet_output = self._convnet(image)

    # Store unflattened convnet output shape for use in decoder.
    self._convnet_output_shape = convnet_output.shape[1:]

    # Flatten convnet outputs and pass through final layer to get image code.
    return self._post_convnet_layer(snt.BatchFlatten()(convnet_output))

  @snt.reuse_variables
  def decode(self, code):
    """Decode the image observation from a latent code."""
    if self._convnet_output_shape is None:
      raise ValueError('Must call `encode` before `decode`.')
    transpose_convnet_in_flat = snt.Linear(
        self._convnet_output_shape.num_elements(),
        name='decode_initial_linear')(
            code)
    transpose_convnet_in_flat = tf.nn.relu(transpose_convnet_in_flat)
    transpose_convnet_in = snt.BatchReshape(
        self._convnet_output_shape.as_list())(transpose_convnet_in_flat)
    return self._convnet.transpose(None)(transpose_convnet_in)

  def _build(self, *args):  # Unused. Use encode/decode instead.
    raise NotImplementedError('Use encode/decode methods instead of __call__.')


class Policy(snt.AbstractModule):
  """A policy module possibly containing a read-only DNC."""

  def __init__(self,
               num_actions,
               num_policy_hiddens=(),
               num_baseline_hiddens=(),
               activation=tf.nn.tanh,
               policy_clip_abs_value=10.0,
               name='Policy'):
    """Construct a policy module possibly containing a read-only DNC.

    Args:
      num_actions: Number of discrete actions to choose from.
      num_policy_hiddens: Tuple or List, sizes of policy MLP hidden layers.
      num_baseline_hiddens: Tuple or List, sizes of baseline MLP hidden layers.
          An empty tuple/list results in a linear layer instead of an MLP.
      activation: Callable, e.g. tf.nn.tanh.
      policy_clip_abs_value: float, Policy gradient clip value.
      name: A string, the module's name
    """
    super(Policy, self).__init__(name=name)

    self._num_actions = num_actions
    self._policy_layers = tuple(num_policy_hiddens) + (num_actions,)
    self._baseline_layers = tuple(num_baseline_hiddens) + (1,)
    self._policy_clip_abs_value = policy_clip_abs_value
    self._activation = activation

  def _build(self, inputs):
    (shared_inputs, extra_policy_inputs) = inputs
    policy_in = tf.concat([shared_inputs, extra_policy_inputs], axis=1)

    policy = snt.nets.MLP(
        output_sizes=self._policy_layers,
        activation=self._activation,
        name='policy_mlp')(
            policy_in)

    # Sample an action from the policy logits.
    action = tf.multinomial(policy, num_samples=1, output_dtype=tf.int32)
    action = tf.squeeze(action, 1)  # [B, 1] -> [B]

    if self._policy_clip_abs_value > 0:
      policy = snt.clip_gradient(
          net=policy,
          clip_value_min=-self._policy_clip_abs_value,
          clip_value_max=self._policy_clip_abs_value)

    baseline_in = tf.concat([shared_inputs, tf.stop_gradient(policy)], axis=1)
    baseline = snt.nets.MLP(
        self._baseline_layers,
        activation=self._activation,
        name='baseline_mlp')(
            baseline_in)
    baseline = tf.squeeze(baseline, axis=-1)  # [B, 1] -> [B]

    if self._policy_clip_abs_value > 0:
      baseline = snt.clip_gradient(
          net=baseline,
          clip_value_min=-self._policy_clip_abs_value,
          clip_value_max=self._policy_clip_abs_value)

    outputs = PolicyOutputs(
        policy=policy,
        action=action,
        baseline=baseline)

    return outputs


class _RMACore(snt.RNNCore):
  """RMA RNN Core."""

  def __init__(self,
               num_actions,
               with_memory=True,
               name='rma_core'):
    super(_RMACore, self).__init__(name=name)

    # MLP activation as callable.
    mlp_activation = tf.nn.tanh

    # Size of latent code written to memory (if using it) and used to
    # reconstruct from (if including reconstructions).
    num_latents = 200

    # Value function decode settings.
    baseline_mlp_num_hiddens = (200,)

    # Policy settings.
    num_policy_hiddens = (200,)  # Only used for non-recurrent core.

    # Controller settings.
    control_hidden_size = 256
    control_num_layers = 2

    # Memory settings (only used if with_memory=True).
    memory_size = 1000
    memory_num_reads = 3
    memory_top_k = 50

    self._with_memory = with_memory

    with self._enter_variable_scope():
      # Construct the features -> latent encoder.
      self._z_encoder_mlp = snt.nets.MLP(
          output_sizes=(2 * num_latents, num_latents),
          activation=mlp_activation,
          activate_final=False,
          name='z_encoder_mlp')

      # Construct controller.
      rnn_cores = [snt.LSTM(control_hidden_size)
                   for _ in range(control_num_layers)]
      self._controller = snt.DeepRNN(
          rnn_cores, skip_connections=True, name='controller')

      # Construct memory.
      if self._with_memory:
        memory_dim = num_latents  # Each write to memory is of size memory_dim.
        self._mem_shape = (memory_size, memory_dim)
        self._memory_reader = memory_module.MemoryReader(
            memory_word_size=memory_dim,
            num_read_heads=memory_num_reads,
            top_k=memory_top_k,
            memory_size=memory_size)
        self._memory_writer = memory_module.MemoryWriter(
            mem_shape=self._mem_shape)

      # Construct policy, starting with policy_core and policy_action_head.
      # `extra_inputs` in this case will be mem_out from current time step (note
      # that mem_out is just the controller output if with_memory=False).
      self._policy = Policy(
          num_policy_hiddens=num_policy_hiddens,
          num_actions=num_actions,
          num_baseline_hiddens=baseline_mlp_num_hiddens,
          activation=mlp_activation,
          policy_clip_abs_value=10.0,)

    # Set state_size and output_size.
    controller_out_size = self._controller.output_size
    controller_state_size = self._controller.state_size
    self._state_size = RNNStateNoMem(controller_outputs=controller_out_size,
                                     h_controller=controller_state_size)
    read_info_size = ()
    if self._with_memory:
      mem_reads_size, read_info_size = self._memory_reader.output_size
      mem_writer_state_size = self._memory_writer.state_size
      self._state_size = RNNState(memory=tf.TensorShape(self._mem_shape),
                                  mem_reads=mem_reads_size,
                                  h_mem_writer=mem_writer_state_size,
                                  **self._state_size._asdict())

    z_size = num_latents
    self._output_size = CoreOutputs(
        action=tf.TensorShape([]),  # Scalar tensor shapes must be explicit.
        policy=num_actions,
        baseline=tf.TensorShape([]),  # Scalar tensor shapes must be explicit.
        z=z_size,
        read_info=read_info_size)

  def _build(self, inputs, h_prev):
    features = inputs

    z_net_inputs = [features, h_prev.controller_outputs]
    if self._with_memory:
      z_net_inputs.append(h_prev.mem_reads)
    z_net_inputs_concat = tf.concat(z_net_inputs, axis=1)
    z = self._z_encoder_mlp(z_net_inputs_concat)

    controller_out, h_controller = self._controller(z, h_prev.h_controller)

    read_info = ()
    if self._with_memory:
      # Perform a memory read/write step before generating the policy_modules.
      mem_reads, read_info = self._memory_reader((controller_out,
                                                  h_prev.memory))
      memory, h_mem_writer = self._memory_writer((z, h_prev.memory),
                                                 h_prev.h_mem_writer)
      policy_extra_input = tf.concat([controller_out, mem_reads], axis=1)
    else:
      policy_extra_input = controller_out

    # Get policy, action and (possible empty) baseline from policy module.
    policy_inputs = (z, policy_extra_input)
    policy_outputs = self._policy(policy_inputs)
    core_outputs = CoreOutputs(
        z=z,
        read_info=read_info,
        **policy_outputs._asdict())

    h_next = RNNStateNoMem(controller_outputs=controller_out,
                           h_controller=h_controller)
    if self._with_memory:
      h_next = RNNState(memory=memory,
                        mem_reads=mem_reads,
                        h_mem_writer=h_mem_writer,
                        **h_next._asdict())

    return core_outputs, h_next

  def initial_state(self, batch_size):
    """Use initial state for RNN modules, otherwise use zero state."""
    zero_state = self.zero_state(batch_size, dtype=tf.float32)
    controller_out = zero_state.controller_outputs
    h_controller = self._controller.initial_state(batch_size)

    state = RNNStateNoMem(controller_outputs=controller_out,
                          h_controller=h_controller)
    if self._with_memory:
      memory = zero_state.memory
      mem_reads = zero_state.mem_reads
      h_mem_writer = self._memory_writer.initial_state(batch_size)
      state = RNNState(memory=memory,
                       mem_reads=mem_reads,
                       h_mem_writer=h_mem_writer,
                       **state._asdict())
    return state

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size


class Agent(snt.AbstractModule):
  """Myriad RMA agent.

  `latents` here refers to a purely deterministic encoding of the inputs, rather
  than VAE-like latents in e.g. the MERLIN agent.
  """

  def __init__(self,
               batch_size,
               with_reconstructions=True,
               with_memory=True,
               image_code_size=500,
               image_cost_weight=50.,
               num_actions=None,
               observation_shape=None,
               entropy_cost=0.01,
               return_cost_weight=0.4,
               gamma=0.96,
               read_strength_cost=5e-5,
               read_strength_tolerance=2.,
               name='rma_agent'):
    super(Agent, self).__init__(name=name)

    self._batch_size = batch_size
    self._with_reconstructions = with_reconstructions
    self._image_cost_weight = image_cost_weight
    self._image_code_size = image_code_size
    self._entropy_cost = entropy_cost
    self._return_cost_weight = return_cost_weight
    self._gamma = gamma
    self._read_strength_cost = read_strength_cost
    self._read_strength_tolerance = read_strength_tolerance
    self._num_actions = num_actions
    self._name = name
    self._logged_values = {}

    # Store total number of pixels across channels (for image loss scaling).
    self._total_num_pixels = np.prod(observation_shape)

    with self._enter_variable_scope():

      # Construct image encoder/decoder.
      self._image_encoder_decoder = ImageEncoderDecoder(
          image_code_size=image_code_size)

      self._core = _RMACore(
          num_actions=self._num_actions,
          with_memory=with_memory)

  def initial_state(self, batch_size):
    with tf.name_scope(self._name + '/initial_state'):
      return AgentState(
          core_state=self._core.initial_state(batch_size),
          prev_action=tf.zeros(shape=(batch_size,), dtype=tf.int32))

  def _prepare_observations(self, observation, last_reward, last_action):
    image = observation

    # Make sure the entries are in [0, 1) range.
    if image.dtype.is_integer:
      image = tf.cast(image, tf.float32) / 255.

    if last_reward is None:
      # For some envs, in the first timestep the last_reward can be None.
      batch_size = observation.shape[0]
      last_reward = tf.zeros((batch_size,), dtype=tf.float32)

    return Observation(
        image=image,
        last_action=last_action,
        last_reward=last_reward)

  @snt.reuse_variables
  def _encode(self, observation, last_reward, last_action):
    inputs = self._prepare_observations(observation, last_reward, last_action)

    # Encode image observation.
    obs_code = self._image_encoder_decoder.encode(inputs.image)

    # Encode last action.
    action_code = tf.one_hot(inputs.last_action, self._num_actions)

    # Encode last reward.
    reward_code = tf.expand_dims(inputs.last_reward, -1)

    features = tf.concat([obs_code, action_code, reward_code], axis=1)

    return inputs, features

  @snt.reuse_variables
  def _decode(self, z):
    # Decode image.
    image_recon = self._image_encoder_decoder.decode(z)

    # Decode action.
    action_recon = snt.Linear(self._num_actions, name='action_recon_linear')(z)

    # Decode reward.
    reward_recon = snt.Linear(1, name='reward_recon_linear')(z)

    # Full reconstructions.
    recons = Observation(
        image=image_recon,
        last_reward=reward_recon,
        last_action=action_recon)

    return recons

  def step(self, reward, observation, prev_state):
    with tf.name_scope(self._name + '/step'):
      _, features = self._encode(observation, reward, prev_state.prev_action)

      core_outputs, next_core_state = self._core(
          features, prev_state.core_state)

      action = core_outputs.action

    step_output = StepOutput(
        action=action,
        baseline=core_outputs.baseline,
        read_info=core_outputs.read_info)
    agent_state = AgentState(
        core_state=next_core_state,
        prev_action=action)
    return step_output, agent_state

  @snt.reuse_variables
  def loss(self, observations, rewards, actions, additional_rewards=None):
    """Compute the loss."""
    dummy_zeroth_step_actions = tf.zeros_like(actions[:1])
    all_actions = tf.concat([dummy_zeroth_step_actions, actions], axis=0)
    inputs, features = snt.BatchApply(self._encode)(
        observations, rewards, all_actions)

    rewards = rewards[1:]  # Zeroth step reward not correlated to actions.
    if additional_rewards is not None:
      # Additional rewards are not passed to the encoder (above) in order to be
      # consistent with the step, nor to the recon loss so that recons are
      # consistent with the observations. Thus, additional rewards only affect
      # the returns used to learn the value function.
      rewards += additional_rewards

    initial_state = self._core.initial_state(self._batch_size)

    rnn_inputs = features
    core_outputs = unroll(self._core, initial_state, rnn_inputs)

    # Remove final timestep of outputs.
    core_outputs = nest.map_structure(lambda t: t[:-1], core_outputs)

    if self._with_reconstructions:
      recons = snt.BatchApply(self._decode)(core_outputs.z)
      recon_targets = nest.map_structure(lambda t: t[:-1], inputs)
      recon_loss, recon_logged_values = losses.reconstruction_losses(
          recons=recons,
          targets=recon_targets,
          image_cost=self._image_cost_weight / self._total_num_pixels,
          action_cost=1.,
          reward_cost=1.)
    else:
      recon_loss = tf.constant(0.0)
      recon_logged_values = dict()

    if core_outputs.read_info is not tuple():
      read_reg_loss, read_reg_logged_values = (
          losses.read_regularization_loss(
              read_info=core_outputs.read_info,
              strength_cost=self._read_strength_cost,
              strength_tolerance=self._read_strength_tolerance,
              strength_reg_mode='L1',
              key_norm_cost=0.,
              key_norm_tolerance=1.))
    else:
      read_reg_loss = tf.constant(0.0)
      read_reg_logged_values = dict()

    # Bootstrap value is at end of episode so is zero.
    bootstrap_value = tf.zeros(shape=(self._batch_size,), dtype=tf.float32)

    discounts = self._gamma * tf.ones_like(rewards)

    a2c_loss, a2c_loss_extra = trfl.sequence_advantage_actor_critic_loss(
        policy_logits=core_outputs.policy,
        baseline_values=core_outputs.baseline,
        actions=actions,
        rewards=rewards,
        pcontinues=discounts,
        bootstrap_value=bootstrap_value,
        lambda_=self._gamma,
        entropy_cost=self._entropy_cost,
        baseline_cost=self._return_cost_weight,
        name='SequenceA2CLoss')

    a2c_loss = tf.reduce_mean(a2c_loss)  # Average over batch.

    total_loss = a2c_loss + recon_loss + read_reg_loss

    a2c_loss_logged_values = dict(
        pg_loss=tf.reduce_mean(a2c_loss_extra.policy_gradient_loss),
        baseline_loss=tf.reduce_mean(a2c_loss_extra.baseline_loss),
        entropy_loss=tf.reduce_mean(a2c_loss_extra.entropy_loss))
    agent_loss_log = losses.combine_logged_values(
        a2c_loss_logged_values,
        recon_logged_values,
        read_reg_logged_values)
    agent_loss_log['total_loss'] = total_loss

    return total_loss, agent_loss_log

  def _build(self, *args):  # Unused.
    # pylint: disable=no-value-for-parameter
    return self.step(*args)
    # pylint: enable=no-value-for-parameter
