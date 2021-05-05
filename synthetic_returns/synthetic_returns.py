# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Episodic Memory and Synthetic Returns Core Wrapper modules."""
import collections

import haiku as hk
import jax
import jax.numpy as jnp

SRCoreWrapperOutput = collections.namedtuple(
    "SRCoreWrapperOutput", ["output", "synthetic_return", "augmented_return",
                            "sr_loss"])


class EpisodicMemory(hk.RNNCore):
  """Episodic Memory module."""

  def __init__(self, memory_size, capacity, name="episodic_memory"):
    """Constructor.

    Args:
      memory_size: Integer. The size of the vectors to be stored.
      capacity: Integer. The maximum number of memories to store before it
          becomes necessary to overwrite old memories.
      name: String. A name for this Haiku module instance.
    """
    super().__init__(name=name)
    self._memory_size = memory_size
    self._capacity = capacity

  def __call__(self, inputs, prev_state):
    """Writes a new memory into the episodic memory.

    Args:
      inputs: A Tensor of shape ``[batch_size, memory_size]``.
      prev_state: The previous state of the episodic memory, which is a tuple
         with a (i) counter of shape ``[batch_size, 1]`` indicating how many
         memories have been written so far, and (ii) a tensor of shape
         ``[batch_size, capacity, memory_size]`` with the full content of the
         episodic memory.
    Returns:
      A tuple with (i) a tensor of shape ``[batch_size, capacity, memory_size]``
          with the full content of the episodic memory, including the newly
          written memory, and (ii) the new state of the episodic memory.
    """
    inputs = jax.lax.stop_gradient(inputs)
    counter, memories = prev_state
    counter_mod = jnp.mod(counter, self._capacity)
    slot_selector = jnp.expand_dims(
        jax.nn.one_hot(counter_mod, self._capacity), axis=2)
    memories = memories * (1 - slot_selector) + (
        slot_selector * jnp.expand_dims(inputs, 1))
    counter = counter + 1
    return memories, (counter, memories)

  def initial_state(self, batch_size):
    """Creates the initial state of the episodic memory.

    Args:
      batch_size: Integer. The batch size of the episodic memory.
    Returns:
      A tuple with (i) a counter of shape ``[batch_size, 1]`` and (ii) a tensor
          of shape ``[batch_size, capacity, memory_size]`` with the full content
          of the episodic memory.
    """
    if batch_size is None:
      shape = []
    else:
      shape = [batch_size]
    counter = jnp.zeros(shape)
    memories = jnp.zeros(shape + [self._capacity, self._memory_size])
    return (counter, memories)


class SyntheticReturnsCoreWrapper(hk.RNNCore):
  """Synthetic Returns core wrapper."""

  def __init__(self, core, memory_size, capacity, hidden_layers, alpha, beta,
               loss_func=(lambda x, y: 0.5 * jnp.square(x - y)),
               apply_core_to_input=False, name="synthetic_returns_wrapper"):
    """Constructor.

    Args:
      core: hk.RNNCore. The recurrent core of the agent. E.g. an LSTM.
      memory_size: Integer. The size of the vectors to be stored in the episodic
          memory.
      capacity: Integer. The maximum number of memories to store before it
          becomes necessary to overwrite old memories.
      hidden_layers: Tuple or list of integers, indicating the size of the
          hidden layers of the MLPs used to produce synthetic returns, current
          state bias, and gate.
      alpha: The multiplier of the synthetic returns term in the augmented
          return.
      beta: The multiplier of the environment returns term in the augmented
          return.
      loss_func: A function of two arguments (predictions and targets) to
          compute the SR loss.
      apply_core_to_input: Boolean. Whether to apply the core on the inputs. If
          true, the synthetic returns will be computed from the outputs of the
          RNN core passed to the constructor. If false, the RNN core will be
          applied only at the output of this wrapper, and the synthetic returns
          will be computed from the inputs.
      name: String. A name for this Haiku module instance.
    """
    super().__init__(name=name)
    self._em = EpisodicMemory(memory_size, capacity)
    self._capacity = capacity
    hidden_layers = list(hidden_layers)
    self._synthetic_return = hk.nets.MLP(hidden_layers + [1])
    self._bias = hk.nets.MLP(hidden_layers + [1])
    self._gate = hk.Sequential([
        hk.nets.MLP(hidden_layers + [1]),
        jax.nn.sigmoid,
    ])
    self._apply_core_to_input = apply_core_to_input
    self._core = core
    self._alpha = alpha
    self._beta = beta
    self._loss = loss_func

  def initial_state(self, batch_size):
    return (
        self._em.initial_state(batch_size),
        self._core.initial_state(batch_size)
    )

  def __call__(self, inputs, prev_state):
    current_input, return_target = inputs

    em_state, core_state = prev_state
    (counter, memories) = em_state

    if self._apply_core_to_input:
      current_input, core_state = self._core(current_input, core_state)

    # Synthetic return for the current state
    synth_return = jnp.squeeze(self._synthetic_return(current_input), -1)

    # Current state bias term
    bias = self._bias(current_input)

    # Gate computed from current state
    gate = self._gate(current_input)

    # When counter > capacity, mask will be all ones
    mask = 1 - jnp.cumsum(jax.nn.one_hot(counter, self._capacity), axis=1)
    mask = jnp.expand_dims(mask, axis=2)

    # Synthetic returns for each state in memory
    past_synth_returns = hk.BatchApply(self._synthetic_return)(memories)

    # Sum of synthetic returns from previous states
    sr_sum = jnp.sum(past_synth_returns * mask, axis=1)

    prediction = jnp.squeeze(sr_sum * gate + bias, -1)
    sr_loss = self._loss(prediction, return_target)

    augmented_return = jax.lax.stop_gradient(
        self._alpha * synth_return + self._beta * return_target)

    # Write current state to memory
    _, em_state = self._em(current_input, em_state)

    if not self._apply_core_to_input:
      output, core_state = self._core(current_input, core_state)
    else:
      output = current_input

    output = SRCoreWrapperOutput(
        output=output,
        synthetic_return=synth_return,
        augmented_return=augmented_return,
        sr_loss=sr_loss,
    )
    return output, (em_state, core_state)
