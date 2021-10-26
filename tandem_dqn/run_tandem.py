# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A Tandem DQN agent implemented in JAX, training on Atari."""

import collections
import itertools
import sys
import typing

from absl import app
from absl import flags
from absl import logging
import dm_env
import jax
from jax.config import config
import numpy as np
import optax

from tandem_dqn import agent as agent_lib
from tandem_dqn import atari_data
from tandem_dqn import gym_atari
from tandem_dqn import losses
from tandem_dqn import networks
from tandem_dqn import parts
from tandem_dqn import processors
from tandem_dqn import replay as replay_lib

# Relevant flag values are expressed in terms of environment frames.
FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'pong', '')
flags.DEFINE_boolean('use_sticky_actions', False, '')
flags.DEFINE_integer('environment_height', 84, '')
flags.DEFINE_integer('environment_width', 84, '')
flags.DEFINE_integer('replay_capacity', int(1e6), '')
flags.DEFINE_bool('compress_state', True, '')
flags.DEFINE_float('min_replay_capacity_fraction', 0.05, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_frames_per_episode', 108000, '')  # 30 mins.
flags.DEFINE_integer('num_action_repeats', 4, '')
flags.DEFINE_integer('num_stacked_frames', 4, '')
flags.DEFINE_float('exploration_epsilon_begin_value', 1., '')
flags.DEFINE_float('exploration_epsilon_end_value', 0.01, '')
flags.DEFINE_float('exploration_epsilon_decay_frame_fraction', 0.02, '')
flags.DEFINE_float('eval_exploration_epsilon', 0.01, '')
flags.DEFINE_integer('target_network_update_period', int(1.2e5), '')
flags.DEFINE_float('additional_discount', 0.99, '')
flags.DEFINE_float('max_abs_reward', 1., '')
flags.DEFINE_integer('seed', 1, '')  # GPU may introduce nondeterminism.
flags.DEFINE_integer('num_iterations', 200, '')
flags.DEFINE_integer('num_train_frames', int(1e6), '')  # Per iteration.
flags.DEFINE_integer('num_eval_frames', int(5e5), '')  # Per iteration.
flags.DEFINE_integer('learn_period', 16, '')
flags.DEFINE_string('results_csv_path', '/tmp/results.csv', '')

# Tandem-specific parameters.

# Using fixed configs for optimizers:
# RMSProp: lr = 0.00025, eps=0.01 / (32 ** 2)
# ADAM:    lr = 0.00005, eps=0.01 / 32
_OPTIMIZERS = ['rmsprop', 'adam']
flags.DEFINE_enum('optimizer_active', 'rmsprop', _OPTIMIZERS, '')
flags.DEFINE_enum('optimizer_passive', 'rmsprop', _OPTIMIZERS, '')

_NETWORKS = ['double_q', 'qr']
flags.DEFINE_enum('network_active', 'double_q', _NETWORKS, '')
flags.DEFINE_enum('network_passive', 'double_q', _NETWORKS, '')

_LOSSES = ['double_q', 'double_q_v', 'double_q_p', 'double_q_pv', 'qr',
           'q_regression']
flags.DEFINE_enum('loss_active', 'double_q', _LOSSES, '')
flags.DEFINE_enum('loss_passive', 'double_q', _LOSSES, '')

flags.DEFINE_integer('tied_layers', 0, '')

TandemTuple = agent_lib.TandemTuple


def make_optimizer(optimizer_type):
  """Constructs optimizer."""
  if optimizer_type == 'rmsprop':
    learning_rate = 0.00025
    epsilon = 0.01 / (32**2)
    optimizer = optax.rmsprop(
        learning_rate=learning_rate,
        decay=0.95,
        eps=epsilon,
        centered=True)
  elif optimizer_type == 'adam':
    learning_rate = 0.00005
    epsilon = 0.01 / 32
    optimizer = optax.adam(
        learning_rate=learning_rate,
        eps=epsilon)
  else:
    raise ValueError('Unknown optimizer "{}"'.format(optimizer_type))
  return optimizer


def main(argv):
  """Trains Tandem DQN agent on Atari."""
  del argv
  logging.info('Tandem DQN on Atari on %s.',
               jax.lib.xla_bridge.get_backend().platform)
  random_state = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(
      random_state.randint(-sys.maxsize - 1, sys.maxsize + 1, dtype=np.int64))

  if FLAGS.results_csv_path:
    writer = parts.CsvWriter(FLAGS.results_csv_path)
  else:
    writer = parts.NullWriter()

  def environment_builder():
    """Creates Atari environment."""
    env = gym_atari.GymAtari(
        FLAGS.environment_name,
        sticky_actions=FLAGS.use_sticky_actions,
        seed=random_state.randint(1, 2**32))
    return gym_atari.RandomNoopsEnvironmentWrapper(
        env,
        min_noop_steps=1,
        max_noop_steps=30,
        seed=random_state.randint(1, 2**32),
    )

  env = environment_builder()

  logging.info('Environment: %s', FLAGS.environment_name)
  logging.info('Action spec: %s', env.action_spec())
  logging.info('Observation spec: %s', env.observation_spec())
  num_actions = env.action_spec().num_values

  # Check: qr network and qr losses can only be used together.
  if ('qr' in FLAGS.network_active) != ('qr' in FLAGS.loss_active):
    raise ValueError('Active loss/net must either both use QR, or neither.')
  if ('qr' in FLAGS.network_passive) != ('qr' in FLAGS.loss_passive):
    raise ValueError('Passive loss/net must either both use QR, or neither.')
  network = TandemTuple(
      active=networks.make_network(FLAGS.network_active, num_actions),
      passive=networks.make_network(FLAGS.network_passive, num_actions),
  )
  loss = TandemTuple(
      active=losses.make_loss_fn(FLAGS.loss_active, active=True),
      passive=losses.make_loss_fn(FLAGS.loss_passive, active=False),
  )

  # Tied layers.
  assert 0 <= FLAGS.tied_layers <= 4
  if FLAGS.tied_layers > 0 and (FLAGS.network_passive != 'double_q'
                                or FLAGS.network_active != 'double_q'):
    raise ValueError('Tied layers > 0 is only supported for double_q networks.')
  layers = [
      'sequential/sequential/conv1',
      'sequential/sequential/conv2',
      'sequential/sequential/conv3',
      'sequential/sequential_1/linear1'
  ]
  tied_layers = set(layers[:FLAGS.tied_layers])

  def preprocessor_builder():
    return processors.atari(
        additional_discount=FLAGS.additional_discount,
        max_abs_reward=FLAGS.max_abs_reward,
        resize_shape=(FLAGS.environment_height, FLAGS.environment_width),
        num_action_repeats=FLAGS.num_action_repeats,
        num_pooled_frames=2,
        zero_discount_on_life_loss=True,
        num_stacked_frames=FLAGS.num_stacked_frames,
        grayscaling=True,
    )

  # Create sample network input from sample preprocessor output.
  sample_processed_timestep = preprocessor_builder()(env.reset())
  sample_processed_timestep = typing.cast(dm_env.TimeStep,
                                          sample_processed_timestep)
  sample_network_input = sample_processed_timestep.observation
  assert sample_network_input.shape == (FLAGS.environment_height,
                                        FLAGS.environment_width,
                                        FLAGS.num_stacked_frames)

  exploration_epsilon_schedule = parts.LinearSchedule(
      begin_t=int(FLAGS.min_replay_capacity_fraction * FLAGS.replay_capacity *
                  FLAGS.num_action_repeats),
      decay_steps=int(FLAGS.exploration_epsilon_decay_frame_fraction *
                      FLAGS.num_iterations * FLAGS.num_train_frames),
      begin_value=FLAGS.exploration_epsilon_begin_value,
      end_value=FLAGS.exploration_epsilon_end_value)

  if FLAGS.compress_state:

    def encoder(transition):
      return transition._replace(
          s_tm1=replay_lib.compress_array(transition.s_tm1),
          s_t=replay_lib.compress_array(transition.s_t))

    def decoder(transition):
      return transition._replace(
          s_tm1=replay_lib.uncompress_array(transition.s_tm1),
          s_t=replay_lib.uncompress_array(transition.s_t))
  else:
    encoder = None
    decoder = None

  replay_structure = replay_lib.Transition(
      s_tm1=None,
      a_tm1=None,
      r_t=None,
      discount_t=None,
      s_t=None,
      a_t=None,
      mc_return_tm1=None,
  )

  replay = replay_lib.TransitionReplay(FLAGS.replay_capacity, replay_structure,
                                       random_state, encoder, decoder)

  optimizer = TandemTuple(
      active=make_optimizer(FLAGS.optimizer_active),
      passive=make_optimizer(FLAGS.optimizer_passive),
  )

  train_rng_key, eval_rng_key = jax.random.split(rng_key)

  train_agent = agent_lib.TandemDqn(
      preprocessor=preprocessor_builder(),
      sample_network_input=sample_network_input,
      network=network,
      optimizer=optimizer,
      loss=loss,
      transition_accumulator=replay_lib.TransitionAccumulatorWithMCReturn(),
      replay=replay,
      batch_size=FLAGS.batch_size,
      exploration_epsilon=exploration_epsilon_schedule,
      min_replay_capacity_fraction=FLAGS.min_replay_capacity_fraction,
      learn_period=FLAGS.learn_period,
      target_network_update_period=FLAGS.target_network_update_period,
      tied_layers=tied_layers,
      rng_key=train_rng_key,
  )
  eval_agent_active = parts.EpsilonGreedyActor(
      preprocessor=preprocessor_builder(),
      network=network.active,
      exploration_epsilon=FLAGS.eval_exploration_epsilon,
      rng_key=eval_rng_key)
  eval_agent_passive = parts.EpsilonGreedyActor(
      preprocessor=preprocessor_builder(),
      network=network.passive,
      exploration_epsilon=FLAGS.eval_exploration_epsilon,
      rng_key=eval_rng_key)

  # Set up checkpointing.
  checkpoint = parts.NullCheckpoint()

  state = checkpoint.state
  state.iteration = 0
  state.train_agent = train_agent
  state.eval_agent_active = eval_agent_active
  state.eval_agent_passive = eval_agent_passive
  state.random_state = random_state
  state.writer = writer
  if checkpoint.can_be_restored():
    checkpoint.restore()

  # Run single iteration of training or evaluation.
  def run_iteration(agent, env, num_frames):
    seq = parts.run_loop(agent, env, FLAGS.max_frames_per_episode)
    seq_truncated = itertools.islice(seq, num_frames)
    trackers = parts.make_default_trackers(agent)
    return parts.generate_statistics(trackers, seq_truncated)

  def eval_log_output(eval_stats, suffix):
    human_normalized_score = atari_data.get_human_normalized_score(
        FLAGS.environment_name, eval_stats['episode_return'])
    capped_human_normalized_score = np.amin([1., human_normalized_score])
    return [
        ('eval_episode_return_' + suffix,
         eval_stats['episode_return'], '% 2.2f'),
        ('eval_num_episodes_' + suffix,
         eval_stats['num_episodes'], '%3d'),
        ('eval_frame_rate_' + suffix,
         eval_stats['step_rate'], '%4.0f'),
        ('normalized_return_' + suffix,
         human_normalized_score, '%.3f'),
        ('capped_normalized_return_' + suffix,
         capped_human_normalized_score, '%.3f'),
        ('human_gap_' + suffix,
         1. - capped_human_normalized_score, '%.3f'),
    ]

  while state.iteration <= FLAGS.num_iterations:
    # New environment for each iteration to allow for determinism if preempted.
    env = environment_builder()

    # Set agent to train active and passive nets on each learning step.
    train_agent.set_training_mode('active_passive')

    logging.info('Training iteration %d.', state.iteration)
    num_train_frames = 0 if state.iteration == 0 else FLAGS.num_train_frames
    train_stats = run_iteration(train_agent, env, num_train_frames)

    logging.info('Evaluation iteration %d - active agent.', state.iteration)
    eval_agent_active.network_params = train_agent.online_params.active
    eval_stats_active = run_iteration(eval_agent_active, env,
                                      FLAGS.num_eval_frames)

    logging.info('Evaluation iteration %d - passive agent.', state.iteration)
    eval_agent_passive.network_params = train_agent.online_params.passive
    eval_stats_passive = run_iteration(eval_agent_passive, env,
                                       FLAGS.num_eval_frames)

    # Logging and checkpointing.
    agent_logs = [
        'loss_active',
        'loss_passive',
        'frac_diff_argmax',
        'mc_error_active',
        'mc_error_passive',
        'mc_error_abs_active',
        'mc_error_abs_passive',
    ]
    log_output = (
        eval_log_output(eval_stats_active, 'active') +
        eval_log_output(eval_stats_passive, 'passive') +
        [('iteration', state.iteration, '%3d'),
         ('frame', state.iteration * FLAGS.num_train_frames, '%5d'),
         ('train_episode_return', train_stats['episode_return'], '% 2.2f'),
         ('train_num_episodes', train_stats['num_episodes'], '%3d'),
         ('train_frame_rate', train_stats['step_rate'], '%4.0f'),
         ] +
        [(k, train_stats[k], '% 2.2f') for k in agent_logs]
    )
    log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
    logging.info(log_output_str)
    writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
    state.iteration += 1
    checkpoint.save()

  writer.close()


if __name__ == '__main__':
  config.update('jax_platform_name', 'gpu')  # Default to GPU.
  config.update('jax_numpy_rank_promotion', 'raise')
  config.config_with_absl()
  app.run(main)
