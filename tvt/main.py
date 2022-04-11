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
"""Batched synchronous actor/learner training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from tvt import batch_env
from tvt import nest_utils
from tvt import rma
from tvt import tvt_rewards as tvt_module
from tvt.pycolab import env as pycolab_env
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest

FLAGS = flags.FLAGS

flags.DEFINE_integer('logging_frequency', 1,
                     'Log training progress every logging_frequency episodes.')
flags.DEFINE_string('logdir', None, 'Directory for tensorboard logging.')

flags.DEFINE_boolean('with_memory', True,
                     'whether or not agent has external memory.')
flags.DEFINE_boolean('with_reconstruction', True,
                     'whether or not agent reconstruct the observation.')
flags.DEFINE_float('gamma', 0.92, 'Agent discount factor')
flags.DEFINE_float('entropy_cost', 0.05, 'weight of the entropy loss')
flags.DEFINE_float('image_cost_weight', 50., 'image recon cost weight.')
flags.DEFINE_float('read_strength_cost', 5e-5,
                   'Cost weight of the memory read strength.')
flags.DEFINE_float('read_strength_tolerance', 2.,
                   'The tolerance of hinge loss of the read_strength_cost.')
flags.DEFINE_boolean('do_tvt', True, 'whether or not do tvt')
flags.DEFINE_enum('pycolab_game', 'key_to_door',
                  ['key_to_door', 'active_visual_match'],
                  'The name of the game in pycolab environment')
flags.DEFINE_integer('num_episodes', None,
                     'Number of episodes to train for. None means run forever.')

flags.DEFINE_integer('batch_size', 16, 'Batch size')

flags.DEFINE_float('learning_rate', 2e-4, 'Adam optimizer learning rate')
flags.DEFINE_float('beta1', 0., 'Adam optimizer beta1')
flags.DEFINE_float('beta2', 0.95, 'Adam optimizer beta2')
flags.DEFINE_float('epsilon', 1e-6, 'Adam optimizer epsilon')

# Pycolab-specific flags:
flags.DEFINE_integer('pycolab_num_apples', 10,
                     'Number of apples to sample from the distractor grid.')
flags.DEFINE_float('pycolab_apple_reward_min', 1.,
                   'A reward range [min, max) to uniformly sample from.')
flags.DEFINE_float('pycolab_apple_reward_max', 10.,
                   'A reward range [min, max) to uniformly sample from.')
flags.DEFINE_boolean('pycolab_fix_apple_reward_in_episode', True,
                     'Fix the sampled apple reward within an episode.')
flags.DEFINE_float('pycolab_final_reward', 10.,
                   'Reward obtained at the last phase.')
flags.DEFINE_boolean('pycolab_crop', True,
                     'Whether to crop observations or not.')


def main(_):

  batch_size = FLAGS.batch_size
  env_builder = pycolab_env.PycolabEnvironment
  env_kwargs = {
      'game': FLAGS.pycolab_game,
      'num_apples': FLAGS.pycolab_num_apples,
      'apple_reward': [FLAGS.pycolab_apple_reward_min,
                       FLAGS.pycolab_apple_reward_max],
      'fix_apple_reward_in_episode': FLAGS.pycolab_fix_apple_reward_in_episode,
      'final_reward': FLAGS.pycolab_final_reward,
      'crop': FLAGS.pycolab_crop
  }
  env = batch_env.BatchEnv(batch_size, env_builder, **env_kwargs)
  ep_length = env.episode_length

  agent = rma.Agent(batch_size=batch_size,
                    num_actions=env.num_actions,
                    observation_shape=env.observation_shape,
                    with_reconstructions=FLAGS.with_reconstruction,
                    gamma=FLAGS.gamma,
                    read_strength_cost=FLAGS.read_strength_cost,
                    read_strength_tolerance=FLAGS.read_strength_tolerance,
                    entropy_cost=FLAGS.entropy_cost,
                    with_memory=FLAGS.with_memory,
                    image_cost_weight=FLAGS.image_cost_weight)

  # Agent step placeholders and agent step.
  batch_shape = (batch_size,)
  observation_ph = tf.placeholder(
      dtype=tf.uint8, shape=batch_shape + env.observation_shape, name='obs')
  reward_ph = tf.placeholder(
      dtype=tf.float32, shape=batch_shape, name='reward')
  state_ph = nest.map_structure(
      lambda s: tf.placeholder(dtype=s.dtype, shape=s.shape, name='state'),
      agent.initial_state(batch_size=batch_size))
  step_outputs, state = agent.step(reward_ph, observation_ph, state_ph)

  # Update op placeholders and update op.
  observations_ph = tf.placeholder(
      dtype=tf.uint8, shape=(ep_length + 1, batch_size) + env.observation_shape,
      name='observations')
  rewards_ph = tf.placeholder(
      dtype=tf.float32, shape=(ep_length + 1, batch_size), name='rewards')
  actions_ph = tf.placeholder(
      dtype=tf.int64, shape=(ep_length, batch_size), name='actions')
  tvt_rewards_ph = tf.placeholder(
      dtype=tf.float32, shape=(ep_length, batch_size), name='tvt_rewards')

  loss, loss_logs = agent.loss(
      observations_ph, rewards_ph, actions_ph, tvt_rewards_ph)

  optimizer = tf.train.AdamOptimizer(
      learning_rate=FLAGS.learning_rate,
      beta1=FLAGS.beta1,
      beta2=FLAGS.beta2,
      epsilon=FLAGS.epsilon)
  update_op = optimizer.minimize(loss)
  initial_state = agent.initial_state(batch_size)

  if FLAGS.logdir:
    if not tf.io.gfile.exists(FLAGS.logdir):
      tf.io.gfile.makedirs(FLAGS.logdir)
    summary_writer = tf.summary.FileWriter(FLAGS.logdir)

  # Do init
  init_ops = (tf.global_variables_initializer(),
              tf.local_variables_initializer())
  tf.get_default_graph().finalize()

  sess = tf.Session()
  sess.run(init_ops)

  run = True
  ep_num = 0
  prev_logging_time = time.time()
  while run:
    observation, reward = env.reset()
    agent_state = sess.run(initial_state)

    # Initialise episode data stores.
    observations = [observation]
    rewards = [reward]
    actions = []
    baselines = []
    read_infos = []

    for _ in range(ep_length):
      step_feed = {reward_ph: reward, observation_ph: observation}
      for ph, ar in zip(nest.flatten(state_ph), nest.flatten(agent_state)):
        step_feed[ph] = ar
      step_output, agent_state = sess.run(
          (step_outputs, state), feed_dict=step_feed)
      action = step_output.action
      baseline = step_output.baseline
      read_info = step_output.read_info

      # Take step in environment, append results.
      observation, reward = env.step(action)

      observations.append(observation)
      rewards.append(reward)
      actions.append(action)
      baselines.append(baseline)
      if read_info is not None:
        read_infos.append(read_info)

    # Stack the lists of length ep_length so that each array (or each element
    # of nest stucture for read_infos) has shape (ep_length, batch_size, ...).
    observations = np.stack(observations)
    rewards = np.array(rewards)
    actions = np.array(actions)
    baselines = np.array(baselines)
    read_infos = nest_utils.nest_stack(read_infos)

    # Compute TVT rewards.
    if FLAGS.do_tvt:
      tvt_rewards = tvt_module.compute_tvt_rewards(read_infos,
                                                   baselines,
                                                   gamma=FLAGS.gamma)
    else:
      tvt_rewards = np.squeeze(np.zeros_like(baselines))

    # Run update op.
    loss_feed = {observations_ph: observations,
                 rewards_ph: rewards,
                 actions_ph: actions,
                 tvt_rewards_ph: tvt_rewards}
    ep_loss, _, ep_loss_logs = sess.run([loss, update_op, loss_logs],
                                        feed_dict=loss_feed)

    # Log episode results.
    if ep_num % FLAGS.logging_frequency == 0:
      steps_per_second = (
          FLAGS.logging_frequency * ep_length * batch_size / (
              time.time() - prev_logging_time))
      mean_reward = np.mean(np.sum(rewards, axis=0))
      mean_last_phase_reward = np.mean(env.last_phase_rewards())
      mean_tvt_reward = np.mean(np.sum(tvt_rewards, axis=0))

      logging.info('Episode %d. SPS: %s', ep_num, steps_per_second)
      logging.info('Episode %d. Mean episode reward: %f', ep_num, mean_reward)
      logging.info('Episode %d. Last phase reward: %f', ep_num,
                   mean_last_phase_reward)
      logging.info('Episode %d. Mean TVT episode reward: %f', ep_num,
                   mean_tvt_reward)
      logging.info('Episode %d. Loss: %s', ep_num, ep_loss)
      logging.info('Episode %d. Loss logs: %s', ep_num, ep_loss_logs)

      if FLAGS.logdir:
        summary = tf.Summary()
        summary.value.add(tag='reward', simple_value=mean_reward)
        summary.value.add(tag='last phase reward',
                          simple_value=mean_last_phase_reward)
        summary.value.add(tag='tvt reward', simple_value=mean_tvt_reward)
        summary.value.add(tag='total loss', simple_value=ep_loss)
        for k, v in ep_loss_logs.items():
          summary.value.add(tag='loss - {}'.format(k), simple_value=v)
        # Tensorboard x-axis is total number of episodes run.
        summary_writer.add_summary(summary, ep_num * batch_size)
        summary_writer.flush()

      prev_logging_time = time.time()

    ep_num += 1
    if FLAGS.num_episodes and ep_num >= FLAGS.num_episodes:
      run = False


if __name__ == '__main__':
  app.run(main)
