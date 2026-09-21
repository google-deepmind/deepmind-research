# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Train simple phi model."""

import collections
import random

from absl import app
from absl import flags
from absl import logging

import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf
import tree

from option_keyboard import scavenger
from option_keyboard import smart_module

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_phis", 2, "Dimensionality of phis.")
flags.DEFINE_integer("num_train_steps", 2000, "Number of training steps.")
flags.DEFINE_integer("num_replay_steps", 500, "Number of replay steps.")
flags.DEFINE_integer("min_replay_size", 1000,
                     "Minimum replay size before starting training.")
flags.DEFINE_integer("num_train_repeats", 10, "Number of training repeats.")
flags.DEFINE_float("learning_rate", 3e-3, "Learning rate.")
flags.DEFINE_bool("use_random_tasks", False, "Use random tasks.")
flags.DEFINE_string("normalisation", "L2",
                    "Normalisation method for cumulant weights.")
flags.DEFINE_string("export_path", None, "Export path.")


StepOutput = collections.namedtuple("StepOutput",
                                    ["obs", "actions", "rewards", "next_obs"])


def collect_experience(env, num_episodes, verbose=False):
  """Collect experience."""

  num_actions = env.action_spec().maximum + 1

  observations = []
  actions = []
  rewards = []
  next_observations = []

  for _ in range(num_episodes):
    timestep = env.reset()
    episode_return = 0
    while not timestep.last():
      action = np.random.randint(num_actions)
      observations.append(timestep.observation)
      actions.append(action)

      timestep = env.step(action)
      rewards.append(timestep.observation["aux_tasks_reward"])
      episode_return += timestep.reward

      next_observations.append(timestep.observation)

    if verbose:
      logging.info("Total return for episode: %f", episode_return)

  observation_spec = tree.map_structure(lambda _: None, observations[0])

  def stack_observations(obs_list):
    obs_list = [
        np.stack(obs) for obs in zip(*[tree.flatten(obs) for obs in obs_list])
    ]
    obs_dict = tree.unflatten_as(observation_spec, obs_list)
    obs_dict.pop("aux_tasks_reward")
    return obs_dict

  observations = stack_observations(observations)
  actions = np.array(actions, dtype=np.int32)
  rewards = np.stack(rewards)
  next_observations = stack_observations(next_observations)

  return StepOutput(observations, actions, rewards, next_observations)


class PhiModel(snt.AbstractModule):
  """A model for learning phi."""

  def __init__(self,
               n_actions,
               n_phis,
               network_kwargs,
               final_activation="sigmoid",
               name="PhiModel"):
    super(PhiModel, self).__init__(name=name)
    self._n_actions = n_actions
    self._n_phis = n_phis
    self._network_kwargs = network_kwargs
    self._final_activation = final_activation

  def _build(self, observation, actions):
    obs = observation["arena"]

    n_outputs = self._n_actions * self._n_phis
    flat_obs = snt.BatchFlatten()(obs)
    net = snt.nets.MLP(**self._network_kwargs)(flat_obs)
    net = snt.Linear(output_size=n_outputs)(net)
    net = snt.BatchReshape((self._n_actions, self._n_phis))(net)

    indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
    values = tf.gather_nd(net, indices)
    if self._final_activation:
      values = getattr(tf.nn, self._final_activation)(values)

    return values


def create_ph(tensor):
  return tf.placeholder(shape=(None,) + tensor.shape[1:], dtype=tensor.dtype)


def main(argv):
  del argv

  if FLAGS.use_random_tasks:
    tasks = np.random.normal(size=(8, 2))
  else:
    tasks = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ]

  if FLAGS.normalisation == "L1":
    tasks /= np.sum(np.abs(tasks), axis=-1, keepdims=True)
  elif FLAGS.normalisation == "L2":
    tasks /= np.linalg.norm(tasks, axis=-1, keepdims=True)
  else:
    raise ValueError("Unknown normlisation_method {}".format(
        FLAGS.normalisation))

  logging.info("Tasks: %s", tasks)

  env_config = dict(
      arena_size=11,
      num_channels=2,
      max_num_steps=100,
      num_init_objects=10,
      object_priors=[1.0, 1.0],
      egocentric=True,
      default_w=None,
      aux_tasks_w=tasks)
  env = scavenger.Scavenger(**env_config)
  num_actions = env.action_spec().maximum + 1

  model_config = dict(
      n_actions=num_actions,
      n_phis=FLAGS.num_phis,
      network_kwargs=dict(
          output_sizes=(64, 128),
          activate_final=True,
      ),
  )
  model = smart_module.SmartModuleExport(lambda: PhiModel(**model_config))

  dummy_steps = collect_experience(env, num_episodes=10, verbose=True)
  num_rewards = dummy_steps.rewards.shape[-1]

  # Placeholders
  steps_ph = tree.map_structure(create_ph, dummy_steps)

  phis = model(steps_ph.obs, steps_ph.actions)
  phis_to_rewards = snt.Linear(
      num_rewards, initializers=dict(w=tf.zeros), use_bias=False)
  preds = phis_to_rewards(phis)
  loss_per_batch = tf.square(preds - steps_ph.rewards)
  loss_op = tf.reduce_mean(loss_per_batch)

  replay = []

  # Optimizer and train op.
  with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Add normalisation of weights in phis_to_rewards
    if FLAGS.normalisation == "L1":
      w_norm = tf.reduce_sum(tf.abs(phis_to_rewards.w), axis=0, keepdims=True)
    elif FLAGS.normalisation == "L2":
      w_norm = tf.norm(phis_to_rewards.w, axis=0, keepdims=True)
    else:
      raise ValueError("Unknown normlisation_method {}".format(
          FLAGS.normalisation))

    normalise_w = tf.assign(phis_to_rewards.w,
                            phis_to_rewards.w / tf.maximum(w_norm, 1e-6))

  def filter_steps(steps):
    mask = np.sum(np.abs(steps.rewards), axis=-1) > 0.1
    nonzero_inds = np.where(mask)[0]
    zero_inds = np.where(np.logical_not(mask))[0]
    zero_inds = np.random.choice(
        zero_inds, size=len(nonzero_inds), replace=False)
    selected_inds = np.concatenate([nonzero_inds, zero_inds])
    selected_steps = tree.map_structure(lambda x: x[selected_inds], steps)
    return selected_steps, selected_inds

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    step = 0
    while step < FLAGS.num_train_steps:
      step += 1
      steps_output = collect_experience(env, num_episodes=10)
      selected_step_outputs, selected_inds = filter_steps(steps_output)

      if len(replay) > FLAGS.min_replay_size:
        # Do training.
        for _ in range(FLAGS.num_train_repeats):
          train_samples = random.choices(replay, k=128)
          train_samples = tree.map_structure(
              lambda *x: np.stack(x, axis=0), *train_samples)
          train_samples = tree.unflatten_as(steps_ph, train_samples)
          feed_dict = dict(
              zip(tree.flatten(steps_ph), tree.flatten(train_samples)))
          _, train_loss = sess.run([train_op, loss_op], feed_dict=feed_dict)
          sess.run(normalise_w)

        # Do evaluation.
        if step % 50 == 0:
          feed_dict = dict(
              zip(tree.flatten(steps_ph), tree.flatten(selected_step_outputs)))
          eval_loss = sess.run(loss_op, feed_dict=feed_dict)
          logging.info("Step %d,   train loss %f,   eval loss %f,   replay %s",
                       step, train_loss, eval_loss, len(replay))
          print(sess.run(phis_to_rewards.get_variables())[0].T)

          values = dict(step=step, train_loss=train_loss, eval_loss=eval_loss)
          logging.info(values)

      # Add to replay.
      if step <= FLAGS.num_replay_steps:
        def select_fn(ind):
          return lambda x: x[ind]
        for idx in range(len(selected_inds)):
          replay.append(
              tree.flatten(
                  tree.map_structure(select_fn(idx), selected_step_outputs)))

    # Export trained model.
    if FLAGS.export_path:
      model.export(FLAGS.export_path, sess, overwrite=True)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
