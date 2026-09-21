# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Atari RL Unplugged datasets.

Examples in the dataset represent SARSA transitions stored during a
DQN training run as described in https://arxiv.org/pdf/1907.04543.

For every training run we have recorded all 50 million transitions corresponding
to 200 million environment steps (4x factor because of frame skipping). There
are 5 separate datasets for each of the 45 games.

Every transition in the dataset is a tuple containing the following features:

* o_t: Observation at time t. Observations have been processed using the
    canonical Atari frame processing, including 4x frame stacking. The shape
    of a single observation is [84, 84, 4].
* a_t: Action taken at time t.
* r_t: Reward after a_t.
* d_t: Discount after a_t.
* o_tp1: Observation at time t+1.
* a_tp1: Action at time t+1.
* extras:
  * episode_id: Episode identifier.
  * episode_return: Total episode return computed using per-step [-1, 1]
      clipping.
"""
import functools
import os
from typing import Dict

from acme import wrappers
import dm_env
from dm_env import specs
from dopamine.discrete_domains import atari_lib
import reverb
import tensorflow as tf


# 9 tuning games.
TUNING_SUITE = [
    'BeamRider',
    'DemonAttack',
    'DoubleDunk',
    'IceHockey',
    'MsPacman',
    'Pooyan',
    'RoadRunner',
    'Robotank',
    'Zaxxon',
]

# 36 testing games.
TESTING_SUITE = [
    'Alien',
    'Amidar',
    'Assault',
    'Asterix',
    'Atlantis',
    'BankHeist',
    'BattleZone',
    'Boxing',
    'Breakout',
    'Carnival',
    'Centipede',
    'ChopperCommand',
    'CrazyClimber',
    'Enduro',
    'FishingDerby',
    'Freeway',
    'Frostbite',
    'Gopher',
    'Gravitar',
    'Hero',
    'Jamesbond',
    'Kangaroo',
    'Krull',
    'KungFuMaster',
    'NameThisGame',
    'Phoenix',
    'Pong',
    'Qbert',
    'Riverraid',
    'Seaquest',
    'SpaceInvaders',
    'StarGunner',
    'TimePilot',
    'UpNDown',
    'VideoPinball',
    'WizardOfWor',
    'YarsRevenge',
]

# Total of 45 games.
ALL = TUNING_SUITE + TESTING_SUITE


def _decode_frames(pngs: tf.Tensor):
  """Decode PNGs.

  Args:
    pngs: String Tensor of size (4,) containing PNG encoded images.

  Returns:
    4 84x84 grayscale images packed in a (84, 84, 4) uint8 Tensor.
  """
  # Statically unroll png decoding
  frames = [tf.image.decode_png(pngs[i], channels=1) for i in range(4)]
  frames = tf.concat(frames, axis=2)
  frames.set_shape((84, 84, 4))
  return frames


def _make_reverb_sample(o_t: tf.Tensor,
                        a_t: tf.Tensor,
                        r_t: tf.Tensor,
                        d_t: tf.Tensor,
                        o_tp1: tf.Tensor,
                        a_tp1: tf.Tensor,
                        extras: Dict[str, tf.Tensor]) -> reverb.ReplaySample:
  """Create Reverb sample with offline data.

  Args:
    o_t: Observation at time t.
    a_t: Action at time t.
    r_t: Reward at time t.
    d_t: Discount at time t.
    o_tp1: Observation at time t+1.
    a_tp1: Action at time t+1.
    extras: Dictionary with extra features.

  Returns:
    Replay sample with fake info: key=0, probability=1, table_size=0.
  """
  info = reverb.SampleInfo(
      key=tf.constant(0, tf.uint64),
      probability=tf.constant(1.0, tf.float64),
      table_size=tf.constant(0, tf.int64),
      priority=tf.constant(1.0, tf.float64),
      times_sampled=tf.constant(1, tf.int32))
  data = (o_t, a_t, r_t, d_t, o_tp1, a_tp1, extras)
  return reverb.ReplaySample(info=info, data=data)


def _tf_example_to_reverb_sample(tf_example: tf.train.Example
                                 ) -> reverb.ReplaySample:
  """Create a Reverb replay sample from a TF example."""

  # Parse tf.Example.
  feature_description = {
      'o_t': tf.io.FixedLenFeature([4], tf.string),
      'o_tp1': tf.io.FixedLenFeature([4], tf.string),
      'a_t': tf.io.FixedLenFeature([], tf.int64),
      'a_tp1': tf.io.FixedLenFeature([], tf.int64),
      'r_t': tf.io.FixedLenFeature([], tf.float32),
      'd_t': tf.io.FixedLenFeature([], tf.float32),
      'episode_id': tf.io.FixedLenFeature([], tf.int64),
      'episode_return': tf.io.FixedLenFeature([], tf.float32),
  }
  data = tf.io.parse_single_example(tf_example, feature_description)

  # Process data.
  o_t = _decode_frames(data['o_t'])
  o_tp1 = _decode_frames(data['o_tp1'])
  a_t = tf.cast(data['a_t'], tf.int32)
  a_tp1 = tf.cast(data['a_tp1'], tf.int32)
  episode_id = tf.bitcast(data['episode_id'], tf.uint64)

  # Build Reverb replay sample.
  extras = {
      'episode_id': episode_id,
      'return': data['episode_return']
  }
  return _make_reverb_sample(o_t, a_t, data['r_t'], data['d_t'], o_tp1, a_tp1,
                             extras)


def dataset(path: str,
            game: str,
            run: int,
            num_shards: int = 100,
            shuffle_buffer_size: int = 100000) -> tf.data.Dataset:
  """TF dataset of Atari SARSA tuples."""
  path = os.path.join(path, f'{game}/run_{run}')
  filenames = [f'{path}-{i:05d}-of-{num_shards:05d}' for i in range(num_shards)]
  file_ds = tf.data.Dataset.from_tensor_slices(filenames)
  file_ds = file_ds.repeat().shuffle(num_shards)
  example_ds = file_ds.interleave(
      functools.partial(tf.data.TFRecordDataset, compression_type='GZIP'),
      cycle_length=tf.data.experimental.AUTOTUNE,
      block_length=5)
  example_ds = example_ds.shuffle(shuffle_buffer_size)
  return example_ds.map(_tf_example_to_reverb_sample,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)


class AtariDopamineWrapper(dm_env.Environment):
  """Wrapper for Atari Dopamine environmnet."""

  def __init__(self, env, max_episode_steps=108000):
    self._env = env
    self._max_episode_steps = max_episode_steps
    self._episode_steps = 0
    self._reset_next_episode = True

  def reset(self):
    self._episode_steps = 0
    self._reset_next_step = False
    observation = self._env.reset()
    return dm_env.restart(observation.squeeze(-1))

  def step(self, action):
    if self._reset_next_step:
      return self.reset()

    observation, reward, terminal, _ = self._env.step(action.item())
    observation = observation.squeeze(-1)
    discount = 1 - float(terminal)
    self._episode_steps += 1
    if terminal:
      self._reset_next_episode = True
      return dm_env.termination(reward, observation)
    elif self._episode_steps == self._max_episode_steps:
      self._reset_next_episode = True
      return dm_env.truncation(reward, observation, discount)
    else:
      return dm_env.transition(reward, observation, discount)

  def observation_spec(self):
    space = self._env.observation_space
    return specs.Array(space.shape[:-1], space.dtype)

  def action_spec(self):
    return specs.DiscreteArray(self._env.action_space.n)


def environment(game: str) -> dm_env.Environment:
  """Atari environment."""
  env = atari_lib.create_atari_environment(game_name=game,
                                           sticky_actions=True)
  env = AtariDopamineWrapper(env)
  env = wrappers.FrameStackingWrapper(env, num_frames=4)
  return wrappers.SinglePrecisionWrapper(env)
