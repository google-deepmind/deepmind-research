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
"""Control RL Unplugged datasets.

Examples in the dataset represent sequences stored when running a partially
trained agent (trained in online way) as described in
https://arxiv.org/abs/2006.13888.

Every dataset has a SARSA version, and datasets for environments for solving
which we believe one may need a recurrent agent also include a version of the
dataset with overlapping sequences of length 40.

Datasets for the dm_control_suite environments only include proprio
observations, while datasets for dm_locomotion include both pixel and proprio
observations.
"""

import collections
import functools
import os
from typing import Dict, Optional, Tuple, Set

from acme import wrappers
from acme.adders import reverb as adders
from dm_control import composer
from dm_control import suite
from dm_control.composer.variation import colors
from dm_control.composer.variation import distributions
from dm_control.locomotion import arenas
from dm_control.locomotion import props
from dm_control.locomotion import tasks
from dm_control.locomotion import walkers
from dm_env import specs
import numpy as np
import reverb
import tensorflow as tf
import tree


def _build_rodent_escape_env():
  """Build environment where a rodent escapes from a bowl."""
  walker = walkers.Rat(
      observable_options={'egocentric_camera': dict(enabled=True)},
  )
  arena = arenas.bowl.Bowl(
      size=(20., 20.),
      aesthetic='outdoor_natural')
  locomotion_task = tasks.escape.Escape(
      walker=walker,
      arena=arena,
      physics_timestep=0.001,
      control_timestep=.02)
  raw_env = composer.Environment(
      time_limit=20,
      task=locomotion_task,
      strip_singleton_obs_buffer_dim=True)

  return raw_env


def _build_rodent_maze_env():
  """Build environment where a rodent runs to targets."""
  walker = walkers.Rat(
      observable_options={'egocentric_camera': dict(enabled=True)},
  )

  wall_textures = arenas.labmaze_textures.WallTextures(
      style='style_01')

  arena = arenas.mazes.RandomMazeWithTargets(
      x_cells=11,
      y_cells=11,
      xy_scale=.5,
      z_height=.3,
      max_rooms=4,
      room_min_size=4,
      room_max_size=5,
      spawns_per_room=1,
      targets_per_room=3,
      wall_textures=wall_textures,
      aesthetic='outdoor_natural')

  rodent_task = tasks.random_goal_maze.ManyGoalsMaze(
      walker=walker,
      maze_arena=arena,
      target_builder=functools.partial(
          props.target_sphere.TargetSphere,
          radius=0.05,
          height_above_ground=.125,
          rgb1=(0, 0, 0.4),
          rgb2=(0, 0, 0.7)),
      target_reward_scale=50.,
      contact_termination=False,
      control_timestep=.02,
      physics_timestep=0.001)
  raw_env = composer.Environment(
      time_limit=30,
      task=rodent_task,
      strip_singleton_obs_buffer_dim=True)

  return raw_env


def _build_rodent_corridor_gaps():
  """Build environment where a rodent runs over gaps."""
  walker = walkers.Rat(
      observable_options={'egocentric_camera': dict(enabled=True)},
  )

  platform_length = distributions.Uniform(low=0.4, high=0.8)
  gap_length = distributions.Uniform(low=0.05, high=0.2)
  arena = arenas.corridors.GapsCorridor(
      corridor_width=2,
      platform_length=platform_length,
      gap_length=gap_length,
      corridor_length=40,
      aesthetic='outdoor_natural')

  rodent_task = tasks.corridors.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_position=(5, 0, 0),
      walker_spawn_rotation=0,
      target_velocity=1.0,
      contact_termination=False,
      terminate_at_height=-0.3,
      physics_timestep=0.001,
      control_timestep=.02)
  raw_env = composer.Environment(
      time_limit=30,
      task=rodent_task,
      strip_singleton_obs_buffer_dim=True)

  return raw_env


def _build_rodent_two_touch_env():
  """Build environment where a rodent touches targets."""
  walker = walkers.Rat(
      observable_options={'egocentric_camera': dict(enabled=True)},
  )

  arena_floor = arenas.floors.Floor(
      size=(10., 10.), aesthetic='outdoor_natural')
  task_reach = tasks.reach.TwoTouch(
      walker=walker,
      arena=arena_floor,
      target_builders=[
          functools.partial(
              props.target_sphere.TargetSphereTwoTouch,
              radius=0.025),
      ],
      randomize_spawn_rotation=True,
      target_type_rewards=[25.],
      shuffle_target_builders=False,
      target_area=(1.5, 1.5),
      physics_timestep=0.001,
      control_timestep=.02)

  raw_env = composer.Environment(
      time_limit=30,
      task=task_reach,
      strip_singleton_obs_buffer_dim=True)

  return raw_env


def _build_humanoid_walls_env():
  """Build humanoid walker walls environment."""
  walker = walkers.CMUHumanoidPositionControlled(
      name='walker',
      observable_options={'egocentric_camera': dict(enabled=True)},
  )
  wall_width = distributions.Uniform(low=1, high=7)
  wall_height = distributions.Uniform(low=2.5, high=4.0)
  swap_wall_side = distributions.Bernoulli(prob=0.5)
  wall_r = distributions.Uniform(low=0.5, high=0.6)
  wall_g = distributions.Uniform(low=0.21, high=0.41)
  wall_rgba = colors.RgbVariation(r=wall_r, g=wall_g, b=0, alpha=1)
  arena = arenas.WallsCorridor(
      wall_gap=5.0,
      wall_width=wall_width,
      wall_height=wall_height,
      swap_wall_side=swap_wall_side,
      wall_rgba=wall_rgba,
      corridor_width=10,
      corridor_length=100)
  humanoid_task = tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_rotation=1.57,  # pi / 2
      physics_timestep=0.005,
      control_timestep=0.03)
  raw_env = composer.Environment(
      time_limit=30,
      task=humanoid_task,
      strip_singleton_obs_buffer_dim=True)

  return raw_env


def _build_humanoid_corridor_env():
  """Build humanoid walker walls environment."""
  walker = walkers.CMUHumanoidPositionControlled(
      name='walker',
      observable_options={'egocentric_camera': dict(enabled=True)},
  )
  arena = arenas.EmptyCorridor(
      corridor_width=10,
      corridor_length=100)
  humanoid_task = tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_rotation=1.57,  # pi / 2
      physics_timestep=0.005,
      control_timestep=0.03)
  raw_env = composer.Environment(
      time_limit=30,
      task=humanoid_task,
      strip_singleton_obs_buffer_dim=True)

  return raw_env


def _build_humanoid_corridor_gaps():
  """Build humanoid walker walls environment."""
  walker = walkers.CMUHumanoidPositionControlled(
      name='walker',
      observable_options={'egocentric_camera': dict(enabled=True)},
  )
  platform_length = distributions.Uniform(low=0.3, high=2.5)
  gap_length = distributions.Uniform(low=0.75, high=1.25)
  arena = arenas.GapsCorridor(
      corridor_width=10,
      platform_length=platform_length,
      gap_length=gap_length,
      corridor_length=100)
  humanoid_task = tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_position=(2, 0, 0),
      walker_spawn_rotation=1.57,  # pi / 2
      physics_timestep=0.005,
      control_timestep=0.03)
  raw_env = composer.Environment(
      time_limit=30,
      task=humanoid_task,
      strip_singleton_obs_buffer_dim=True)

  return raw_env


class MujocoActionNormalizer(wrappers.EnvironmentWrapper):
  """Rescale actions to [-1, 1] range for mujoco physics engine.

  For control environments whose actions have bounded range in [-1, 1], this
    adaptor rescale actions to the desired range. This allows actor network to
    output unscaled actions for better gradient dynamics.
  """

  def __init__(self, environment, rescale='clip'):
    super().__init__(environment)
    self._rescale = rescale

  def step(self, action):
    """Rescale actions to [-1, 1] range before stepping wrapped environment."""
    if self._rescale == 'tanh':
      scaled_actions = tree.map_structure(np.tanh, action)
    elif self._rescale == 'clip':
      scaled_actions = tree.map_structure(lambda a: np.clip(a, -1., 1.), action)
    else:
      raise ValueError('Unrecognized scaling option: %s' % self._rescale)
    return self._environment.step(scaled_actions)


class NormilizeActionSpecWrapper(wrappers.EnvironmentWrapper):
  """Turn each dimension of the actions into the range of [-1, 1]."""

  def __init__(self, environment):
    super().__init__(environment)

    action_spec = environment.action_spec()
    # pytype: disable=attribute-error  # always-use-return-annotations
    self._scale = action_spec.maximum - action_spec.minimum
    self._offset = action_spec.minimum

    minimum = action_spec.minimum * 0 - 1.
    maximum = action_spec.minimum * 0 + 1.
    self._action_spec = specs.BoundedArray(
        action_spec.shape,
        action_spec.dtype,
        minimum,
        maximum,
        name=action_spec.name)
    # pytype: enable=attribute-error  # always-use-return-annotations

  def _from_normal_actions(self, actions):
    actions = 0.5 * (actions + 1.0)  # a_t is now in the range [0, 1]
    # scale range to [minimum, maximum]
    return actions * self._scale + self._offset

  def step(self, action):
    action = self._from_normal_actions(action)
    return self._environment.step(action)

  def action_spec(self):
    return self._action_spec


class FilterObservationsWrapper(wrappers.EnvironmentWrapper):
  """Filter out all the observations not specified to this wrapper."""

  def __init__(self, environment, observations_to_keep):
    super().__init__(environment)
    self._observations_to_keep = observations_to_keep
    spec = self._environment.observation_spec()
    filtered = [(k, spec[k]) for k in observations_to_keep]
    self._observation_spec = collections.OrderedDict(filtered)

  def _filter_observation(self, timestep):
    observation = timestep.observation
    filtered = [(k, observation[k]) for k in self._observations_to_keep]
    return timestep._replace(observation=collections.OrderedDict(filtered))

  def step(self, action):
    return self._filter_observation(self._environment.step(action))

  def reset(self):
    return self._filter_observation(self._environment.reset())

  def observation_spec(self):
    return self._observation_spec


class ControlSuite:
  """Create bits needed to run agents on an Control Suite dataset."""

  def __init__(self, task_name='humanoid_run'):
    """Initializes datasets/environments for the Deepmind Control suite.

    Args:
      task_name: take name. Must be one of,
        finger_turn_hard, manipulator_insert_peg, humanoid_run,
        cartpole_swingup, cheetah_run, fish_swim, manipulator_insert_ball,
        walker_stand, walker_walk
    """
    self.task_name = task_name
    self._uint8_features = set([])
    self._environment = None

    if task_name == 'swim':
      self._domain_name = 'fish'
      self._task_name = 'swim'

      self._shapes = {
          'observation/target': (3,),
          'observation/velocity': (13,),
          'observation/upright': (1,),
          'observation/joint_angles': (7,),
          'action': (5,),
          'discount': (),
          'reward': (),
          'episodic_reward': (),
          'step_type': ()
      }
    elif task_name == 'humanoid_run':
      self._domain_name = 'humanoid'
      self._task_name = 'run'

      self._shapes = {
          'observation/velocity': (27,),
          'observation/com_velocity': (3,),
          'observation/torso_vertical': (3,),
          'observation/extremities': (12,),
          'observation/head_height': (1,),
          'observation/joint_angles': (21,),
          'action': (21,),
          'discount': (),
          'reward': (),
          'episodic_reward': (),
          'step_type': ()
      }
    elif task_name == 'manipulator_insert_ball':
      self._domain_name = 'manipulator'
      self._task_name = 'insert_ball'
      self._shapes = {
          'observation/arm_pos': (16,),
          'observation/arm_vel': (8,),
          'observation/touch': (5,),
          'observation/hand_pos': (4,),
          'observation/object_pos': (4,),
          'observation/object_vel': (3,),
          'observation/target_pos': (4,),
          'action': (5,),
          'discount': (),
          'reward': (),
          'episodic_reward': (),
          'step_type': ()}
    elif task_name == 'manipulator_insert_peg':
      self._domain_name = 'manipulator'
      self._task_name = 'insert_peg'
      self._shapes = {
          'observation/arm_pos': (16,),
          'observation/arm_vel': (8,),
          'observation/touch': (5,),
          'observation/hand_pos': (4,),
          'observation/object_pos': (4,),
          'observation/object_vel': (3,),
          'observation/target_pos': (4,),
          'episodic_reward': (),
          'action': (5,),
          'discount': (),
          'reward': (),
          'step_type': ()}
    elif task_name == 'cartpole_swingup':
      self._domain_name = 'cartpole'
      self._task_name = 'swingup'
      self._shapes = {
          'observation/position': (3,),
          'observation/velocity': (2,),
          'action': (1,),
          'discount': (),
          'reward': (),
          'episodic_reward': (),
          'step_type': ()}
    elif task_name == 'walker_walk':
      self._domain_name = 'walker'
      self._task_name = 'walk'
      self._shapes = {
          'observation/orientations': (14,),
          'observation/velocity': (9,),
          'observation/height': (1,),
          'action': (6,),
          'discount': (),
          'reward': (),
          'episodic_reward': (),
          'step_type': ()}
    elif task_name == 'walker_stand':
      self._domain_name = 'walker'
      self._task_name = 'stand'
      self._shapes = {
          'observation/orientations': (14,),
          'observation/velocity': (9,),
          'observation/height': (1,),
          'action': (6,),
          'discount': (),
          'reward': (),
          'episodic_reward': (),
          'step_type': ()}
    elif task_name == 'cheetah_run':
      self._domain_name = 'cheetah'
      self._task_name = 'run'
      self._shapes = {
          'observation/position': (8,),
          'observation/velocity': (9,),
          'action': (6,),
          'discount': (),
          'reward': (),
          'episodic_reward': (),
          'step_type': ()}
    elif task_name == 'finger_turn_hard':
      self._domain_name = 'finger'
      self._task_name = 'turn_hard'
      self._shapes = {
          'observation/position': (4,),
          'observation/velocity': (3,),
          'observation/touch': (2,),
          'observation/target_position': (2,),
          'observation/dist_to_target': (1,),
          'action': (2,),
          'discount': (),
          'reward': (),
          'episodic_reward': (),
          'step_type': ()}
    else:
      raise ValueError('Task \'{}\' not found.'.format(task_name))

    self._data_path = 'dm_control_suite/{}/train'.format(task_name)

  @property
  def shapes(self):
    return self._shapes

  @property
  def data_path(self):
    return self._data_path

  @property
  def uint8_features(self):
    return self._uint8_features

  @property
  def environment(self):
    """Build and return the environment."""
    if self._environment is not None:
      return self._environment

    self._environment = suite.load(
        domain_name=self._domain_name,
        task_name=self._task_name)

    self._environment = wrappers.SinglePrecisionWrapper(self._environment)
    self._environment = NormilizeActionSpecWrapper(self._environment)

    return self._environment


class CmuThirdParty:
  """Create bits needed to run agents on an locomotion humanoid dataset."""

  def __init__(self, task_name='humanoid_walls'):
    # 'humanoid_corridor|humanoid_gaps|humanoid_walls'
    self._task_name = task_name
    self._pixel_keys = self.get_pixel_keys()
    self._uint8_features = set(['observation/walker/egocentric_camera'])
    self.additional_paths = {}
    self._proprio_keys = [
        'walker/joints_vel',
        'walker/sensors_velocimeter',
        'walker/sensors_gyro',
        'walker/joints_pos',
        'walker/world_zaxis',
        'walker/body_height',
        'walker/sensors_accelerometer',
        'walker/end_effectors_pos'
    ]

    self._shapes = {
        'observation/walker/joints_vel': (56,),
        'observation/walker/sensors_velocimeter': (3,),
        'observation/walker/sensors_gyro': (3,),
        'observation/walker/joints_pos': (56,),
        'observation/walker/world_zaxis': (3,),
        'observation/walker/body_height': (1,),
        'observation/walker/sensors_accelerometer': (3,),
        'observation/walker/end_effectors_pos': (12,),
        'observation/walker/egocentric_camera': (
            64,
            64,
            3,
        ),
        'action': (56,),
        'discount': (),
        'reward': (),
        'episodic_reward': (),
        'step_type': ()
    }

    if task_name == 'humanoid_corridor':
      self._data_path = 'dm_locomotion/humanoid_corridor/seq2/train'
    elif task_name == 'humanoid_gaps':
      self._data_path = 'dm_locomotion/humanoid_gaps/seq2/train'
    elif task_name == 'humanoid_walls':
      self._data_path = 'dm_locomotion/humanoid_walls/seq40/train'
    else:
      raise ValueError('Task \'{}\' not found.'.format(task_name))

  @staticmethod
  def get_pixel_keys():
    return ('walker/egocentric_camera',)

  @property
  def uint8_features(self):
    return self._uint8_features

  @property
  def shapes(self):
    return self._shapes

  @property
  def data_path(self):
    return self._data_path

  @property
  def environment(self):
    """Build and return the environment."""

    if self._task_name == 'humanoid_corridor':
      self._environment = _build_humanoid_corridor_env()
    elif self._task_name == 'humanoid_gaps':
      self._environment = _build_humanoid_corridor_gaps()
    elif self._task_name == 'humanoid_walls':
      self._environment = _build_humanoid_walls_env()

    self._environment = NormilizeActionSpecWrapper(self._environment)
    self._environment = MujocoActionNormalizer(
        environment=self._environment, rescale='clip')
    self._environment = wrappers.SinglePrecisionWrapper(self._environment)

    all_observations = list(self._proprio_keys) + list(self._pixel_keys)
    self._environment = FilterObservationsWrapper(self._environment,
                                                  all_observations)

    return self._environment


class Rodent:
  """Create bits needed to run agents on an Rodent dataset."""

  def __init__(self, task_name='rodent_gaps'):
    # 'rodent_escape|rodent_two_touch|rodent_gaps|rodent_mazes'
    self._task_name = task_name
    self._pixel_keys = self.get_pixel_keys()
    self._uint8_features = set(['observation/walker/egocentric_camera'])

    self._proprio_keys = [
        'walker/joints_pos', 'walker/joints_vel', 'walker/tendons_pos',
        'walker/tendons_vel', 'walker/appendages_pos', 'walker/world_zaxis',
        'walker/sensors_accelerometer', 'walker/sensors_velocimeter',
        'walker/sensors_gyro', 'walker/sensors_touch',
    ]

    self._shapes = {
        'observation/walker/joints_pos': (30,),
        'observation/walker/joints_vel': (30,),
        'observation/walker/tendons_pos': (8,),
        'observation/walker/tendons_vel': (8,),
        'observation/walker/appendages_pos': (15,),
        'observation/walker/world_zaxis': (3,),
        'observation/walker/sensors_accelerometer': (3,),
        'observation/walker/sensors_velocimeter': (3,),
        'observation/walker/sensors_gyro': (3,),
        'observation/walker/sensors_touch': (4,),
        'observation/walker/egocentric_camera': (64, 64, 3),
        'action': (38,),
        'discount': (),
        'reward': (),
        'step_type': ()
    }

    if task_name == 'rodent_gaps':
      self._data_path = 'dm_locomotion/rodent_gaps/seq2/train'
    elif task_name == 'rodent_escape':
      self._data_path = 'dm_locomotion/rodent_bowl_escape/seq2/train'
    elif task_name == 'rodent_two_touch':
      self._data_path = 'dm_locomotion/rodent_two_touch/seq40/train'
    elif task_name == 'rodent_mazes':
      self._data_path = 'dm_locomotion/rodent_mazes/seq40/train'
    else:
      raise ValueError('Task \'{}\' not found.'.format(task_name))

  @staticmethod
  def get_pixel_keys():
    return ('walker/egocentric_camera',)

  @property
  def shapes(self):
    return self._shapes

  @property
  def uint8_features(self):
    return self._uint8_features

  @property
  def data_path(self):
    return self._data_path

  @property
  def environment(self):
    """Return environment."""
    if self._task_name == 'rodent_escape':
      self._environment = _build_rodent_escape_env()
    elif self._task_name == 'rodent_gaps':
      self._environment = _build_rodent_corridor_gaps()
    elif self._task_name == 'rodent_two_touch':
      self._environment = _build_rodent_two_touch_env()
    elif self._task_name == 'rodent_mazes':
      self._environment = _build_rodent_maze_env()

    self._environment = NormilizeActionSpecWrapper(self._environment)
    self._environment = MujocoActionNormalizer(
        environment=self._environment, rescale='clip')
    self._environment = wrappers.SinglePrecisionWrapper(self._environment)

    all_observations = list(self._proprio_keys) + list(self._pixel_keys)
    self._environment = FilterObservationsWrapper(self._environment,
                                                  all_observations)

    return self._environment


def _parse_seq_tf_example(example, uint8_features, shapes):
  """Parse tf.Example containing one or two episode steps."""
  def to_feature(key, shape):
    if key in uint8_features:
      return tf.io.FixedLenSequenceFeature(
          shape=[], dtype=tf.string, allow_missing=True)
    else:
      return tf.io.FixedLenSequenceFeature(
          shape=shape, dtype=tf.float32, allow_missing=True)

  feature_map = {}
  for k, v in shapes.items():
    feature_map[k] = to_feature(k, v)

  parsed = tf.io.parse_single_example(example, features=feature_map)

  observation = {}
  restructured = {}
  for k in parsed.keys():
    if 'observation' not in k:
      restructured[k] = parsed[k]
      continue

    if k in uint8_features:
      observation[k.replace('observation/', '')] = tf.reshape(
          tf.io.decode_raw(parsed[k], out_type=tf.uint8), (-1,) + shapes[k])
    else:
      observation[k.replace('observation/', '')] = parsed[k]

  restructured['observation'] = observation

  restructured['length'] = tf.shape(restructured['action'])[0]

  return restructured


def _build_sequence_example(sequences):
  """Convert raw sequences into a Reverb sequence sample."""
  data = adders.Step(
      observation=sequences['observation'],
      action=sequences['action'],
      reward=sequences['reward'],
      discount=sequences['discount'],
      start_of_episode=(),
      extras=())

  info = reverb.SampleInfo(
      key=tf.constant(0, tf.uint64),
      probability=tf.constant(1.0, tf.float64),
      table_size=tf.constant(0, tf.int64),
      priority=tf.constant(1.0, tf.float64),
      times_sampled=tf.constant(1.0, tf.int32))
  return reverb.ReplaySample(info=info, data=data)


def _build_sarsa_example(sequences):
  """Convert raw sequences into a Reverb n-step SARSA sample."""

  o_tm1 = tree.map_structure(lambda t: t[0], sequences['observation'])
  o_t = tree.map_structure(lambda t: t[1], sequences['observation'])
  a_tm1 = tree.map_structure(lambda t: t[0], sequences['action'])
  a_t = tree.map_structure(lambda t: t[1], sequences['action'])
  r_t = tree.map_structure(lambda t: t[0], sequences['reward'])
  p_t = tree.map_structure(lambda t: t[0], sequences['discount'])

  info = reverb.SampleInfo(
      key=tf.constant(0, tf.uint64),
      probability=tf.constant(1.0, tf.float64),
      table_size=tf.constant(0, tf.int64),
      priority=tf.constant(1.0, tf.float64),
      times_sampled=tf.constant(1.0, tf.int32))
  return reverb.ReplaySample(info=info, data=(o_tm1, a_tm1, r_t, p_t, o_t, a_t))


def _padded_batch(example_ds, batch_size, shapes, drop_remainder=False):
  """Batch data while handling unequal lengths."""
  padded_shapes = {}
  padded_shapes['observation'] = {}
  for k, v in shapes.items():
    if 'observation' in k:
      padded_shapes['observation'][
          k.replace('observation/', '')] = (-1,) + v
    else:
      padded_shapes[k] = (-1,) + v

  padded_shapes['length'] = ()

  return example_ds.padded_batch(batch_size,
                                 padded_shapes=padded_shapes,
                                 drop_remainder=drop_remainder)


def dataset(root_path: str,
            data_path: str,
            shapes: Dict[str, Tuple[int]],
            num_threads: int,
            batch_size: int,
            uint8_features: Optional[Set[str]] = None,
            num_shards: int = 100,
            shuffle_buffer_size: int = 100000,
            sarsa: bool = True) -> tf.data.Dataset:
  """Create tf dataset for training."""

  uint8_features = uint8_features if uint8_features else {}
  path = os.path.join(root_path, data_path)

  filenames = [f'{path}-{i:05d}-of-{num_shards:05d}' for i in range(num_shards)]
  file_ds = tf.data.Dataset.from_tensor_slices(filenames)
  file_ds = file_ds.repeat().shuffle(num_shards)

  example_ds = file_ds.interleave(
      functools.partial(tf.data.TFRecordDataset, compression_type='GZIP'),
      cycle_length=tf.data.experimental.AUTOTUNE,
      block_length=5)
  example_ds = example_ds.shuffle(shuffle_buffer_size)

  def map_func(example):
    example = _parse_seq_tf_example(example, uint8_features, shapes)
    return example
  example_ds = example_ds.map(map_func, num_parallel_calls=num_threads)
  example_ds = example_ds.repeat().shuffle(batch_size * 10)

  if sarsa:
    example_ds = example_ds.map(
        _build_sarsa_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    example_ds.batch(batch_size)
  else:
    example_ds = _padded_batch(
        example_ds, batch_size, shapes, drop_remainder=True)

    example_ds = example_ds.map(
        _build_sequence_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  example_ds = example_ds.prefetch(tf.data.experimental.AUTOTUNE)

  return example_ds
