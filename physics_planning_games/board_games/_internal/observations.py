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
# ============================================================================

"""Configuration for observations."""

import collections
import numpy as np


class ObservableSpec(collections.namedtuple(
    'ObservableSpec',
    ['enabled', 'update_interval', 'buffer_size', 'delay', 'aggregator',
     'corruptor'])):
  """Configuration options for generic observables."""
  __slots__ = ()


class CameraObservableSpec(collections.namedtuple(
    'CameraObservableSpec', ('height', 'width') + ObservableSpec._fields)):
  """Configuration options for camera observables."""
  __slots__ = ()


class ObservationSettings(collections.namedtuple(
    'ObservationSettings',
    ['proprio', 'ftt', 'prop_pose', 'board_state', 'camera'])):
  """Container of `ObservableSpecs` grouped by category."""
  __slots__ = ()


class ObservableNames(collections.namedtuple(
    'ObservableNames',
    ['proprio', 'ftt', 'prop_pose', 'board_state', 'camera'])):
  """Container that groups the names of observables by category."""
  __slots__ = ()

  def __new__(cls, proprio=(), ftt=(), prop_pose=(), board_state=(), camera=()):
    return super(ObservableNames, cls).__new__(
        cls,
        proprio=proprio,
        ftt=ftt,
        prop_pose=prop_pose,
        board_state=board_state,
        camera=camera)


# Global defaults for "feature" observables (i.e. anything that isn't a camera).
_DISABLED_FEATURE = ObservableSpec(
    enabled=False,
    update_interval=1,
    buffer_size=1,
    delay=0,
    aggregator=None,
    corruptor=None)
_ENABLED_FEATURE = _DISABLED_FEATURE._replace(enabled=True)

# Force, torque and touch-sensor readings are scaled using a symmetric
# logarithmic transformation that handles 0 and negative values.
_symlog1p = lambda x, random_state: np.sign(x) * np.log1p(abs(x))
_DISABLED_FTT = _DISABLED_FEATURE._replace(corruptor=_symlog1p)
_ENABLED_FTT = _ENABLED_FEATURE._replace(corruptor=_symlog1p)

# Global defaults for camera observables.
_DISABLED_CAMERA = CameraObservableSpec(
    height=84,
    width=84,
    enabled=False,
    update_interval=1,
    buffer_size=1,
    delay=0,
    aggregator=None,
    corruptor=None)
_ENABLED_CAMERA = _DISABLED_CAMERA._replace(enabled=True)

# Predefined sets of configurations options to apply to each category of
# observable.
PERFECT_FEATURES = ObservationSettings(
    proprio=_ENABLED_FEATURE,
    ftt=_ENABLED_FTT,
    prop_pose=_ENABLED_FEATURE,
    board_state=_ENABLED_FEATURE,
    camera=_ENABLED_CAMERA)

ARENA_OBSERVABLES = ObservableNames(camera=['front_camera', 'front_camera_2'])
JACO_ARM_OBSERVABLES = ObservableNames(
    proprio=['joints_pos', 'joints_vel'], ftt=['joints_torque'])
JACO_HAND_OBSERVABLES = ObservableNames(
    proprio=['joints_pos', 'joints_vel', 'pinch_site_pos', 'pinch_site_rmat'])
MARKER_OBSERVABLES = ObservableNames(prop_pose=['position'])


def make_options(obs_settings, obs_names):
  """Constructs a dict of configuration options for a set of named observables.

  Args:
    obs_settings: An `ObservationSettings` instance.
    obs_names: An `ObservableNames` instance.

  Returns:
    A nested dict containing `{observable_name: {option_name: value}}`.
  """
  observable_options = {}
  for category, spec in obs_settings._asdict().items():
    for observable_name in getattr(obs_names, category):
      observable_options[observable_name] = spec._asdict()
  return observable_options
