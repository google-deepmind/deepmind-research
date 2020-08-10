# Copyright 2020 Deepmind Technologies Limited.
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

"""Metadata for mocap clips that correspond to a walker carrying a prop."""

import collections
import enum
import os

from dm_control.locomotion.mocap import loader as mocap_loader


from catch_carry import trajectories

H5_DIR = os.path.dirname(__file__)
H5_PATH = os.path.join(H5_DIR, 'mocap_data.h5')

IDENTIFIER_PREFIX = 'DeepMindCatchCarry'
IDENTIFIER_TEMPLATE = IDENTIFIER_PREFIX + '-{:03d}'

ClipInfo = collections.namedtuple(
    'ClipInfo', ('clip_identifier', 'num_steps', 'dt', 'flags'))


class Flag(enum.IntEnum):
  BOX = 1 << 0
  BALL = 1 << 1
  LIGHT_PROP = 1 << 2
  HEAVY_PROP = 1 << 3
  SMALL_PROP = 1 << 4
  LARGE_PROP = 1 << 5
  FLOOR_LEVEL = 1 << 6
  MEDIUM_PEDESTAL = 1 << 7
  HIGH_PEDESTAL = 1 << 8


_ALL_CLIPS = None


def _get_clip_info(loader, clip_number, flags):
  clip = loader.get_trajectory(IDENTIFIER_TEMPLATE.format(clip_number))
  return ClipInfo(
      clip_identifier=clip.identifier,
      num_steps=clip.num_steps,
      dt=clip.dt,
      flags=flags)


def _get_all_clip_infos_if_necessary():
  """Creates the global _ALL_CLIPS list if it has not already been created."""
  global _ALL_CLIPS
  if _ALL_CLIPS is None:
    loader = mocap_loader.HDF5TrajectoryLoader(
        H5_PATH, trajectories.WarehouseTrajectory)
    clip_numbers = (1, 2, 3, 4, 5, 6, 9, 10,
                    11, 12, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 33, 34, 35, 36,
                    37, 38, 39, 40, 42, 43, 44, 45,
                    46, 47, 48, 49, 50, 51, 52, 53)

    clip_infos = []
    for i, clip_number in enumerate(clip_numbers):
      flags = 0

      if i in _FLOOR_LEVEL:
        flags |= Flag.FLOOR_LEVEL
      elif i in _MEDIUM_PEDESTAL:
        flags |= Flag.MEDIUM_PEDESTAL
      elif i in _HIGH_PEDESTAL:
        flags |= Flag.HIGH_PEDESTAL

      if i in _LIGHT_PROP:
        flags |= Flag.LIGHT_PROP
      elif i in _HEAVY_PROP:
        flags |= Flag.HEAVY_PROP

      if i in _SMALL_BOX:
        flags |= Flag.SMALL_PROP
        flags |= Flag.BOX
      elif i in _LARGE_BOX:
        flags |= Flag.LARGE_PROP
        flags |= Flag.BOX
      elif i in _SMALL_BALL:
        flags |= Flag.SMALL_PROP
        flags |= Flag.BALL
      elif i in _LARGE_BALL:
        flags |= Flag.LARGE_PROP
        flags |= Flag.BALL
      clip_infos.append(_get_clip_info(loader, clip_number, flags))

    _ALL_CLIPS = tuple(clip_infos)


def _assert_partitions_all_clips(*args):
  """Asserts that a given set of subcollections partitions ALL_CLIPS."""
  sets = tuple(set(arg) for arg in args)

  # Check that the union of all the sets is ALL_CLIPS.
  union = set()
  for subset in sets:
    union = union | set(subset)
  assert union == set(range(48))

  # Check that the sets are pairwise disjoint.
  for i in range(len(sets)):
    for j in range(i + 1, len(sets)):
      assert sets[i] & sets[j] == set()


_FLOOR_LEVEL = tuple(range(0, 16))
_MEDIUM_PEDESTAL = tuple(range(16, 32))
_HIGH_PEDESTAL = tuple(range(32, 48))
_assert_partitions_all_clips(_FLOOR_LEVEL, _MEDIUM_PEDESTAL, _HIGH_PEDESTAL)

_LIGHT_PROP = (0, 1, 2, 3, 8, 9, 12, 13, 16, 17, 18, 19, 24,
               25, 26, 27, 34, 35, 38, 39, 42, 43, 46, 47)
_HEAVY_PROP = (4, 5, 6, 7, 10, 11, 14, 15, 20, 21, 22, 23, 28,
               29, 30, 31, 32, 33, 36, 37, 40, 41, 44, 45)
_assert_partitions_all_clips(_LIGHT_PROP, _HEAVY_PROP)

_SMALL_BOX = (0, 1, 4, 5, 16, 17, 20, 21, 34, 35, 36, 37)
_LARGE_BOX = (2, 3, 6, 7, 18, 19, 22, 23, 32, 33, 38, 39)
_SMALL_BALL = (8, 9, 10, 11, 24, 25, 30, 31, 40, 41, 46, 47)
_LARGE_BALL = (12, 13, 14, 15, 26, 27, 28, 29, 42, 43, 44, 45)
_assert_partitions_all_clips(_SMALL_BOX, _LARGE_BOX, _SMALL_BALL, _LARGE_BALL)


def all_clips():
  _get_all_clip_infos_if_necessary()
  return _ALL_CLIPS


def floor_level():
  clips = all_clips()
  return tuple(clips[i] for i in _FLOOR_LEVEL)


def medium_pedestal():
  clips = all_clips()
  return tuple(clips[i] for i in _MEDIUM_PEDESTAL)


def high_pedestal():
  clips = all_clips()
  return tuple(clips[i] for i in _HIGH_PEDESTAL)


def light_prop():
  clips = all_clips()
  return tuple(clips[i] for i in _LIGHT_PROP)


def heavy_prop():
  clips = all_clips()
  return tuple(clips[i] for i in _HEAVY_PROP)


def small_box():
  clips = all_clips()
  return tuple(clips[i] for i in _SMALL_BOX)


def large_box():
  clips = all_clips()
  return tuple(clips[i] for i in _LARGE_BOX)


def small_ball():
  clips = all_clips()
  return tuple(clips[i] for i in _SMALL_BALL)


def large_ball():
  clips = all_clips()
  return tuple(clips[i] for i in _LARGE_BALL)
