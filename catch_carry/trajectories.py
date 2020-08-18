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

"""Mocap trajectory that assumes props start stationary on pedestals."""

import copy
import enum
import itertools

from dm_control.locomotion.mocap import mocap_pb2
from dm_control.locomotion.mocap import trajectory
from dm_control.utils import transformations
import numpy as np

_PEDESTAL_SIZE = (0.2, 0.2, 0.02)
_MAX_SETTLE_STEPS = 100


@enum.unique
class ClipSegment(enum.Enum):
  """Annotations for subsegments within a warehouse clips."""

  # Clip segment corresponding to a walker approaching an object
  APPROACH = 1

  # Clip segment corresponding to a walker picking up an object.
  PICKUP = 2

  # Clip segment corresponding to the "first half" of the walker carrying an
  # object, beginning from the walker backing away from a pedestal with
  # object in hand.
  CARRY1 = 3

  # Clip segment corresponding to the "second half" of the walker carrying an
  # object, ending in the walker approaching a pedestal the object in hand.
  CARRY2 = 4

  # Clip segment corresponding to a walker putting down an object on a pedestal.
  PUTDOWN = 5

  # Clip segment corresponding to a walker backing off after successfully
  # placing an object on a pedestal.
  BACKOFF = 6


def _get_rotated_bounding_box(size, quaternion):
  """Calculates the bounding box of a rotated 3D box.

  Args:
    size: An array of length 3 specifying the half-lengths of a box.
    quaternion: A unit quaternion specifying the box's orientation.

  Returns:
    An array of length 3 specifying the half-lengths of the bounding box of
    the rotated box.
  """
  corners = ((size[0], size[1], size[2]),
             (size[0], size[1], -size[2]),
             (size[0], -size[1], size[2]),
             (-size[0], size[1], size[2]))
  rotated_corners = tuple(
      transformations.quat_rotate(quaternion, corner) for corner in corners)
  return np.amax(np.abs(rotated_corners), axis=0)


def _get_prop_z_extent(prop_proto, quaternion):
  """Calculates the "z-extent" of the prop in given orientation.

  This is the distance from the centre of the prop to its lowest point in the
  world frame, taking into account the prop's orientation.

  Args:
    prop_proto: A `mocap_pb2.Prop` protocol buffer defining a prop.
    quaternion: A unit quaternion specifying the prop's orientation.

  Returns:
    the distance from the centre of the prop to its lowest point in the
    world frame in the specified orientation.
  """
  if prop_proto.shape == mocap_pb2.Prop.BOX:
    return _get_rotated_bounding_box(prop_proto.size, quaternion)[2]
  elif prop_proto.shape == mocap_pb2.Prop.SPHERE:
    return prop_proto.size[0]
  else:
    raise NotImplementedError(
        'Unsupported prop shape: {}'.format(prop_proto.shape))


class WarehouseTrajectory(trajectory.Trajectory):
  """Mocap trajectory that assumes props start stationary on pedestals."""

  def infer_pedestal_positions(self, num_averaged_steps=30,
                               ground_height_tolerance=0.1,
                               proto_modifier=None):
    proto = self._proto
    if proto_modifier is not None:
      proto = copy.copy(proto)
      proto_modifier(proto)

    if not proto.props:
      return []

    positions = []
    for timestep in itertools.islice(proto.timesteps, num_averaged_steps):
      positions_for_timestep = []
      for prop_proto, prop_timestep in zip(proto.props, timestep.props):
        z_extent = _get_prop_z_extent(prop_proto, prop_timestep.quaternion)
        positions_for_timestep.append([prop_timestep.position[0],
                                       prop_timestep.position[1],
                                       prop_timestep.position[2] - z_extent])
      positions.append(positions_for_timestep)

    median_positions = np.median(positions, axis=0)
    median_positions[:, 2][median_positions[:, 2] < ground_height_tolerance] = 0
    return median_positions

  def get_props_z_extent(self, physics):
    timestep = self._proto.timesteps[self._get_step_id(physics.time())]
    out = []
    for prop_proto, prop_timestep in zip(self._proto.props, timestep.props):
      z_extent = _get_prop_z_extent(prop_proto, prop_timestep.quaternion)
      out.append(z_extent)
    return out


class SinglePropCarrySegmentedTrajectory(WarehouseTrajectory):
  """A mocap trajectory class that automatically segments prop-carry clips.

  The algorithm implemented in the class only works if the trajectory consists
  of exactly one walker and one prop. The value of `pedestal_zone_distance`
  the exact nature of zone crossings are determined empirically from the
  DeepMindCatchCarry dataset, and are likely to not work well outside of this
  setting.
  """

  def __init__(self,
               proto,
               start_time=None,
               end_time=None,
               pedestal_zone_distance=0.65,
               start_step=None,
               end_step=None,
               zero_out_velocities=True):
    super(SinglePropCarrySegmentedTrajectory, self).__init__(
        proto, start_time, end_time, start_step=start_step, end_step=end_step,
        zero_out_velocities=zero_out_velocities)
    self._pedestal_zone_distance = pedestal_zone_distance
    self._generate_segments()

  def _generate_segments(self):
    pedestal_position = self.infer_pedestal_positions()[0]

    # First we find the timesteps at which the walker cross the pedestal's
    # vicinity zone. This should happen exactly 4 times: enter it to pick up,
    # leave it, enter it again to put down, and leave it again.
    was_in_pedestal_zone = False
    crossings = []
    for i, timestep in enumerate(self._proto.timesteps):
      pedestal_dist = np.linalg.norm(
          timestep.walkers[0].position[:2] - pedestal_position[:2])
      if pedestal_dist > self._pedestal_zone_distance and was_in_pedestal_zone:
        crossings.append(i)
        was_in_pedestal_zone = False
      elif (pedestal_dist <= self._pedestal_zone_distance and
            not was_in_pedestal_zone):
        crossings.append(i)
        was_in_pedestal_zone = True
    if len(crossings) < 3:
      raise RuntimeError(
          'Failed to segment the given trajectory: '
          'walker should cross the pedestal zone\'s boundary >= 3 times '
          'but got {}'.format(len(crossings)))
    elif len(crossings) == 3:
      crossings.append(len(self._proto.timesteps) - 1)
    elif len(crossings) > 4:
      crossings = [crossings[0], crossings[1], crossings[-2], crossings[-1]]

    # Identify the pick up event during the first in-zone interval.
    start_position = np.array(self._proto.timesteps[0].props[0].position)
    end_position = np.array(self._proto.timesteps[-1].props[0].position)
    pick_up_step = crossings[1] - 1
    while pick_up_step > crossings[0]:
      prev_position = self._proto.timesteps[pick_up_step - 1].props[0].position
      if np.linalg.norm(start_position[2] - prev_position[2]) < 0.001:
        break
      pick_up_step -= 1

    # Identify the put down event during the second in-zone interval.
    put_down_step = crossings[2]
    while put_down_step <= crossings[3]:
      next_position = self._proto.timesteps[put_down_step + 1].props[0].position
      if np.linalg.norm(end_position[2] - next_position[2]) < 0.001:
        break
      put_down_step += 1

    carry_halfway_step = int((crossings[1] + crossings[2]) / 2)

    self._segment_intervals = {
        ClipSegment.APPROACH: (0, crossings[0]),
        ClipSegment.PICKUP: (crossings[0], pick_up_step),
        ClipSegment.CARRY1: (pick_up_step, carry_halfway_step),
        ClipSegment.CARRY2: (carry_halfway_step, crossings[2]),
        ClipSegment.PUTDOWN: (crossings[2], put_down_step),
        ClipSegment.BACKOFF: (put_down_step, len(self._proto.timesteps))
    }

  def segment_interval(self, segment):
    start_step, end_step = self._segment_intervals[segment]
    return (start_step * self._proto.dt, (end_step - 1) * self._proto.dt)

  def get_random_timestep_in_segment(self, segment, random_step):
    return self._proto.timesteps[
        random_step.randint(*self._segment_intervals[segment])]

