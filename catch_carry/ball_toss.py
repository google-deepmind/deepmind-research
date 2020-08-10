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

"""A ball-tossing task."""

import collections

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.mocap import loader as mocap_loader
import numpy as np

from catch_carry import mocap_data
from catch_carry import props
from catch_carry import trajectories

_PHYSICS_TIMESTEP = 0.005

_BUCKET_SIZE = (0.2, 0.2, 0.02)

# Magnitude of the sparse reward.
_SPARSE_REWARD = 1.0


class BallToss(composer.Task):
  """A task involving catching and throwing a ball."""

  def __init__(self, walker,
               proto_modifier=None,
               negative_reward_on_failure_termination=True,
               priority_friction=False,
               bucket_offset=1.,
               y_range=0.5,
               toss_delay=0.5,
               randomize_init=False,
              ):
    """Initialize ball tossing task.

    Args:
      walker: the walker to be used in this task.
      proto_modifier: function to modify trajectory proto.
      negative_reward_on_failure_termination: flag to provide negative reward
        as task fails.
      priority_friction: sets friction priority thereby making prop objects have
        higher friction.
      bucket_offset: distance in meters to push bucket (away from walker)
      y_range: range (uniformly sampled) of distance in meters the ball is
        thrown left/right of the walker.
      toss_delay: time in seconds to delay after catching before changing reward
        to encourage throwing the ball.
      randomize_init: flag to randomize initial pose.
    """
    self._proto_modifier = proto_modifier
    self._negative_reward_on_failure_termination = (
        negative_reward_on_failure_termination)
    self._priority_friction = priority_friction
    self._bucket_rewarded = False
    self._bucket_offset = bucket_offset
    self._y_range = y_range
    self._toss_delay = toss_delay
    self._randomize_init = randomize_init

    # load a clip to grab a ball prop and initializations
    loader = mocap_loader.HDF5TrajectoryLoader(
        mocap_data.H5_PATH, trajectories.WarehouseTrajectory)
    clip_number = 54
    self._trajectory = loader.get_trajectory(
        mocap_data.IDENTIFIER_TEMPLATE.format(clip_number))

    # create the floor arena
    self._arena = floors.Floor()

    self._walker = walker
    self._walker_geoms = tuple(self._walker.mjcf_model.find_all('geom'))
    self._feet_geoms = (
        walker.mjcf_model.find('body', 'lfoot').find_all('geom') +
        walker.mjcf_model.find('body', 'rfoot').find_all('geom'))
    self._lhand_geoms = (
        walker.mjcf_model.find('body', 'lhand').find_all('geom'))
    self._rhand_geoms = (
        walker.mjcf_model.find('body', 'rhand').find_all('geom'))

    # resize the humanoid based on the motion capture data subject
    self._trajectory.configure_walkers([self._walker])
    walker.create_root_joints(self._arena.attach(walker))

    control_timestep = self._trajectory.dt
    self.set_timesteps(control_timestep, _PHYSICS_TIMESTEP)

    # build and attach the bucket to the arena
    self._bucket = props.Bucket(_BUCKET_SIZE)
    self._arena.attach(self._bucket)

    self._prop = self._trajectory.create_props(
        priority_friction=self._priority_friction)[0]
    self._arena.add_free_entity(self._prop)

    self._task_observables = collections.OrderedDict()

    # define feature based observations (agent may or may not use these)
    def ego_prop_xpos(physics):
      prop_xpos, _ = self._prop.get_pose(physics)
      walker_xpos = physics.bind(self._walker.root_body).xpos
      return self._walker.transform_vec_to_egocentric_frame(
          physics, prop_xpos - walker_xpos)
    self._task_observables['prop_{}/xpos'.format(0)] = (
        observable.Generic(ego_prop_xpos))

    def prop_zaxis(physics):
      prop_xmat = physics.bind(
          mjcf.get_attachment_frame(self._prop.mjcf_model)).xmat
      return prop_xmat[[2, 5, 8]]
    self._task_observables['prop_{}/zaxis'.format(0)] = (
        observable.Generic(prop_zaxis))

    def ego_bucket_xpos(physics):
      bucket_xpos, _ = self._bucket.get_pose(physics)
      walker_xpos = physics.bind(self._walker.root_body).xpos
      return self._walker.transform_vec_to_egocentric_frame(
          physics, bucket_xpos - walker_xpos)
    self._task_observables['bucket_{}/xpos'.format(0)] = (
        observable.Generic(ego_bucket_xpos))

    for obs in (self._walker.observables.proprioception +
                self._walker.observables.kinematic_sensors +
                self._walker.observables.dynamic_sensors +
                list(self._task_observables.values())):
      obs.enabled = True

  @property
  def root_entity(self):
    return self._arena

  @property
  def task_observables(self):
    return self._task_observables

  @property
  def name(self):
    return 'ball_toss'

  def initialize_episode_mjcf(self, random_state):
    self._reward = 0.0
    self._discount = 1.0
    self._should_terminate = False

    self._prop.detach()

    if self._proto_modifier:
      trajectory = self._trajectory.get_modified_trajectory(
          self._proto_modifier)

    self._prop = trajectory.create_props(
        priority_friction=self._priority_friction)[0]
    self._arena.add_free_entity(self._prop)

    # set the bucket position for this episode
    bucket_distance = 1.*random_state.rand()+self._bucket_offset
    mjcf.get_attachment_frame(self._bucket.mjcf_model).pos = [bucket_distance,
                                                              0, 0]

  def initialize_episode(self, physics, random_state):
    self._ground_geomid = physics.bind(
        self._arena.mjcf_model.worldbody.geom[0]).element_id
    self._feet_geomids = set(physics.bind(self._feet_geoms).element_id)
    self._lhand_geomids = set(physics.bind(self._lhand_geoms).element_id)
    self._rhand_geomids = set(physics.bind(self._rhand_geoms).element_id)
    self._walker_geomids = set(physics.bind(self._walker_geoms).element_id)
    self._bucket_rewarded = False

    if self._randomize_init:
      timestep_ind = random_state.randint(
          len(self._trajectory._proto.timesteps))  # pylint: disable=protected-access
    else:
      timestep_ind = 0
    walker_init_timestep = self._trajectory._proto.timesteps[timestep_ind]  # pylint: disable=protected-access
    prop_init_timestep = self._trajectory._proto.timesteps[0]  # pylint: disable=protected-access

    self._walker.set_pose(
        physics,
        position=walker_init_timestep.walkers[0].position,
        quaternion=walker_init_timestep.walkers[0].quaternion)
    self._walker.set_velocity(
        physics, velocity=walker_init_timestep.walkers[0].velocity,
        angular_velocity=walker_init_timestep.walkers[0].angular_velocity)
    physics.bind(self._walker.mocap_joints).qpos = (
        walker_init_timestep.walkers[0].joints)
    physics.bind(self._walker.mocap_joints).qvel = (
        walker_init_timestep.walkers[0].joints_velocity)

    initial_prop_pos = np.copy(prop_init_timestep.props[0].position)
    initial_prop_pos[0] += 1.  # move ball (from mocap) relative to origin
    initial_prop_pos[1] = 0  # align ball with walker along y-axis
    self._prop.set_pose(
        physics,
        position=initial_prop_pos,
        quaternion=prop_init_timestep.props[0].quaternion)

    # specify the distributions of ball velocity componentwise
    x_vel_mag = 4.5*random_state.rand()+1.5  # m/s
    x_dist = 3  # approximate initial distance from walker to ball
    self._t_dist = x_dist/x_vel_mag  # target time at which to hit the humanoid
    z_offset = .4*random_state.rand()+.1  # height at which to hit person
    # compute velocity to satisfy desired projectile trajectory
    z_vel_mag = (4.9*(self._t_dist**2) + z_offset)/self._t_dist

    y_range = variation.evaluate(self._y_range, random_state=random_state)
    y_vel_mag = y_range*random_state.rand()-y_range/2
    trans_vel = [-x_vel_mag, y_vel_mag, z_vel_mag]
    ang_vel = 1.5*random_state.rand(3)-0.75
    self._prop.set_velocity(
        physics,
        velocity=trans_vel,
        angular_velocity=ang_vel)

  def after_step(self, physics, random_state):
    # First we check for failure termination (walker or ball touches ground).
    ground_failure = False
    for contact in physics.data.contact:
      if ((contact.geom1 == self._ground_geomid and
           contact.geom2 not in self._feet_geomids) or
          (contact.geom2 == self._ground_geomid and
           contact.geom1 not in self._feet_geomids)):
        ground_failure = True
        break

    contact_features = self._evaluate_contacts(physics)
    prop_lhand, prop_rhand, bucket_prop, bucket_walker, walker_prop = contact_features

    # or also fail if walker hits bucket
    if ground_failure or bucket_walker:
      if self._negative_reward_on_failure_termination:
        self._reward = -_SPARSE_REWARD
      else:
        self._reward = 0.0
      self._should_terminate = True
      self._discount = 0.0
      return

    self._reward = 0.0
    # give reward if prop is in bucket (prop touching bottom surface of bucket)
    if bucket_prop:
      self._reward += _SPARSE_REWARD/10

    # shaping reward for being closer to bucket
    if physics.data.time > (self._t_dist + self._toss_delay):
      bucket_xy = physics.bind(self._bucket.geom).xpos[0][:2]
      prop_xy = self._prop.get_pose(physics)[0][:2]
      xy_dist = np.sum(np.array(np.abs(bucket_xy - prop_xy)))
      self._reward += np.exp(-xy_dist/3.)*_SPARSE_REWARD/50
    else:
      # bonus for hands touching ball
      if prop_lhand:
        self._reward += _SPARSE_REWARD/100
      if prop_rhand:
        self._reward += _SPARSE_REWARD/100
      # combined with penalty for other body parts touching the ball
      if walker_prop:
        self._reward -= _SPARSE_REWARD/100

  def get_reward(self, physics):
    return self._reward

  def get_discount(self, physics):
    return self._discount

  def should_terminate_episode(self, physics):
    return self._should_terminate

  def _evaluate_contacts(self, physics):
    prop_elem_id = physics.bind(self._prop.geom).element_id
    bucket_bottom_elem_id = physics.bind(self._bucket.geom[0]).element_id
    bucket_any_elem_id = set(physics.bind(self._bucket.geom).element_id)
    prop_lhand_contact = False
    prop_rhand_contact = False
    bucket_prop_contact = False
    bucket_walker_contact = False
    walker_prop_contact = False

    for contact in physics.data.contact:
      has_prop = (contact.geom1 == prop_elem_id or
                  contact.geom2 == prop_elem_id)
      has_bucket_bottom = (contact.geom1 == bucket_bottom_elem_id or
                           contact.geom2 == bucket_bottom_elem_id)
      has_bucket_any = (contact.geom1 in bucket_any_elem_id or
                        contact.geom2 in bucket_any_elem_id)
      has_lhand = (contact.geom1 in self._lhand_geomids or
                   contact.geom2 in self._lhand_geomids)
      has_rhand = (contact.geom1 in self._rhand_geomids or
                   contact.geom2 in self._rhand_geomids)
      has_walker = (contact.geom1 in self._walker_geomids or
                    contact.geom2 in self._walker_geomids)
      if has_prop and has_bucket_bottom:
        bucket_prop_contact = True
      if has_walker and has_bucket_any:
        bucket_walker_contact = True
      if has_walker and has_prop:
        walker_prop_contact = True
      if has_prop and has_lhand:
        prop_lhand_contact = True
      if has_prop and has_rhand:
        prop_rhand_contact = True

    return (prop_lhand_contact, prop_rhand_contact, bucket_prop_contact,
            bucket_walker_contact, walker_prop_contact)
