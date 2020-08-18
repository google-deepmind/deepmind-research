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

"""A prop-carry task that transition between multiple phases."""

import collections
import colorsys
import enum

from absl import logging
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.mocap import loader as mocap_loader
from dm_control.mujoco.wrapper import mjbindings
import numpy as np

from catch_carry import arm_opener
from catch_carry import mocap_data
from catch_carry import props
from catch_carry import trajectories

_PHYSICS_TIMESTEP = 0.005

# Maximum number of physics steps to run when settling props onto pedestals
# during episode initialization.
_MAX_SETTLE_STEPS = 1000

# Maximum velocity for prop to be considered settled.
# Used during episode initialization only.
_SETTLE_QVEL_TOL = 1e-5

# Magnitude of the sparse reward.
_SPARSE_REWARD = 1.0

# Maximum distance for walkers to be considered to be "near" a pedestal/target.
_TARGET_TOL = 0.65

# Defines how pedestals are placed around the arena.
# Pedestals are placed at constant angle intervals around the arena's center.
_BASE_PEDESTAL_DIST = 3  # Base distance from center.
_PEDESTAL_DIST_DELTA = 0.5  # Maximum variation on the base distance.

# Base hue-luminosity-saturation of the pedestal colors.
# We rotate through the hue for each pedestal created in the environment.
_BASE_PEDESTAL_H = 0.1
_BASE_PEDESTAL_L = 0.3
_BASE_PEDESTAL_S = 0.7

# Pedestal luminosity when active.
_ACTIVATED_PEDESTAL_L = 0.8

_PEDESTAL_SIZE = (0.2, 0.2, 0.02)

_SINGLE_PEDESTAL_COLOR = colorsys.hls_to_rgb(.3, .15, .35) + (1.0,)

WALKER_PEDESTAL = 'walker_pedestal'
WALKER_PROP = 'walker_prop'
PROP_PEDESTAL = 'prop_pedestal'
TARGET_STATE = 'target_state/'
CURRENT_STATE = 'meta/current_state/'


def _is_same_state(state_1, state_2):
  if state_1.keys() != state_2.keys():
    return False
  for k in state_1:
    if not np.all(state_1[k] == state_2[k]):
      return False
  return True


def _singleton_or_none(iterable):
  iterator = iter(iterable)
  try:
    return next(iterator)
  except StopIteration:
    return None


def _generate_pedestal_colors(num_pedestals):
  """Function to get colors for pedestals."""
  colors = []
  for i in range(num_pedestals):
    h = _BASE_PEDESTAL_H + i / num_pedestals
    while h > 1:
      h -= 1
    colors.append(
        colorsys.hls_to_rgb(h, _BASE_PEDESTAL_L, _BASE_PEDESTAL_S) + (1.0,))
  return colors


InitializationParameters = collections.namedtuple(
    'InitializationParameters', ('clip_segment', 'prop_id', 'pedestal_id'))


def _rotate_vector_by_quaternion(vec, quat):
  result = np.empty(3)
  mjbindings.mjlib.mju_rotVecQuat(result, np.asarray(vec), np.asarray(quat))
  return result


@enum.unique
class WarehousePhase(enum.Enum):
  TERMINATED = 0
  GOTOTARGET = 1
  PICKUP = 2
  CARRYTOTARGET = 3
  PUTDOWN = 4


def _find_random_free_pedestal_id(target_state, random_state):
  free_pedestals = (
      np.where(np.logical_not(np.any(target_state, axis=0)))[0])
  return random_state.choice(free_pedestals)


def _find_random_occupied_pedestal_id(target_state, random_state):
  occupied_pedestals = (
      np.where(np.any(target_state, axis=0))[0])
  return random_state.choice(occupied_pedestals)


def one_hot(values, num_unique):
  return np.squeeze(np.eye(num_unique)[np.array(values).reshape(-1)])


class SinglePropFourPhases(object):
  """A phase manager that transitions between four phases for a single prop."""

  def __init__(self, fixed_initialization_phase=None):
    self._phase = WarehousePhase.TERMINATED
    self._fixed_initialization_phase = fixed_initialization_phase

  def initialize_episode(self, target_state, random_state):
    """Randomly initializes an episode into one of the four phases."""

    if self._fixed_initialization_phase is None:
      self._phase = random_state.choice([
          WarehousePhase.GOTOTARGET, WarehousePhase.PICKUP,
          WarehousePhase.CARRYTOTARGET, WarehousePhase.PUTDOWN
      ])
    else:
      self._phase = self._fixed_initialization_phase
    self._prop_id = random_state.randint(len(target_state[PROP_PEDESTAL]))
    self._pedestal_id = np.nonzero(
        target_state[PROP_PEDESTAL][self._prop_id])[0][0]
    pedestal_id_for_initialization = self._pedestal_id

    if self._phase == WarehousePhase.GOTOTARGET:
      clip_segment = trajectories.ClipSegment.APPROACH
      target_state[WALKER_PROP][:] = 0
      target_state[WALKER_PEDESTAL][self._pedestal_id] = 1
    elif self._phase == WarehousePhase.PICKUP:
      clip_segment = trajectories.ClipSegment.PICKUP
      target_state[WALKER_PROP][self._prop_id] = 1
      target_state[WALKER_PEDESTAL][self._pedestal_id] = 1
      # Set self._pedestal_id to the next pedestal after pickup is successful.
      self._pedestal_id = _find_random_free_pedestal_id(
          target_state[PROP_PEDESTAL], random_state)
      target_state[PROP_PEDESTAL][self._prop_id, :] = 0
    elif self._phase == WarehousePhase.CARRYTOTARGET:
      clip_segment = random_state.choice([
          trajectories.ClipSegment.CARRY1, trajectories.ClipSegment.CARRY2])
      self._pedestal_id = _find_random_free_pedestal_id(
          target_state[PROP_PEDESTAL], random_state)
      if clip_segment == trajectories.ClipSegment.CARRY2:
        pedestal_id_for_initialization = self._pedestal_id
      target_state[WALKER_PROP][self._prop_id] = 1
      target_state[WALKER_PEDESTAL][self._pedestal_id] = 1
      target_state[PROP_PEDESTAL][self._prop_id, :] = 0
    elif self._phase == WarehousePhase.PUTDOWN:
      clip_segment = trajectories.ClipSegment.PUTDOWN
      target_state[WALKER_PROP][:] = 0
      target_state[WALKER_PEDESTAL][self._pedestal_id] = 1

    return InitializationParameters(
        clip_segment, self._prop_id, pedestal_id_for_initialization)

  def on_success(self, target_state, random_state):
    """Transitions into the next phase upon success of current phase."""
    if self._phase == WarehousePhase.GOTOTARGET:
      if self._prop_id is not None:
        self._phase = WarehousePhase.PICKUP
        # Set self._pedestal_id to the next pedestal after pickup is successful.
        self._pedestal_id = (
            _find_random_free_pedestal_id(
                target_state[PROP_PEDESTAL], random_state))
        target_state[WALKER_PROP][self._prop_id] = 1
        target_state[PROP_PEDESTAL][self._prop_id, :] = 0
      else:
        # If you go to an empty pedestal, go to pedestal with a prop.
        self._pedestal_id = (
            _find_random_occupied_pedestal_id(
                target_state[PROP_PEDESTAL], random_state))
        target_state[WALKER_PEDESTAL][:] = 0
        target_state[WALKER_PEDESTAL][self._pedestal_id] = 1
        self._prop_id = np.argwhere(
            target_state[PROP_PEDESTAL][:, self._pedestal_id])[0, 0]
    elif self._phase == WarehousePhase.PICKUP:
      self._phase = WarehousePhase.CARRYTOTARGET
      target_state[WALKER_PEDESTAL][:] = 0
      target_state[WALKER_PEDESTAL][self._pedestal_id] = 1
    elif self._phase == WarehousePhase.CARRYTOTARGET:
      self._phase = WarehousePhase.PUTDOWN
      target_state[WALKER_PROP][:] = 0
      target_state[PROP_PEDESTAL][self._prop_id, self._pedestal_id] = 1
    elif self._phase == WarehousePhase.PUTDOWN:
      self._phase = WarehousePhase.GOTOTARGET
      # Set self._pedestal_id to the next pedestal after putdown is successful.
      self._pedestal_id = (
          _find_random_free_pedestal_id(
              target_state[PROP_PEDESTAL], random_state))
      self._prop_id = None
      target_state[WALKER_PEDESTAL][:] = 0
      target_state[WALKER_PEDESTAL][self._pedestal_id] = 1
    return self._phase

  @property
  def phase(self):
    return self._phase

  @property
  def prop_id(self):
    return self._prop_id

  @property
  def pedestal_id(self):
    return self._pedestal_id


class PhasedBoxCarry(composer.Task):
  """A prop-carry task that transitions between multiple phases."""

  def __init__(
      self,
      walker,
      num_props,
      num_pedestals,
      proto_modifier=None,
      transition_class=SinglePropFourPhases,
      min_prop_gap=0.05,
      pedestal_height_range=(0.45, 0.75),
      log_transitions=False,
      negative_reward_on_failure_termination=True,
      use_single_pedestal_color=True,
      priority_friction=False,
      fixed_initialization_phase=None):
    """Initialize phased/instructed box-carrying ("warehouse") task.

    Args:
      walker: the walker to be used in this task.
      num_props: the number of props in the task scene.
      num_pedestals: the number of floating shelves (pedestals) in the task
        scene.
      proto_modifier: function to modify trajectory proto.
      transition_class: the object that handles the transition logic.
      min_prop_gap: arms are automatically opened to leave a gap around the prop
        to avoid problematic collisions upon initialization.
      pedestal_height_range: range of heights for the pedestal.
      log_transitions: logging/printing of transitions.
      negative_reward_on_failure_termination: boolean for whether to provide
        negative sparse rewards on failure termination.
      use_single_pedestal_color: boolean option for pedestals being the same
        color or different colors.
      priority_friction: sets friction priority thereby making prop objects have
        higher friction.
      fixed_initialization_phase: an instance of the `WarehousePhase` enum that
        specifies the phase in which to always initialize the task, or `None` if
        the initial task phase should be chosen randomly for each episode.
    """
    self._num_props = num_props
    self._num_pedestals = num_pedestals
    self._proto_modifier = proto_modifier
    self._transition_manager = transition_class(
        fixed_initialization_phase=fixed_initialization_phase)
    self._min_prop_gap = min_prop_gap
    self._pedestal_height_range = pedestal_height_range
    self._log_transitions = log_transitions
    self._target_state = collections.OrderedDict([
        (WALKER_PEDESTAL, np.zeros(num_pedestals)),
        (WALKER_PROP, np.zeros(num_props)),
        (PROP_PEDESTAL, np.zeros([num_props, num_pedestals]))
    ])
    self._current_state = collections.OrderedDict([
        (WALKER_PEDESTAL, np.zeros(num_pedestals)),
        (WALKER_PROP, np.zeros(num_props)),
        (PROP_PEDESTAL, np.zeros([num_props, num_pedestals]))
    ])
    self._negative_reward_on_failure_termination = (
        negative_reward_on_failure_termination)
    self._priority_friction = priority_friction

    clips = sorted(
        set(mocap_data.medium_pedestal())
        & (set(mocap_data.small_box()) | set(mocap_data.large_box())))
    loader = mocap_loader.HDF5TrajectoryLoader(
        mocap_data.H5_PATH, trajectories.SinglePropCarrySegmentedTrajectory)
    self._trajectories = [
        loader.get_trajectory(clip.clip_identifier) for clip in clips]

    self._arena = floors.Floor()

    self._walker = walker
    self._feet_geoms = (
        walker.mjcf_model.find('body', 'lfoot').find_all('geom') +
        walker.mjcf_model.find('body', 'rfoot').find_all('geom'))
    self._lhand_geoms = (
        walker.mjcf_model.find('body', 'lhand').find_all('geom'))
    self._rhand_geoms = (
        walker.mjcf_model.find('body', 'rhand').find_all('geom'))
    self._trajectories[0].configure_walkers([self._walker])
    walker.create_root_joints(self._arena.attach(walker))

    control_timestep = self._trajectories[0].dt
    for i, trajectory in enumerate(self._trajectories):
      if trajectory.dt != control_timestep:
        raise ValueError(
            'Inconsistent control timestep: '
            'trajectories[{}].dt == {} but trajectories[0].dt == {}'
            .format(i, trajectory.dt, control_timestep))
    self.set_timesteps(control_timestep, _PHYSICS_TIMESTEP)

    if use_single_pedestal_color:
      self._pedestal_colors = [_SINGLE_PEDESTAL_COLOR] * num_pedestals
    else:
      self._pedestal_colors = _generate_pedestal_colors(num_pedestals)
    self._pedestals = [props.Pedestal(_PEDESTAL_SIZE, rgba)
                       for rgba in self._pedestal_colors]
    for pedestal in self._pedestals:
      self._arena.attach(pedestal)

    self._props = [
        self._trajectories[0].create_props(
            priority_friction=self._priority_friction)[0]
        for _ in range(num_props)
    ]
    for prop in self._props:
      self._arena.add_free_entity(prop)

    self._task_observables = collections.OrderedDict()

    self._task_observables['target_phase'] = observable.Generic(
        lambda _: one_hot(self._transition_manager.phase.value, num_unique=5))

    def ego_prop_xpos(physics):
      prop_id = self._focal_prop_id
      if prop_id is None:
        return np.zeros((3,))
      prop = self._props[prop_id]
      prop_xpos, _ = prop.get_pose(physics)
      walker_xpos = physics.bind(self._walker.root_body).xpos
      return self._walker.transform_vec_to_egocentric_frame(
          physics, prop_xpos - walker_xpos)
    self._task_observables['target_prop/xpos'] = (
        observable.Generic(ego_prop_xpos))

    def prop_zaxis(physics):
      prop_id = self._focal_prop_id
      if prop_id is None:
        return np.zeros((3,))
      prop = self._props[prop_id]
      prop_xmat = physics.bind(
          mjcf.get_attachment_frame(prop.mjcf_model)).xmat
      return prop_xmat[[2, 5, 8]]
    self._task_observables['target_prop/zaxis'] = (
        observable.Generic(prop_zaxis))

    def ego_pedestal_xpos(physics):
      pedestal_id = self._focal_pedestal_id
      if pedestal_id is None:
        return np.zeros((3,))
      pedestal = self._pedestals[pedestal_id]
      pedestal_xpos, _ = pedestal.get_pose(physics)
      walker_xpos = physics.bind(self._walker.root_body).xpos
      return self._walker.transform_vec_to_egocentric_frame(
          physics, pedestal_xpos - walker_xpos)
    self._task_observables['target_pedestal/xpos'] = (
        observable.Generic(ego_pedestal_xpos))

    for obs in (self._walker.observables.proprioception +
                self._walker.observables.kinematic_sensors +
                self._walker.observables.dynamic_sensors +
                list(self._task_observables.values())):
      obs.enabled = True

    self._focal_prop_id = None
    self._focal_pedestal_id = None

  @property
  def root_entity(self):
    return self._arena

  @property
  def task_observables(self):
    return self._task_observables

  @property
  def name(self):
    return 'warehouse'

  def initialize_episode_mjcf(self, random_state):
    self._reward = 0.0
    self._discount = 1.0
    self._should_terminate = False
    self._before_step_success = False
    for target_value in self._target_state.values():
      target_value[:] = 0
    for pedestal_id, pedestal in enumerate(self._pedestals):
      angle = 2 * np.pi * pedestal_id / len(self._pedestals)
      dist = (_BASE_PEDESTAL_DIST +
              _PEDESTAL_DIST_DELTA * random_state.uniform(-1, 1))

      height = random_state.uniform(*self._pedestal_height_range)
      pedestal_pos = [dist * np.cos(angle), dist * np.sin(angle),
                      height - pedestal.geom.size[2]]
      mjcf.get_attachment_frame(pedestal.mjcf_model).pos = pedestal_pos

    for prop in self._props:
      prop.detach()
    self._props = []
    self._trajectory_for_prop = []

    for prop_id in range(self._num_props):
      trajectory = random_state.choice(self._trajectories)
      if self._proto_modifier:
        trajectory = trajectory.get_modified_trajectory(
            self._proto_modifier, random_state=random_state)
      prop = trajectory.create_props(
          priority_friction=self._priority_friction)[0]
      prop.mjcf_model.model = 'prop_{}'.format(prop_id)
      self._arena.add_free_entity(prop)
      self._props.append(prop)
      self._trajectory_for_prop.append(trajectory)

  def _settle_props(self, physics):
    prop_freejoints = [mjcf.get_attachment_frame(prop.mjcf_model).freejoint
                       for prop in self._props]
    physics.bind(prop_freejoints).qvel = 0
    physics.forward()
    for _ in range(_MAX_SETTLE_STEPS):
      self._update_current_state(physics)
      success = self._evaluate_target_state()
      stopped = max(abs(physics.bind(prop_freejoints).qvel)) < _SETTLE_QVEL_TOL
      if success and stopped:
        break
      else:
        physics.step()
    physics.data.time = 0

  def initialize_episode(self, physics, random_state):
    self._ground_geomid = physics.bind(
        self._arena.mjcf_model.worldbody.geom[0]).element_id
    self._feet_geomids = set(physics.bind(self._feet_geoms).element_id)
    self._lhand_geomids = set(physics.bind(self._lhand_geoms).element_id)
    self._rhand_geomids = set(physics.bind(self._rhand_geoms).element_id)

    for prop_id in range(len(self._props)):
      pedestal_id = _find_random_free_pedestal_id(
          self._target_state[PROP_PEDESTAL], random_state)
      pedestal = self._pedestals[pedestal_id]
      self._target_state[PROP_PEDESTAL][prop_id, pedestal_id] = 1

    for prop_id, prop in enumerate(self._props):
      trajectory = self._trajectory_for_prop[prop_id]
      pedestal_id = np.nonzero(
          self._target_state[PROP_PEDESTAL][prop_id])[0][0]
      pedestal = self._pedestals[pedestal_id]
      pedestal_pos, _ = pedestal.get_pose(physics)
      pedestal_delta = np.array(
          pedestal_pos - trajectory.infer_pedestal_positions()[0])
      pedestal_delta[2] += pedestal.geom.size[2]
      prop_timestep = trajectory.get_timestep_data(0).props[0]
      prop_pos = prop_timestep.position + np.array(pedestal_delta)
      prop_quat = prop_timestep.quaternion
      prop_pos[:2] += random_state.uniform(
          -pedestal.geom.size[:2] / 2, pedestal.geom.size[:2] / 2)
      prop.set_pose(physics, prop_pos, prop_quat)
    self._settle_props(physics)

    init_params = self._transition_manager.initialize_episode(
        self._target_state, random_state)
    if self._log_transitions:
      logging.info(init_params)
    self._on_transition(physics)

    init_prop = self._props[init_params.prop_id]
    init_pedestal = self._pedestals[init_params.pedestal_id]
    self._init_prop_id = init_params.prop_id
    self._init_pedestal_id = init_params.pedestal_id
    init_trajectory = self._trajectory_for_prop[init_params.prop_id]
    init_timestep = init_trajectory.get_random_timestep_in_segment(
        init_params.clip_segment, random_state)

    trajectory_pedestal_pos = init_trajectory.infer_pedestal_positions()[0]
    init_pedestal_pos = np.array(init_pedestal.get_pose(physics)[0])
    delta_pos = init_pedestal_pos - trajectory_pedestal_pos
    delta_pos[2] = 0
    delta_angle = np.pi + np.arctan2(init_pedestal_pos[1], init_pedestal_pos[0])
    delta_quat = (np.cos(delta_angle / 2), 0, 0, np.sin(delta_angle / 2))

    trajectory_pedestal_to_walker = (
        init_timestep.walkers[0].position - trajectory_pedestal_pos)
    rotated_pedestal_to_walker = _rotate_vector_by_quaternion(
        trajectory_pedestal_to_walker, delta_quat)

    self._walker.set_pose(
        physics,
        position=trajectory_pedestal_pos + rotated_pedestal_to_walker,
        quaternion=init_timestep.walkers[0].quaternion)
    self._walker.set_velocity(
        physics, velocity=init_timestep.walkers[0].velocity,
        angular_velocity=init_timestep.walkers[0].angular_velocity)
    self._walker.shift_pose(
        physics, position=delta_pos, quaternion=delta_quat,
        rotate_velocity=True)
    physics.bind(self._walker.mocap_joints).qpos = (
        init_timestep.walkers[0].joints)
    physics.bind(self._walker.mocap_joints).qvel = (
        init_timestep.walkers[0].joints_velocity)

    if init_params.clip_segment in (trajectories.ClipSegment.CARRY1,
                                    trajectories.ClipSegment.CARRY2,
                                    trajectories.ClipSegment.PUTDOWN):
      trajectory_pedestal_to_prop = (
          init_timestep.props[0].position - trajectory_pedestal_pos)
      rotated_pedestal_to_prop = _rotate_vector_by_quaternion(
          trajectory_pedestal_to_prop, delta_quat)
      init_prop.set_pose(
          physics,
          position=trajectory_pedestal_pos + rotated_pedestal_to_prop,
          quaternion=init_timestep.props[0].quaternion)
      init_prop.set_velocity(
          physics, velocity=init_timestep.props[0].velocity,
          angular_velocity=init_timestep.props[0].angular_velocity)
      init_prop.shift_pose(
          physics, position=delta_pos,
          quaternion=delta_quat, rotate_velocity=True)

      # If we have moved the pedestal upwards during height initialization,
      # the prop may now be lodged inside it. We fix that here.
      if init_pedestal_pos[2] > trajectory_pedestal_pos[2]:
        init_prop_geomid = physics.bind(init_prop.geom).element_id
        init_pedestal_geomid = physics.bind(init_pedestal.geom).element_id
        disallowed_contact = sorted((init_prop_geomid, init_pedestal_geomid))
        def has_disallowed_contact():
          physics.forward()
          for contact in physics.data.contact:
            if sorted((contact.geom1, contact.geom2)) == disallowed_contact:
              return True
          return False
        while has_disallowed_contact():
          init_prop.shift_pose(physics, (0, 0, 0.001))

    self._move_arms_if_necessary(physics)
    self._update_current_state(physics)
    self._previous_step_success = self._evaluate_target_state()

    self._focal_prop_id = self._init_prop_id
    self._focal_pedestal_id = self._init_pedestal_id

  def _move_arms_if_necessary(self, physics):
    if self._min_prop_gap is not None:
      for entity in self._props + self._pedestals:
        try:
          arm_opener.open_arms_for_prop(
              physics, self._walker.left_arm_root, self._walker.right_arm_root,
              entity.mjcf_model, self._min_prop_gap)
        except RuntimeError as e:
          raise composer.EpisodeInitializationError(e)

  def after_step(self, physics, random_state):
    # First we check for failure termination.
    for contact in physics.data.contact:
      if ((contact.geom1 == self._ground_geomid and
           contact.geom2 not in self._feet_geomids) or
          (contact.geom2 == self._ground_geomid and
           contact.geom1 not in self._feet_geomids)):
        if self._negative_reward_on_failure_termination:
          self._reward = -_SPARSE_REWARD
        else:
          self._reward = 0.0
        self._should_terminate = True
        self._discount = 0.0
        return

    # Then check for normal reward and state transitions.
    self._update_current_state(physics)
    success = self._evaluate_target_state()
    if success and not self._previous_step_success:
      self._reward = _SPARSE_REWARD
      new_phase = (
          self._transition_manager.on_success(self._target_state, random_state))
      self._should_terminate = (new_phase == WarehousePhase.TERMINATED)
      self._on_transition(physics)
      self._previous_step_success = self._evaluate_target_state()
    else:
      self._reward = 0.0

  def _on_transition(self, physics):
    self._focal_prop_id = self._transition_manager.prop_id
    self._focal_pedestal_id = self._transition_manager.pedestal_id
    if self._log_transitions:
      logging.info('target_state:\n%s', self._target_state)
    for pedestal_id, pedestal_active in enumerate(
        self._target_state[WALKER_PEDESTAL]):
      r, g, b, a = self._pedestal_colors[pedestal_id]
      if pedestal_active:
        h, _, s = colorsys.rgb_to_hls(r, g, b)
        r, g, b = colorsys.hls_to_rgb(h, _ACTIVATED_PEDESTAL_L, s)
      physics.bind(self._pedestals[pedestal_id].geom).rgba = (r, g, b, a)

  def get_reward(self, physics):
    return self._reward

  def get_discount(self, physics):
    return self._discount

  def should_terminate_episode(self, physics):
    return self._should_terminate

  def _update_current_state(self, physics):
    for current_state_value in self._current_state.values():
      current_state_value[:] = 0

    # Check if the walker is near each pedestal.
    walker_pos, _ = self._walker.get_pose(physics)
    for pedestal_id, pedestal in enumerate(self._pedestals):
      target_pos, _ = pedestal.get_pose(physics)
      walker_to_target_dist = np.linalg.norm(walker_pos[:2] - target_pos[:2])
      if walker_to_target_dist <= _TARGET_TOL:
        self._current_state[WALKER_PEDESTAL][pedestal_id] = 1

    prop_geomids = {
        physics.bind(prop.geom).element_id: prop_id
        for prop_id, prop in enumerate(self._props)}
    pedestal_geomids = {
        physics.bind(pedestal.geom).element_id: pedestal_id
        for pedestal_id, pedestal in enumerate(self._pedestals)}

    prop_pedestal_contact_counts = np.zeros(
        [self._num_props, self._num_pedestals])
    prop_lhand_contact = [False] * self._num_props
    prop_rhand_contact = [False] * self._num_props
    for contact in physics.data.contact:
      prop_id = prop_geomids.get(contact.geom1, prop_geomids.get(contact.geom2))
      pedestal_id = pedestal_geomids.get(
          contact.geom1, pedestal_geomids.get(contact.geom2))
      has_lhand = (contact.geom1 in self._lhand_geomids or
                   contact.geom2 in self._lhand_geomids)
      has_rhand = (contact.geom1 in self._rhand_geomids or
                   contact.geom2 in self._rhand_geomids)
      if prop_id is not None and pedestal_id is not None:
        prop_pedestal_contact_counts[prop_id, pedestal_id] += 1
      if prop_id is not None and has_lhand:
        prop_lhand_contact[prop_id] = True
      if prop_id is not None and has_rhand:
        prop_rhand_contact[prop_id] = True

    for prop_id in range(self._num_props):
      if prop_lhand_contact[prop_id] and prop_rhand_contact[prop_id]:
        self._current_state[WALKER_PROP][prop_id] = 1
      pedestal_contact_counts = prop_pedestal_contact_counts[prop_id]
      for pedestal_id in range(self._num_pedestals):
        if pedestal_contact_counts[pedestal_id] >= 4:
          self._current_state[PROP_PEDESTAL][prop_id, pedestal_id] = 1

  def _evaluate_target_state(self):
    return _is_same_state(self._current_state, self._target_state)
