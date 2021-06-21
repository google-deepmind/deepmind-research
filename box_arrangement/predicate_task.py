# Copyright 2018 Deepmind Technologies Limited.
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

"""A task where different `Predicate`s need to be solved.

In each episode a spiking reward is given for each `Predicate` solved with an
extra reward bonus added when all of the predicates are solved. On each episode
the number of predicates are sampled randomly. This provides a common interface
to specify distributions over tasks ranging in difficulty levels but with common
components. Each `Predicate` involves some manipulation of the walker, props and
targets which thus allows for rich configurations of tasks to be defined.
"""
import colorsys
import functools
from dm_control import composer
from dm_control.composer.observation import observable
from dm_env import specs

import numpy as np


_FLOOR_GAP_CHAR = '#'
_AMBIENT_HEADLIGHT = 0.8
_HSV_SATURATION = 0.5
_HSV_ACTIVATED_SATURATION = 0.75
_HSV_VALUE = 1.0
_PROP_SIZE = 0.5
_MAX_ITERATIONS = 1000


def _generate_target_permutation(num_targets, random_state):
  targets = list(range(num_targets))
  random_state.shuffle(targets)
  return targets


class PredicateTask(composer.Task):
  """Requires objects to be moved onto targets."""

  def __init__(self,
               walker,
               maze_arena,
               predicates,
               props=None,
               targets=None,
               max_num_predicates=1,
               randomize_num_predicates=False,
               predicate_prob=None,
               reward_scale=1.0,
               terminating_reward_bonus=5.0,
               regenerate_predicates=False,
               physics_timestep=0.001,
               control_timestep=0.025,
               alive_threshold=-0.5):
    """Initializes a task with multiple sub-components(predicates) to be solved.

       This task essentially contains different flavors of go to target. The
       task contains a walker, props and target positions. To solve the entire
       task, the walker would need to solve a certain number of 'predicates' or
       sub-tasks. For instance, the task could contain 2 predicates for the
       walker going to a target position and the walker moving a box to a target
       position. In such a case, there is an implicit ordering of the way the
       walker needs to solve things to achieve the net task.

    Args:
      walker: A `Walker` instance.
      maze_arena: An `Entity` that defines a maze-like arena.
      predicates: A list of `Predicate` instances for ths task.
      props: An optional list of `manipulation.prop` instances for the task.
        These are used to generate observables for the task.
      targets: An optional list of `locomotion.prop` instances for the task.
        These are used to generate observables for the task.
      max_num_predicates: The maximum number of predicates to use in each
        episode of the task.
      randomize_num_predicates: A `bool` flag indicating whether the number of
        `valid` predicates should be randomized for each task. If set to `True`,
        then on each episode, between 1 and `num_predicates` are chosen as valid
        predicates and `predicate.invalid_observation_value` is output for the
        remaining slots in the observation.
      predicate_prob: An optional `list` containing the probabilities for each
        of the `predicates`. If not `None`, must have the same length as
        `predicates.
      reward_scale: `float` to scale the reward.
      terminating_reward_bonus: A bonus added to the reward when all predicates
        have been solved.
      regenerate_predicates: A `bool` flag indicating which when set, spawns a
        new set of predicates when the previous set is successful instead of
        terminating.
      physics_timestep: The time step of the physics simulation.
      control_timestep: Should be an integer multiple of the physics time step.
      alive_threshold: Aliveness in [-1., 0.].

    Raises:
      ValueError: If `num_props` is greater than `num_targets` or if
        `num_predicates` is greater than `num_targets`.
    """
    if max_num_predicates > len(predicates):
      raise ValueError('Not enough predicates for task. The maximum number of '
                       'predicates can be '
                       '{} but only {} predicates provided.'.format(
                           max_num_predicates, len(predicates)))
    self._arena = maze_arena
    self._walker = walker
    self._reward_scale = reward_scale

    self._alive_threshold = alive_threshold
    self._terminating_reward_bonus = terminating_reward_bonus
    self._arena.mjcf_model.visual.headlight.ambient = [_AMBIENT_HEADLIGHT] * 3
    maze_arena.text_maze_regenerated_hook = self._regenerate_positions

    self._max_num_predicates = max_num_predicates
    self._predicates = predicates
    self._predicate_prob = predicate_prob

    self._randomize_num_predicates = randomize_num_predicates
    self._active_predicates = []
    self._regen_predicates = regenerate_predicates
    self._reward = 0

    # Targets.
    self._targets = targets
    for target in targets:
      self._arena.attach(target)

    if props is None:
      props = []
    # Props.
    self._props = props
    # M Props + 1 Walker and we choose 'N' predicates as the task.
    for prop in props:
      prop.geom.rgba = [0, 0, 0, 1]  # Will be randomized for each episode.
      self._arena.add_free_entity(prop)

    # Create walkers and corresponding observables.
    walker.create_root_joints(self._arena.attach(walker))
    self._create_per_walker_observables(walker)
    self._generate_target_permutation = None
    maze_arena.text_maze_regenerated_hook = self._regenerate_positions

    # Set time steps.
    self.set_timesteps(
        physics_timestep=physics_timestep, control_timestep=control_timestep)

  def _create_per_walker_observables(self, walker):
    # Enable proprioceptive observables.
    for obs in (walker.observables.proprioception +
                walker.observables.kinematic_sensors +
                [walker.observables.position, walker.observables.orientation]):
      obs.enabled = True
    xpos_origin_callable = lambda phys: phys.bind(walker.root_body).xpos

    # Egocentric prop positions.
    # For each prop, we add the positions for the 8 corners using the sites.
    for prop_id, prop in enumerate(self._props):

      def _prop_callable(physics, prop=prop):
        return [physics.bind(s).xpos for s in prop.corner_sites]

      if len(self._props) > 1:
        observable_name = 'prop_{}_position'.format(prop_id)
      else:
        observable_name = 'prop_position'
      walker.observables.add_egocentric_vector(
          observable_name,
          observable.Generic(_prop_callable),
          origin_callable=xpos_origin_callable)

    # Egocentric target positions.
    def _target_callable(physics):
      target_list = []
      for target in self._targets:
        target_list.append(target.site_pos(physics))
      return np.array(target_list)

    walker.observables.add_egocentric_vector(
        'target_positions',
        observable.Generic(_target_callable),
        origin_callable=xpos_origin_callable)

    # Whether targets are activated.
    def _predicate_activated_callable(physics):
      predicate_activated_list = np.full(self._max_num_predicates, True)
      for i, predicate in enumerate(self._active_predicates):
        predicate_activated_list[i] = predicate.is_active(physics)
      return predicate_activated_list

    walker.observables.add_observable(
        'predicates_activated',
        observable.Generic(_predicate_activated_callable))
    self._observables = self._walker.observables.as_dict()

    # Predicate observables.
    for pred_idx in range(self._max_num_predicates):

      def _predicate_callable(_, pred_idx=pred_idx):
        """Callable for the predicate observation."""
        if pred_idx in range(len(self._active_predicates)):
          predicate = self._active_predicates[pred_idx]
          return predicate.observation_value
        else:
          # Use any predicates inactive observation to fill the rest.
          predicate = self._predicates[0]
          return predicate.inactive_observation_value

      predicate_name = 'predicate_{}'.format(pred_idx)
      self._observables[predicate_name] = observable.Generic(
          _predicate_callable)
      self._observables[predicate_name].enabled = True

  @property
  def observables(self):
    return self._observables

  @property
  def name(self):
    return 'predicate_task'

  @property
  def root_entity(self):
    return self._arena

  def _regenerate_positions(self):
    target_permutation = self._generate_target_permutation(
        len(self._arena.target_positions))
    num_permutations = len(self._props) + len(self._targets)
    target_permutation = target_permutation[:num_permutations]

    if len(self._props) + len(self._targets) > len(
        self._arena.target_positions):
      raise RuntimeError(
          'The generated maze does not contain enough target positions '
          'for the requested number of props ({}) and targets ({}): got {}.'
          .format(
              len(self._props), len(self._targets),
              len(self._arena.target_positions)))

    self._prop_positions = []
    for i in range(len(self._props)):
      self._prop_positions.append(
          self._arena.target_positions[target_permutation[i]])

    self._target_positions = []
    for i in range(len(self._targets)):
      idx = i + len(self._props)
      self._target_positions.append(
          self._arena.target_positions[target_permutation[idx]])

  def initialize_episode_mjcf(self, random_state):
    self._generate_target_permutation = functools.partial(
        _generate_target_permutation, random_state=random_state)
    self._arena.regenerate()

    # Set random colors for the props and targets.
    self._set_random_colors(random_state)
    self._set_active_predicates(random_state)

  def _set_active_predicates(self, random_state):
    # Reinitialize predicates to set any properties they want.
    iteration = 0
    valid_set_found = False
    while not valid_set_found and iteration < _MAX_ITERATIONS:
      for predicate in self._predicates:
        predicate.reinitialize(random_state)
      if self._randomize_num_predicates and self._max_num_predicates > 1:
        num_predicates = random_state.choice(
            list(range(1, self._max_num_predicates + 1)), size=1)[0]
      else:
        num_predicates = self._max_num_predicates
      valid_set_found = self._choose_random_predicates(random_state,
                                                       num_predicates)
      iteration += 1

    if not valid_set_found:
      raise ValueError(
          'Could not find set of active predicates with '
          'unique objects are after {} iterations.'.format(_MAX_ITERATIONS))
    for predicate in self._active_predicates:
      predicate.activate_predicate()

  def _choose_random_predicates(self, random_state, num_predicates):
    self._active_predicates = random_state.choice(
        self._predicates,
        replace=False,
        size=num_predicates,
        p=self._predicate_prob)
    objects_in_common = self._active_predicates[0].objects_in_use
    for predicate in self._active_predicates[1:]:
      new_objects = predicate.objects_in_use
      if objects_in_common.intersection(new_objects):
        return False
      objects_in_common.union(new_objects)
    return True

  def _set_random_colors(self, random_state):
    hue0 = random_state.uniform()
    hues = [(hue0 + i / len(self._targets)) % 1.0
            for i in range(len(self._targets))]
    rgbs = [
        colorsys.hsv_to_rgb(hue, _HSV_SATURATION, _HSV_VALUE) for hue in hues
    ]
    activated_rgbs = [
        colorsys.hsv_to_rgb(hue, _HSV_ACTIVATED_SATURATION, _HSV_VALUE)
        for hue in hues
    ]

    # There are fewer props than targets.
    # Pick as far apart colors for each prop as possible.
    if self._props:
      targets_per_prop = len(self._targets) // len(self._props)
    else:
      targets_per_prop = len(self._targets)

    for prop_id in range(len(self._props)):
      # The first few targets have to match the props' color.
      rgb_id = prop_id * targets_per_prop
      self._props[prop_id].geom.rgba[:3] = rgbs[rgb_id]
      self._targets[prop_id].set_colors(rgbs[rgb_id], activated_rgbs[rgb_id])

      # Assign colors not used by any prop to decoy targets.
      for decoy_target_offset in range(targets_per_prop - 1):
        target_id = len(
            self._props) + prop_id * targets_per_prop + decoy_target_offset
        rgb_id = prop_id * targets_per_prop + decoy_target_offset
        self._targets[target_id].set_colors(rgbs[rgb_id], rgbs[rgb_id])

    # Remainder loop for targets.
    for target_id in range(targets_per_prop * len(self._props),
                           len(self._targets)):
      self._targets[target_id].set_colors(rgbs[target_id], rgbs[target_id])

  def initialize_episode(self, physics, random_state):
    self._first_step = True
    self._was_active = [False] * len(self._active_predicates)

    walker = self._walker
    spawn_indices = random_state.permutation(len(self._arena.spawn_positions))
    spawn_index = spawn_indices[0]
    walker.reinitialize_pose(physics, random_state)
    spawn_position = self._arena.spawn_positions[spawn_index]
    spawn_rotation = random_state.uniform(-np.pi, np.pi)
    spawn_quat = np.array(
        [np.cos(spawn_rotation / 2), 0, 0,
         np.sin(spawn_rotation / 2)])
    walker.shift_pose(
        physics, [spawn_position[0], spawn_position[1], 0.0],
        spawn_quat,
        rotate_velocity=True)

    for prop, prop_xy_position in zip(self._props, self._prop_positions):
      # Position at the middle of a maze cell.
      prop_position = np.array(
          [prop_xy_position[0], prop_xy_position[1], prop.geom.size[2]])

      # Randomly rotate the prop around the z-axis.
      prop_rotation = random_state.uniform(-np.pi, np.pi)
      prop_quat = np.array(
          [np.cos(prop_rotation / 2), 0, 0,
           np.sin(prop_rotation / 2)])

      # Taking into account the prop's orientation, first calculate how much we
      # can displace the prop from the center of a maze cell without any part of
      # it sticking out of the cell.
      x, y, _ = prop.geom.size
      cos = np.cos(prop_rotation)
      sin = np.sin(prop_rotation)
      x_max = max([np.abs(x * cos - y * sin), np.abs(x * cos + y * sin)])
      y_max = max([np.abs(y * cos + x * sin), np.abs(y * cos - x * sin)])
      prop_max_displacement = self._arena.xy_scale / 2 - np.array(
          [x_max, y_max])
      assert np.all(prop_max_displacement >= 0)
      prop_max_displacement *= 0.99  # Safety factor.

      # Then randomly displace the prop from the center of the maze cell.
      prop_position[:2] += prop_max_displacement * random_state.uniform(
          -1, 1, 2)

      # Commit the prop's final pose.
      prop.set_pose(physics, position=prop_position, quaternion=prop_quat)

    for target, target_position in zip(self._targets, self._target_positions):
      target_position[2] = _PROP_SIZE
      target.set_position(physics, target_position)

  def before_step(self, physics, actions, random_state):
    if isinstance(actions, list):
      actions = np.concatenate(actions)
    super(PredicateTask, self).before_step(physics, actions, random_state)
    if self._first_step:
      self._first_step = False
    else:
      self._was_active = [
          predicate.is_active(physics) for predicate in self._active_predicates
      ]

  def after_step(self, physics, random_state):
    if self._all_predicates_satisfied() and self._regen_predicates:
      self._set_random_colors(random_state)
      self._set_active_predicates(random_state)
      super(PredicateTask, self).after_step(physics, random_state)

  def get_reward(self, physics):
    reward = 0.0
    for predicate, was_active in zip(self._active_predicates, self._was_active):
      if predicate.is_active(physics) and not was_active:
        reward += 1.0
      elif was_active and not predicate.is_active(physics):
        reward -= 1.0

    if self._all_predicates_satisfied():
      reward += self._terminating_reward_bonus
    self._reward = reward
    return reward * self._reward_scale

  def _all_predicates_satisfied(self):
    return sum(self._was_active) == len(self._active_predicates)

  def should_terminate_episode(self, physics):
    return ((self._all_predicates_satisfied() and not self._regen_predicates) or
            self._walker.aliveness(physics) < self._alive_threshold)

  def get_discount(self, physics):
    if self.should_terminate_episode(physics):
      return 0.0
    return 1.0

  def get_reward_spec(self):
    return specs.Array(shape=[], dtype=np.float32)

  def get_discount_spec(self):
    return specs.Array(shape=[], dtype=np.float32)
