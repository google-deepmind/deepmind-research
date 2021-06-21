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

"""Defines some `predicates` for the predicate_task."""
import abc
import colorsys
import numpy as np

HSV_SATURATION = 0.5
HSV_ACTIVATED_SATURATION = 0.75
HSV_VALUE = 1.0

WALKER_GOAL_RGBA = [0, 0, 0]
WALKER_GOAL_PRESSED_RGBA = [128, 128, 128]
INACTIVE_OBSERVATION_VALUE = [-1] * 5

# Define globals for the special encoding.
MOVABLE_TYPES = {'walker': 0, 'box': 1}
TARGET_TYPES = {'box': 0, 'target': 1}
PREDICATE_TYPES = {'on': 0, 'close_to': 1, 'far_from': 2}


class BasePredicate(object, metaclass=abc.ABCMeta):
  """Base class for all predicates."""

  def __init__(self, walker):
    self._walker = walker

  @abc.abstractmethod
  def reinitialize(self, random_state):
    """Reinitializes a new, potentially random, predicate state.

    The reinitialize method should reset to a new predicate state which could
    update the `objects_in_use` by the `Predicate`. This method could be called
    multiple times before a finally binding predicate set has been found.
    Therefore no changes to the model that are not reversible should be made
    here (setting colors etc). Any changes affecting the Mujoco model should
    instead be made in the `activate_predicate` method call.

    Args:
      random_state: An instance of `np.RandomState` which may be seeded to
        ensure a deterministic environment.
    """
    pass

  @abc.abstractmethod
  def activate_predicate(self):
    """Activates the current predicate configuration.

    Any changes that are non-reversible like setting object properties or
    affinities *must* only be done in this method. At this point, the
    `predicate_task` logic has confirmed that a valid predicate configuration
    has been found.
    """
    pass

  @property
  def objects_in_use(self):
    """Returns the `set` of objects used for this episode."""
    return set()

  @abc.abstractproperty
  def observation_value(self):
    """Returns a `dict` to be used as the predicate observable."""
    pass

  @abc.abstractmethod
  def is_active(self, physics):
    """Boolean method indicating whether the predicate has been activated.

    If `True`, it implies the condition for the predicate has been satisfied
    and the walker can be rewarded.

    Args:
      physics: An instance of `control.Physics`.
    """
    pass

  @property
  def inactive_observation_value(self):
    """observation_value indicating a `Predicate` is inactive.

    The `PredicateTask` randomly samples the number of active predicates to be
    used on each episode. For a consistent `observation_spec`, the predicates
    that are not active need a special observation_value that cannot be used
    anywhere else.

    Returns:
      A special value indicating that the predicate is inactive and is not used
      by any other predicate in the task.
    """
    return INACTIVE_OBSERVATION_VALUE


class MoveWalkerToTarget(BasePredicate):
  """Predicate to move a walker to a specific target."""

  def __init__(self, walker, target, target_index=0):
    """Predicate to move a walker or box to a target.

    Args:
      walker: An locomotion `Walker` instance to use for this predicate.
      target: `locomotion.prop` instance containing an `activated` property.
      target_index: An 'int' argument to add to the observable to indicate the
        index of the target.
    """
    super(MoveWalkerToTarget, self).__init__(walker)
    self._target = target
    self._target_id = target_index

  def reinitialize(self, random_state):
    self._target.deregister_entities()

  def activate_predicate(self):
    self._target.register_entities(self._walker)
    self._target.set_colors(WALKER_GOAL_RGBA, WALKER_GOAL_PRESSED_RGBA)

  @property
  def objects_in_use(self):
    return set([self._walker, self._target])

  @property
  def observation_value(self):
    return np.array([
        MOVABLE_TYPES['walker'], 0, TARGET_TYPES['target'], self._target_id,
        PREDICATE_TYPES['close_to']
    ])

  def is_active(self, physics):
    return self._target.activated


class MoveWalkerToRandomTarget(BasePredicate):
  """Predicate to move a walker to a random target."""

  def __init__(self, walker, targets=None):
    """Predicate to move a walker or box to a target.

    Args:
      walker: An locomotion `Walker` instance to use for this predicate.
      targets: An optional list of `locomotion.prop` instances each of which
        contains an `activated` property.
    """
    super(MoveWalkerToRandomTarget, self).__init__(walker)
    self._targets = targets
    self._target_to_move_to = None

  def reinitialize(self, random_state):
    if self._target_to_move_to is not None:
      self._target_to_move_to.deregister_entities()
    self._target_to_move_to = random_state.choice(self._targets)
    self._target_idx = self._targets.index(self._target_to_move_to)

  def activate_predicate(self):
    self._target_to_move_to.register_entities(self._walker)
    self._target_to_move_to.set_colors(WALKER_GOAL_RGBA,
                                       WALKER_GOAL_PRESSED_RGBA)

  @property
  def objects_in_use(self):
    return set([self._walker, self._target_to_move_to])

  @property
  def observation_value(self):
    return np.array([
        MOVABLE_TYPES['walker'], 0, TARGET_TYPES['target'], self._target_idx,
        PREDICATE_TYPES['close_to']
    ])

  def is_active(self, physics):
    return self._target_to_move_to.activated


class MoveWalkerToBox(BasePredicate):
  """Predicate to move a walker to a specific box."""

  def __init__(self, walker, box, box_index=0, detection_region=None):
    """Predicate to move a walker to a specific box.

    Args:
      walker: An locomotion `Walker` instance to use for this predicate.
      box: A `manipulation.prop` instance to move.
      box_index: An integer index to use for the observable to identify the
        `box`.
      detection_region: A 2-tuple indicating the tolerances in x and y for the
        walker to be deemed `close_to` the box. If `None`, contact based
        detection is used.
    """
    super(MoveWalkerToBox, self).__init__(walker)
    self._box = box
    self._detection_region = detection_region
    self._box_index = box_index
    self._walker_geoms = None

  def reinitialize(self, random_state):
    if self._walker_geoms is None:
      # pylint: disable=protected-access
      self._walker_geoms = set(self._walker._mjcf_root.find_all('geom'))

  def activate_predicate(self):
    self._box.geom.rgba[:3] = WALKER_GOAL_RGBA

  @property
  def objects_in_use(self):
    return set([self._walker, self._box])

  @property
  def observation_value(self):
    return np.array([
        MOVABLE_TYPES['walker'], 0, TARGET_TYPES['box'], self._box_index,
        PREDICATE_TYPES['close_to']
    ])

  def is_active(self, physics):
    if self._detection_region is None:
      return self._is_walker_contacting_box(physics)
    else:
      return np.all(
          np.abs(
              physics.bind(self._walker.root_body).xpos -
              physics.bind(self._box.geom).xpos)[:2] < self._detection_region)

  def _is_walker_contacting_box(self, physics):
    walker_geom_ids = [
        physics.bind(geom).element_id for geom in self._walker_geoms
    ]
    for contact in physics.data.contact:
      contact_geoms = set([contact.geom1, contact.geom2])
      if (physics.bind(self._box.geom).element_id in contact_geoms and
          contact_geoms.intersection(walker_geom_ids)):
        return True
    return False


class MoveBoxToBox(BasePredicate):
  """Predicate to move a walker to a specific box."""

  def __init__(self,
               walker,
               first_box,
               second_box,
               first_box_index=0,
               second_box_index=1,
               detection_region=None):
    """Predicate to move a walker to a specific box.

    Args:
      walker: An locomotion `Walker` instance to use for this predicate.
      first_box: A `manipulation.prop` instance to move.
      second_box: A `manipulation.prop` instance to move.
      first_box_index: An integer index to use for the observable to identify
        the  `box`.
      second_box_index: An integer index to use for the observable to identify
        the  `box`.
      detection_region: A 2-tuple indicating the tolerances in x and y for the
        walker to be deemed `close_to` the box. If `None`, contact based
        detection is used.
    """
    super(MoveBoxToBox, self).__init__(walker)
    self._first_box = first_box
    self._second_box = second_box
    self._detection_region = detection_region
    self._first_box_index = first_box_index
    self._second_box_index = second_box_index
    self._walker_geoms = None

  def reinitialize(self, random_state):
    if self._walker_geoms is None:
      # pylint: disable=protected-access
      self._walker_geoms = set(self._walker._mjcf_root.find_all('geom'))

  def activate_predicate(self):
    self._first_box.geom.rgba[:3] = WALKER_GOAL_RGBA

  @property
  def objects_in_use(self):
    return set([self._first_box, self._second_box])

  @property
  def observation_value(self):
    return np.array([
        MOVABLE_TYPES['box'], self._first_box_index, TARGET_TYPES['box'],
        self._second_box_index, PREDICATE_TYPES['close_to']
    ])

  def is_active(self, physics):
    if self._detection_region is None:
      return self._are_boxes_in_contact(physics)
    else:
      return np.all(
          np.abs(
              physics.bind(self._first_box.geom).xpos -
              physics.bind(self._second_box.geom).xpos)[:2] <
          self._detection_region)

  def _are_boxes_in_contact(self, physics):
    for contact in physics.data.contact:
      contact_geoms = set([contact.geom1, contact.geom2])
      if (physics.bind(self._first_box.geom).element_id in contact_geoms and
          physics.bind(self._second_box.geom).element_id in contact_geoms):
        return True
    return False


class MoveBoxToTarget(BasePredicate):
  """Predicate to move a walker to a specific target."""

  def __init__(self, walker, box, target, box_index=0, target_index=0):
    """Predicate to move a walker or box to a target.

    Args:
      walker: An locomotion `Walker` instance to use for this predicate.
      box: A `manipulation.prop` to move to the target.
      target: `locomotion.prop` instance containing an `activated` property.
      box_index: An 'int' argument to add to the observable to indicate the
        index of the box.
      target_index: An 'int' argument to add to the observable to indicate the
        index of the target.
    """
    super(MoveBoxToTarget, self).__init__(walker)
    self._box = box
    self._target = target
    self._box_id = box_index
    self._target_id = target_index
    self._original_box_size = np.copy(box.geom.size)
    self._rgb = None
    self._activated_rgb = None

  def reinitialize(self, random_state):
    self._target.deregister_entities()
    self._get_box_properties(random_state)

  def _get_box_properties(self, random_state):
    hue0 = random_state.uniform()
    hue = (hue0 + self._target_id) % 1.0
    self._rgb = colorsys.hsv_to_rgb(hue, HSV_SATURATION, HSV_VALUE)
    self._activated_rgb = colorsys.hsv_to_rgb(hue, HSV_ACTIVATED_SATURATION,
                                              HSV_VALUE)

  def activate_predicate(self):
    self._target.set_colors(self._rgb, self._activated_rgb)
    self._box.geom.rgba[:3] = self._rgb
    self._target.register_entities(self._box)

  @property
  def objects_in_use(self):
    return set([self._box, self._target])

  @property
  def observation_value(self):
    return np.array([
        MOVABLE_TYPES['box'], self._box_id, TARGET_TYPES['target'],
        self._target_id, PREDICATE_TYPES['close_to']
    ])

  def is_active(self, physics):
    return self._target.activated


class MoveBoxToRandomTarget(BasePredicate):
  """Predicate to move a walker to a random target."""

  def __init__(self, walker, box, box_index=0, targets=None):
    """Predicate to move a walker or box to a target.

    Args:
      walker: An locomotion `Walker` instance to use for this predicate.
      box: A `manipulation.prop` to move to the target.
      box_index: An optional 'int' argument to add to the observable to indicate
        the index of the box.
      targets: An optional list of `locomotion.prop` instances each of which
        contains an `activated` property.
    """
    super(MoveBoxToRandomTarget, self).__init__(walker)
    self._targets = targets
    self._box_to_move = box
    self._box_index = box_index
    self._target_to_move_to = None
    self._original_box_size = np.copy(box.geom.size)
    self._rgb = None
    self._activated_rgb = None

  def reinitialize(self, random_state):
    if self._target_to_move_to is not None:
      self._target_to_move_to.deregister_entities()
    self._target_to_move_to = random_state.choice(self._targets)
    self._target_idx = self._targets.index(self._target_to_move_to)
    self._get_box_properties(random_state)

  def _get_box_properties(self, random_state):
    hue0 = random_state.uniform()
    hue = (hue0 + (self._target_idx / len(self._targets))) % 1.0
    self._rgb = colorsys.hsv_to_rgb(hue, HSV_SATURATION, HSV_VALUE)
    self._activated_rgb = colorsys.hsv_to_rgb(hue, HSV_ACTIVATED_SATURATION,
                                              HSV_VALUE)

  def activate_predicate(self):
    self._target_to_move_to.set_colors(self._rgb, self._activated_rgb)
    self._box_to_move.geom.rgba[:3] = self._rgb
    self._target_to_move_to.register_entities(self._box_to_move)

  @property
  def objects_in_use(self):
    return set([self._box_to_move, self._target_to_move_to])

  @property
  def observation_value(self):
    return np.array([
        MOVABLE_TYPES['box'], self._box_index,
        TARGET_TYPES['target'], self._target_idx,
        PREDICATE_TYPES['close_to']
    ])

  def is_active(self, physics):
    return self._target_to_move_to.activated
