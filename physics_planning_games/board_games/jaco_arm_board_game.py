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

"""Shared base class for two-player Jaco arm board games.
"""

import functools

from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.entities.manipulators import base
from dm_control.entities.manipulators import kinova
import numpy as np

from physics_planning_games.board_games._internal import arenas
from physics_planning_games.board_games._internal import observations

_ARM_Y_OFFSET = 0.4
_TCP_LOWER_BOUNDS = (-0.1, -0.1, 0.2)
_TCP_UPPER_BOUNDS = (0.1, 0.1, 0.4)

# Player IDs
SELF = 0
OPPONENT = 1


def _uniform_downward_rotation():
  angle = distributions.Uniform(-np.pi, np.pi, single_sample=True)
  quaternion = rotations.QuaternionFromAxisAngle(axis=(0., 0., 1.), angle=angle)
  return functools.partial(rotations.QuaternionPreMultiply(quaternion),
                           initial_value=base.DOWN_QUATERNION)


class JacoArmBoardGame(composer.Task):
  """Base class for two-player checker-like board games."""

  def __init__(self, observation_settings, opponent, game_logic, board,
               markers):
    """Initializes the task.

    Args:
      observation_settings: An `observations.ObservationSettings` namedtuple
        specifying configuration options for each category of observation.
      opponent: Opponent used for generating opponent moves.
      game_logic: Logic for keeping track of the logical state of the board.
      board: Board to use.
      markers: Markers to use.
    """
    self._game_logic = game_logic
    self._game_opponent = opponent
    arena = arenas.Standard(observable_options=observations.make_options(
        observation_settings, observations.ARENA_OBSERVABLES))
    arena.attach(board)
    arm = kinova.JacoArm(observable_options=observations.make_options(
        observation_settings, observations.JACO_ARM_OBSERVABLES))
    hand = kinova.JacoHand(observable_options=observations.make_options(
        observation_settings, observations.JACO_HAND_OBSERVABLES))
    arm.attach(hand)
    arena.attach_offset(arm, offset=(0, _ARM_Y_OFFSET, 0))
    arena.attach(markers)

    # Geoms belonging to the arm and hand are placed in a custom group in order
    # to disable their visibility to the top-down camera. NB: we assume that
    # there are no other geoms in ROBOT_GEOM_GROUP that don't belong to the
    # robot (this is usually the case since the default geom group is 0). If
    # there are then these will also be invisible to the top-down camera.
    for robot_geom in arm.mjcf_model.find_all('geom'):
      robot_geom.group = arenas.ROBOT_GEOM_GROUP

    self._arena = arena
    self._board = board
    self._arm = arm
    self._hand = hand
    self._markers = markers
    self._tcp_initializer = initializers.ToolCenterPointInitializer(
        hand=hand, arm=arm,
        position=distributions.Uniform(_TCP_LOWER_BOUNDS, _TCP_UPPER_BOUNDS),
        quaternion=_uniform_downward_rotation())

    # Add an observable exposing the logical state of the board.
    board_state_observable = observable.Generic(
        lambda physics: self._game_logic.get_board_state())
    board_state_observable.configure(
        **observation_settings.board_state._asdict())
    self._task_observables = {'board_state': board_state_observable}

  @property
  def root_entity(self):
    return self._arena

  @property
  def arm(self):
    return self._arm

  @property
  def hand(self):
    return self._hand

  @property
  def task_observables(self):
    return self._task_observables

  def get_reward(self, physics):
    del physics  # Unused.
    return self._game_logic.get_reward[SELF]

  def should_terminate_episode(self, physics):
    return self._game_logic.is_game_over

  def initialize_episode(self, physics, random_state):
    self._tcp_initializer(physics, random_state)
    self._game_logic.reset()
    self._game_opponent.reset()

  def before_step(self, physics, action, random_state):
    super(JacoArmBoardGame, self).before_step(physics, action, random_state)
    self._made_move_this_step = False

  def after_substep(self, physics, random_state):
    raise NotImplementedError('Subclass must implement after_substep.')
