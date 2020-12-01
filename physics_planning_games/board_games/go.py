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

"""A Go board game."""

from dm_control.composer.observation import observable
import numpy as np
from physics_planning_games.board_games import go_logic
from physics_planning_games.board_games import jaco_arm_board_game
from physics_planning_games.board_games._internal import boards
from physics_planning_games.board_games._internal import observations
from physics_planning_games.board_games._internal import pieces
from physics_planning_games.board_games._internal import registry
from physics_planning_games.board_games._internal import tags

_BLACK = (0., 0., 0., 0.75)
_WHITE = (1., 1., 1., 0.75)

_GO_PIECE_SIZE = 0.04
_DEFAULT_OPPONENT_MIXTURE = 0.2


class Go(jaco_arm_board_game.JacoArmBoardGame):
  """Single-player Go of configurable size."""

  def __init__(self, board_size, observation_settings, opponent=None,
               reset_arm_after_move=True):
    """Initializes a `Go` task.

    Args:
      board_size: board size
      observation_settings: An `observations.ObservationSettings` namedtuple
        specifying configuration options for each category of observation.
      opponent: Go opponent to use for the opponent player actions.
      reset_arm_after_move: Whether to reset arm to random position after every
        piece being placed on the board.
    """
    game_logic = go_logic.GoGameLogic(board_size=board_size)

    if opponent is None:
      opponent = go_logic.GoGTPOpponent(board_size=board_size,
                                        mixture_p=_DEFAULT_OPPONENT_MIXTURE)

    self._last_valid_move_is_pass = False
    super(Go, self).__init__(observation_settings=observation_settings,
                             opponent=opponent,
                             game_logic=game_logic,
                             board=boards.GoBoard(boardsize=board_size),
                             markers=pieces.Markers(
                                 player_colors=(_BLACK, _WHITE),
                                 halfwidth=_GO_PIECE_SIZE,
                                 num_per_player=board_size*board_size*2,
                                 observable_options=observations.make_options(
                                     observation_settings,
                                     observations.MARKER_OBSERVABLES),
                                 board_size=board_size))
    self._reset_arm_after_move = reset_arm_after_move
    # Add an observable exposing the move history (to reconstruct game states)
    move_history_observable = observable.Generic(
        lambda physics: self._game_logic.get_move_history())
    move_history_observable.configure(
        **observation_settings.board_state._asdict())
    self._task_observables['move_history'] = move_history_observable

  @property
  def name(self):
    return 'Go'

  @property
  def control_timestep(self):
    return 0.05

  def after_substep(self, physics, random_state):
    if not self._made_move_this_step:
      # which board square received the most contact pressure
      indices = self._board.get_contact_indices(physics)
      if not indices:
        return
      row, col = indices
      # Makes sure that contact with that board square involved a finger
      finger_touch = self._board.validate_finger_touch(physics,
                                                       row, col, self._hand)
      if not finger_touch:
        return

      pass_action = True if (row == -1 and col == -1) else False
      if pass_action and self._last_valid_move_is_pass:
        # Don't allow two passes in a row (otherwise hard to only pass once)
        valid_move = False
      else:
        valid_move = self._game_logic.apply(
            player=jaco_arm_board_game.SELF,
            action=go_logic.GoMarkerAction(row=int(row), col=int(col),
                                           pass_action=pass_action))

      if valid_move:
        self._made_move_this_step = True
        if not pass_action:
          self._last_valid_move_is_pass = False
          marker_pos = self._board.get_contact_pos(
              physics=physics, row=row, col=col)
          self._markers.mark(physics=physics,
                             player_id=jaco_arm_board_game.SELF,
                             pos=marker_pos,
                             bpos=(row, col))
        else:
          self._last_valid_move_is_pass = True
        if not self._game_logic.is_game_over:
          opponent_move = self._game_opponent.policy(
              game_logic=self._game_logic, player=jaco_arm_board_game.OPPONENT,
              random_state=random_state)
          assert opponent_move
          assert self._game_logic.apply(player=jaco_arm_board_game.OPPONENT,
                                        action=opponent_move)
          marker_pos = self._board.sample_pos_inside_touch_sensor(
              physics=physics,
              random_state=random_state,
              row=opponent_move.row,
              col=opponent_move.col)
          self._markers.mark(physics=physics,
                             player_id=jaco_arm_board_game.OPPONENT,
                             pos=marker_pos,
                             bpos=(opponent_move.row,
                                   opponent_move.col))
        if self._reset_arm_after_move:
          self._tcp_initializer(physics, random_state)

        # Redraw all markers that are on the board (after captures)
        self._markers.make_all_invisible(physics)
        board = self._game_logic.get_board_state()
        black_stones = np.transpose(np.nonzero(board[:, :, 1]))
        white_stones = np.transpose(np.nonzero(board[:, :, 2]))
        if black_stones.size > 0:
          self._markers.make_visible_by_bpos(physics, 0, black_stones)
        if white_stones.size > 0:
          self._markers.make_visible_by_bpos(physics, 1, white_stones)


@registry.add(tags.EASY, tags.FEATURES)
def go_7x7():
  return Go(board_size=7,
            observation_settings=observations.PERFECT_FEATURES)
