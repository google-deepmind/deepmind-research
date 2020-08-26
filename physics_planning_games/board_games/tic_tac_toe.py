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

"""A Tic Tac Toe task."""

from physics_planning_games.board_games import jaco_arm_board_game
from physics_planning_games.board_games import tic_tac_toe_logic
from physics_planning_games.board_games._internal import boards
from physics_planning_games.board_games._internal import observations
from physics_planning_games.board_games._internal import pieces
from physics_planning_games.board_games._internal import registry
from physics_planning_games.board_games._internal import tags


class TicTacToe(jaco_arm_board_game.JacoArmBoardGame):
  """Single-player Tic Tac Toe."""

  def __init__(self, observation_settings, opponent=None,
               reset_arm_after_move=True):
    """Initializes a `TicTacToe` task.

    Args:
      observation_settings: An `observations.ObservationSettings` namedtuple
        specifying configuration options for each category of observation.
      opponent: TicTacToeOpponent used for generating opponent moves.
      reset_arm_after_move: Whether to reset arm to random position after every
        piece being placed on the board.
    """
    game_logic = tic_tac_toe_logic.TicTacToeGameLogic()
    if opponent is None:
      opponent = tic_tac_toe_logic.TicTacToeRandomOpponent()

    markers = pieces.Markers(num_per_player=5,
                             observable_options=observations.make_options(
                                 observation_settings,
                                 observations.MARKER_OBSERVABLES))
    self._reset_arm_after_move = reset_arm_after_move
    super(TicTacToe, self).__init__(observation_settings=observation_settings,
                                    opponent=opponent,
                                    game_logic=game_logic,
                                    board=boards.CheckerBoard(),
                                    markers=markers)

  @property
  def control_timestep(self):
    return 0.05

  def after_substep(self, physics, random_state):
    if not self._made_move_this_step:
      indices = self._board.get_contact_indices(physics)
      if not indices:
        return
      row, col = indices
      valid_move = self._game_logic.apply(
          player=jaco_arm_board_game.SELF,
          action=tic_tac_toe_logic.SingleMarkerAction(row=row, col=col))
      if valid_move:
        self._made_move_this_step = True
        marker_pos = self._board.get_contact_pos(
            physics=physics, row=row, col=col)
        self._markers.mark(physics=physics, player_id=jaco_arm_board_game.SELF,
                           pos=marker_pos)
        if not self._game_logic.is_game_over:
          opponent_move = self._game_opponent.policy(
              game_logic=self._game_logic, random_state=random_state)
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
                             pos=marker_pos)
          if self._reset_arm_after_move:
            self._tcp_initializer(physics, random_state)


@registry.add(tags.EASY, tags.FEATURES)
def tic_tac_toe_markers_features(**unused_kwargs):
  return TicTacToe(observation_settings=observations.PERFECT_FEATURES)


@registry.add(tags.MED, tags.FEATURES)
def tic_tac_toe_mixture_opponent_markers_features(mixture_p=0.25):
  print('Creating tictactoe task with random/optimal opponent mixture, p={}'
        .format(mixture_p))
  return TicTacToe(
      observation_settings=observations.PERFECT_FEATURES,
      opponent=tic_tac_toe_logic.TicTacToeMixtureOpponent(mixture_p))


@registry.add(tags.HARD, tags.FEATURES)
def tic_tac_toe_optimal_opponent_markers_features(**unused_kwargs):
  return TicTacToe(observation_settings=observations.PERFECT_FEATURES,
                   opponent=tic_tac_toe_logic.TicTacToeOptimalOpponent())
