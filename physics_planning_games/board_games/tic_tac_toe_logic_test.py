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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from physics_planning_games.board_games import tic_tac_toe_logic


class TicTacToeGameLogicTest(parameterized.TestCase):

  def setUp(self):
    super(TicTacToeGameLogicTest, self).setUp()
    self.logic = tic_tac_toe_logic.TicTacToeGameLogic()
    self.expected_board_state = np.zeros((3, 3, 3), dtype=bool)
    self.expected_board_state[..., 0] = True  # All positions initially empty.

  def test_valid_move_sequence(self):
    np.testing.assert_array_equal(self.logic.get_board_state(),
                                  self.expected_board_state)

    action = tic_tac_toe_logic.SingleMarkerAction(col=1, row=2)
    self.assertTrue(self.logic.apply(player=0, action=action),
                    msg='Invalid action: {}'.format(action))
    self.expected_board_state[action.row, action.col, 0] = False
    self.expected_board_state[action.row, action.col, 1] = True
    np.testing.assert_array_equal(self.logic.get_board_state(),
                                  self.expected_board_state)

    action = tic_tac_toe_logic.SingleMarkerAction(col=0, row=1)
    self.assertTrue(self.logic.apply(player=1, action=action),
                    msg='Invalid action: {}'.format(action))
    self.expected_board_state[action.row, action.col, 0] = False
    self.expected_board_state[action.row, action.col, 2] = True
    np.testing.assert_array_equal(self.logic.get_board_state(),
                                  self.expected_board_state)

  def test_invalid_move_sequence(self):
    np.testing.assert_array_equal(self.logic.get_board_state(),
                                  self.expected_board_state)
    action = tic_tac_toe_logic.SingleMarkerAction(col=1, row=2)
    self.assertTrue(self.logic.apply(player=0, action=action),
                    msg='Invalid action: {}'.format(action))
    self.expected_board_state[action.row, action.col, 0] = False
    self.expected_board_state[action.row, action.col, 1] = True
    np.testing.assert_array_equal(self.logic.get_board_state(),
                                  self.expected_board_state)

    # Player 0 tries to move again in the same location.
    action = tic_tac_toe_logic.SingleMarkerAction(col=1, row=2)
    self.assertFalse(self.logic.apply(player=0, action=action),
                     msg='Invalid action was accepted: {}'.format(action))

    # Player 1 tries to move in the same location as player 0.
    self.assertFalse(self.logic.apply(player=1, action=action),
                     msg='Invalid action was accepted: {}'.format(action))

    # The board state should not have changed as a result of invalid actions.
    np.testing.assert_array_equal(self.logic.get_board_state(),
                                  self.expected_board_state)

  @parameterized.named_parameters([
      dict(testcase_name='player_0_win',
           move_sequence=((0, 0, 0),
                          (1, 0, 1),
                          (0, 1, 0),
                          (1, 2, 1),
                          (0, 2, 0)),
           winner_id=0),
      dict(testcase_name='player_1_win',
           move_sequence=((0, 0, 0),
                          (1, 0, 2),
                          (0, 1, 0),
                          (1, 1, 1),
                          (0, 0, 1),
                          (1, 2, 0)),
           winner_id=1),
      dict(testcase_name='draw',
           move_sequence=((0, 0, 0),
                          (1, 1, 1),
                          (0, 1, 0),
                          (1, 2, 0),
                          (0, 0, 2),
                          (1, 0, 1),
                          (0, 2, 1),
                          (1, 2, 2),
                          (0, 1, 2)),
           winner_id=None)])
  def test_reward_and_termination(self, move_sequence, winner_id):
    for (player_id, row, col) in move_sequence:
      self.assertFalse(self.logic.is_game_over)
      self.assertDictEqual(self.logic.get_reward, {0: 0.0, 1: 0.0})
      action = tic_tac_toe_logic.SingleMarkerAction(col=col, row=row)
      self.assertTrue(self.logic.apply(player=player_id, action=action),
                      msg='Invalid action: {}'.format(action))
    self.assertTrue(self.logic.is_game_over)
    rewards = self.logic.get_reward
    if winner_id is not None:
      loser_id = 1 - winner_id
      self.assertDictEqual(rewards, {winner_id: 1.0, loser_id: 0.0})
    else:  # Draw
      self.assertDictEqual(rewards, {0: 0.5, 1: 0.5})

  def test_random_opponent_vs_optimal(self):
    """Play random v optimal opponents and check that optimal largely wins.
    """
    rand_state = np.random.RandomState(42)
    optimal_opponent = tic_tac_toe_logic.TicTacToeOptimalOpponent()
    random_opponent = tic_tac_toe_logic.TicTacToeRandomOpponent()
    players = [optimal_opponent, random_opponent]
    optimal_returns = []
    random_returns = []

    for _ in range(20):
      logic = tic_tac_toe_logic.TicTacToeGameLogic()
      optimal_opponent.reset()
      random_opponent.reset()

      rand_state.shuffle(players)
      current_player_idx = 0

      while not logic.is_game_over:
        current_player = players[current_player_idx]
        action = current_player.policy(logic, rand_state)
        self.assertTrue(logic.apply(current_player_idx, action),
                        msg='Opponent {} selected invalid action {}'.format(
                            current_player, action))
        current_player_idx = (current_player_idx + 1) % 2

      # Record the winner.
      reward = logic.get_reward
      if players[0] == optimal_opponent:
        optimal_return = reward[0]
        random_return = reward[1]
      else:
        optimal_return = reward[1]
        random_return = reward[0]
      optimal_returns.append(optimal_return)
      random_returns.append(random_return)

    mean_optimal_returns = np.mean(optimal_returns)
    mean_random_returns = np.mean(random_returns)
    self.assertGreater(mean_optimal_returns, 0.9)
    self.assertLess(mean_random_returns, 0.1)

  @parameterized.named_parameters([
      dict(testcase_name='pos0',
           move_sequence=((0, 0, 1),
                          (1, 1, 1),
                          (0, 0, 2),
                          (1, 1, 2)),
           optimal_move=(0, 0)),
      dict(testcase_name='pos1',
           move_sequence=((0, 0, 1),
                          (1, 1, 2),
                          (0, 0, 2),
                          (1, 1, 1)),
           optimal_move=(0, 0)),
      dict(testcase_name='pos2',
           move_sequence=((0, 2, 1),
                          (1, 1, 2),
                          (0, 2, 2),
                          (1, 1, 1)),
           optimal_move=(2, 0)),
  ])
  def test_minimax_policy(self, move_sequence, optimal_move):
    rand_state = np.random.RandomState(42)
    for (player_id, row, col) in move_sequence:
      action = tic_tac_toe_logic.SingleMarkerAction(col=col, row=row)
      self.assertTrue(self.logic.apply(player=player_id, action=action),
                      msg='Invalid action: {}'.format(action))

    state = self.logic.open_spiel_state
    planner_action = tic_tac_toe_logic.tic_tac_toe_minimax(state,
                                                           rand_state)
    self.assertEqual(planner_action, optimal_move)

    # Do the same but with np array as input
    self.logic = tic_tac_toe_logic.TicTacToeGameLogic()
    for (player_id, row, col) in move_sequence:
      action = tic_tac_toe_logic.SingleMarkerAction(col=col, row=row)
      self.assertTrue(self.logic.apply(player=player_id, action=action),
                      msg='Invalid action: {}'.format(action))

    board = self.logic.get_board_state()
    planner_action = tic_tac_toe_logic.tic_tac_toe_minimax(board,
                                                           rand_state)
    self.assertEqual(planner_action, optimal_move)

if __name__ == '__main__':
  absltest.main()
