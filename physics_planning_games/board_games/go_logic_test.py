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

from physics_planning_games.board_games import go_logic


class GoGameLogicTest(parameterized.TestCase):

  def setUp(self):
    super(GoGameLogicTest, self).setUp()
    self.logic = go_logic.GoGameLogic(board_size=5)
    self.expected_board_state = np.zeros((5, 5, 4), dtype=bool)
    self.expected_board_state[:, :, 0] = True

  def test_valid_move_sequence(self):
    np.testing.assert_array_equal(self.logic.get_board_state(),
                                  self.expected_board_state)

    action = go_logic.GoMarkerAction(col=1, row=2, pass_action=False)
    self.assertTrue(self.logic.apply(player=0, action=action),
                    msg='Invalid action: {}'.format(action))

  def test_pass(self):
    action = go_logic.GoMarkerAction(col=0, row=0, pass_action=True)
    self.assertTrue(self.logic.apply(player=0, action=action),
                    msg='Invalid action: {}'.format(action))
    self.expected_board_state[:, :, 3] = True
    np.testing.assert_array_equal(self.logic.get_board_state(),
                                  self.expected_board_state)

  def test_invalid_move_sequence(self):
    np.testing.assert_array_equal(self.logic.get_board_state(),
                                  self.expected_board_state)
    action = go_logic.GoMarkerAction(col=1, row=2, pass_action=False)
    self.assertTrue(self.logic.apply(player=0, action=action),
                    msg='Invalid action: {}'.format(action))
    self.expected_board_state[action.row, action.col, 0] = False
    self.expected_board_state[action.row, action.col, 1] = True
    self.expected_board_state[:, :, 3] = True
    np.testing.assert_array_equal(self.logic.get_board_state(),
                                  self.expected_board_state)

    action = go_logic.GoMarkerAction(col=1, row=2, pass_action=False)
    self.assertFalse(self.logic.apply(player=0, action=action),
                     msg='Invalid action was accepted: {}'.format(action))

    # Player 1 tries to move in the same location as player 0.
    self.assertFalse(self.logic.apply(player=1, action=action),
                     msg='Invalid action was accepted: {}'.format(action))

    # The board state should not have changed as a result of invalid actions.
    np.testing.assert_array_equal(self.logic.get_board_state(),
                                  self.expected_board_state)

  def test_random_opponent_vs_gnugo(self):
    """Play random v gnugo opponents and check that optimal largely wins.
    """
    board_size = 9
    rand_state = np.random.RandomState(42)
    pachi_opponent = go_logic.GoGTPOpponent(board_size)
    random_opponent = go_logic.GoRandomOpponent(board_size)
    players = [pachi_opponent, random_opponent]
    pachi_returns = []
    random_returns = []

    for _ in range(3):
      logic = go_logic.GoGameLogic(board_size)
      pachi_opponent.reset()
      random_opponent.reset()

      rand_state.shuffle(players)
      current_player_idx = 0

      while not logic.is_game_over:
        current_player = players[current_player_idx]
        action = current_player.policy(logic, current_player_idx, rand_state)
        valid_action = logic.apply(current_player_idx, action)
        self.assertTrue(valid_action,
                        msg='Opponent {} selected invalid action {}'.format(
                            current_player, action))
        current_player_idx = (current_player_idx + 1) % 2

      # Record the winner.
      reward = logic.get_reward
      if players[0] == pachi_opponent:
        pachi_return = reward[0]
        random_return = reward[1]
      else:
        pachi_return = reward[1]
        random_return = reward[0]
      pachi_returns.append(pachi_return)
      random_returns.append(random_return)

    mean_pachi_returns = np.mean(pachi_returns)
    mean_random_returns = np.mean(random_returns)
    self.assertGreater(mean_pachi_returns, 0.95)
    self.assertLess(mean_random_returns, 0.05)

  @parameterized.named_parameters([
      dict(testcase_name='00',
           row=0, col=0),
      dict(testcase_name='01',
           row=1, col=0)])
  def test_go_marker_to_int(self, row, col):
    go_marker = go_logic.GoMarkerAction(row=row, col=col, pass_action=False)
    int_action = go_logic._go_marker_to_int(go_marker, board_size=19)
    recovered_go_marker = go_logic._int_to_go_marker(int_action, board_size=19)
    self.assertEqual(go_marker, recovered_go_marker,
                     msg='Initial go marker {}, recovered {}'.format(
                         go_marker, recovered_go_marker))

  @parameterized.named_parameters([
      dict(testcase_name='00',
           row=0, col=0),
      dict(testcase_name='01',
           row=1, col=0)])
  def test_go_marker_to_str(self, row, col):
    go_marker = go_logic.GoMarkerAction(row=row, col=col, pass_action=False)
    str_action = go_logic._go_marker_to_str(go_marker)
    recovered_go_marker = go_logic._str_to_go_marker(str_action)
    self.assertEqual(go_marker,
                     recovered_go_marker,
                     msg='Initial go marker {}, recovered {}, '
                         'str_action {}'.format(go_marker, recovered_go_marker,
                                                str_action))


if __name__ == '__main__':
  absltest.main()
