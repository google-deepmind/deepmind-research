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

"""TicTacToe logic wrapper for use in manipulation tasks."""

import collections
import itertools

import numpy as np

from physics_planning_games.board_games import logic_base
from open_spiel.python.algorithms import minimax
import pyspiel


SingleMarkerAction = collections.namedtuple('SingleMarkerAction',
                                            ['row', 'col'])
force_random_start_position = False


class TicTacToeGameLogic(logic_base.OpenSpielBasedLogic):
  """Logic for TicTacToe game."""

  def __init__(self):
    self.reset()

  def reset(self):
    """Resets the game state."""
    # For now we always assume we are the starting player.

    game = pyspiel.load_game('tic_tac_toe')
    self._open_spiel_state = game.new_initial_state()

    if force_random_start_position:
      # For debugging purposes only, force some random moves
      rand_state = np.random.RandomState(46)
      rand_player = TicTacToeRandomOpponent()
      num_moves = 4
      for _ in range(num_moves):
        action = rand_player.policy(self, rand_state)
        action_1d = np.ravel_multi_index(action, (3, 3))
        self._open_spiel_state.apply_action(action_1d)

  def get_board_state(self):
    """Returns the logical board state as a numpy array.

    Returns:
      A boolean array of shape (H, W, C), where H=3, W=3 (height and width
      of the board) and C=3 for the 3 planes. The 3 planes are, in order,
      unmarked squares, x's (player 0) and y's (player 1).
    """
    board_state = np.reshape(
        np.array(self._open_spiel_state.observation_tensor(0), dtype=bool),
        [3, 3, 3])
    board_state = np.transpose(board_state, [1, 2, 0])
    board_state = board_state[:, :, [0, 2, 1]]
    return board_state

  def apply(self, player, action):
    """Checks whether action is valid, and if so applies it to the game state.

    Args:
      player: Integer specifying the player ID; either 0 or 1.
      action: A `SingleMarkerAction` instance.

    Returns:
      True if the action was valid, else False.
    """
    action_value = np.ravel_multi_index((action.row, action.col), (3, 3))
    if self._open_spiel_state.current_player() != player:
      return False

    try:
      self._open_spiel_state.apply_action(action_value)
      was_valid_move = True
    except RuntimeError:
      was_valid_move = False

    return was_valid_move


class TicTacToeRandomOpponent(logic_base.Opponent):
  """An easy opponent for TicTacToe."""

  def __init__(self):
    pass

  def reset(self):
    """Resets the opponent's internal state (not implemented)."""
    pass

  def policy(self, game_logic, random_state):
    """Return a random, valid move.

    Args:
      game_logic: TicTacToeGameLogic state of the game.
      random_state: An instance of `np.random.RandomState`

    Returns:
      SingleMarkerAction of opponent.
    """
    if game_logic.is_game_over:
      return None

    valid_moves = game_logic.open_spiel_state.legal_actions()
    assert valid_moves
    move = random_state.choice(valid_moves)
    row, col = np.unravel_index(move, shape=(3, 3))
    return SingleMarkerAction(row=row, col=col)


class TicTacToeMixtureOpponent(logic_base.Opponent):
  """A TicTacToe opponent which makes a mixture of optimal and random moves.

  The optimal mixture component uses minimax search.
  """

  def __init__(self, mixture_p):
    """Initialize the mixture opponent.

    Args:
      mixture_p: The mixture probability. We choose moves from the random
        opponent with probability mixture_p and moves from the optimal
        opponent with probability 1 - mixture_p.
    """

    self._random_opponent = TicTacToeRandomOpponent()
    self._optimal_opponent = TicTacToeOptimalOpponent()
    self._mixture_p = mixture_p

  def reset(self):
    pass

  def policy(self, game_logic, random_state):
    if random_state.rand() < self._mixture_p:
      return self._random_opponent.policy(game_logic, random_state)
    else:
      return self._optimal_opponent.policy(game_logic, random_state)


class TicTacToeOptimalOpponent(logic_base.Opponent):
  """A TicTacToe opponent which makes perfect moves.

  Uses minimax search.
  """

  def __init__(self):
    pass

  def reset(self):
    pass

  def policy(self, game_logic, random_state):
    action = tic_tac_toe_minimax(game_logic.open_spiel_state, random_state)
    return action


def numpy_array_to_open_spiel_state(board_state):
  """Take a numpy observation [3x3x3] bool area and create an OpenSpiel state.

  Args:
    board_state: 3x3x3 bool array with [col, row, c] with c indexing, in order,
      empty squares, x moves, y moves.

  Returns:
    open_spiel_state: OpenSpiel state of this position.
  """
  game = pyspiel.load_game('tic_tac_toe')
  open_spiel_state = game.new_initial_state()

  x_moves = np.flatnonzero(board_state[:, :, 1])
  y_moves = np.flatnonzero(board_state[:, :, 2])

  for x_m, y_m in itertools.zip_longest(x_moves, y_moves):
    if open_spiel_state.is_terminal():
      break
    open_spiel_state.apply_action(x_m)
    if open_spiel_state.is_terminal():
      break
    if y_m is not None:
      open_spiel_state.apply_action(y_m)

  return open_spiel_state


def open_spiel_move_to_single_marker_action(action):
  row, col = np.unravel_index(action, shape=(3, 3))
  return SingleMarkerAction(row=row, col=col)


def tic_tac_toe_random_move(state, random_state):
  """Returns a legal move at random from current state.

  Args:
    state: World state of the game. Either an OpenSpiel state
      or a numpy encoding of the board.
    random_state: numpy random state used for choosing randomly if there is more
      than one optimal action.

  Returns:
    action: SingleMarkerAction of a random move.
  """
  if isinstance(state, np.ndarray):
    spiel_state = numpy_array_to_open_spiel_state(state)
  else:
    spiel_state = state
  if spiel_state.is_terminal():
    return False

  legal_actions = spiel_state.legal_actions()
  action = random_state.choice(legal_actions)
  return open_spiel_move_to_single_marker_action(action)


def tic_tac_toe_minimax(state, random_state):
  """Tree search from the world_state in order to find the optimal action.

  Args:
    state: World state of the game. Either an OpenSpiel state
      or a numpy encoding of the board.
    random_state: numpy random state used for choosing randomly if there is more
      than one optimal action.

  Returns:
    action: SingleMarkerAction of an optimal move.
  """
  if isinstance(state, np.ndarray):
    spiel_state = numpy_array_to_open_spiel_state(state)
  else:
    spiel_state = state
  if spiel_state.is_terminal():
    return False

  current_player = spiel_state.current_player()
  legal_actions = spiel_state.legal_actions()
  best_actions = []
  best_value = -100

  for action in legal_actions:
    state_after_action = spiel_state.clone()
    state_after_action.apply_action(action)
    value, _ = minimax.expectiminimax(state_after_action, 100, None,
                                      current_player)
    if value > best_value:
      best_value = value
      best_actions = [action]
    elif value == best_value:
      best_actions.append(action)

  assert best_actions
  action = random_state.choice(best_actions)

  return open_spiel_move_to_single_marker_action(action)
