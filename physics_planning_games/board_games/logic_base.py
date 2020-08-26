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

"""Abstract base classes and utility functions for logical aspects of the games.
"""

import abc

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class GameLogic(ABC):
  """Define the abstrat game logic class.
  """

  @abc.abstractmethod
  def __init__(self):
    pass

  @abc.abstractmethod
  def reset(self):
    pass

  @abc.abstractproperty
  def is_game_over(self):
    """Boolean specifying whether the current game has ended."""

  @abc.abstractproperty
  def get_reward(self):
    pass

  @abc.abstractmethod
  def get_board_state(self):
    """Returns the logical board state as a numpy array."""

  @abc.abstractmethod
  def apply(self, player, action):
    """Checks whether action is valid, and if so applies it to the game state.

    Args:
      player: Integer specifying the player ID; either 0 or 1.
      action: A `GoMarkerAction` instance.

    Returns:
      True if the action was valid, else False.
    """


class OpenSpielBasedLogic(GameLogic):
  """GameLogic using OpenSpiel for tracking game state.
  """

  @property
  def is_game_over(self):
    """Boolean specifying whether the current game has ended."""
    return  self._open_spiel_state.is_terminal()

  @property
  def get_reward(self):
    """Returns a dictionary that maps from `{player_id: player_reward}`."""

    if self.is_game_over:
      player0_return = self._open_spiel_state.player_return(0)
      # Translate from OpenSpiel returns to 0.5 for draw, -1 for loss,
      # +1 for win.
      if player0_return == 0.:
        reward = {0: 0.5, 1: 0.5}
      elif player0_return == 1.:
        reward = {0: 1., 1: 0.}
      else:
        assert player0_return == -1.
        reward = {0: 0., 1: 1.}
    else:
      reward = {0: 0.,
                1: 0.}
    return reward

  @property
  def open_spiel_state(self):
    """OpenSpiel object representing the underlying game state."""
    return self._open_spiel_state


class Opponent(ABC):
  """Abstract Opponent class."""

  @abc.abstractmethod
  def __init__(self):
    pass

  @abc.abstractmethod
  def reset(self):
    pass

  @abc.abstractmethod
  def policy(self, game_logic, random_state):
    """Return policy action.

    Args:
      game_logic: Go game logic state.
      random_state: Numpy random state object.
    Returns:
      NamedTuple indicating opponent move.
    """
