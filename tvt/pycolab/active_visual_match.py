# Lint as: python2, python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Active visual match task.

The game is split up into three phases:
1. (exploration phase) player is in one room and there's a colour in the other,
2. (distractor phase) player is collecting apples,
3. (reward phase) player sees three doors of different colours and has to select
    the one of the same color as the colour in the first phase.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pycolab import ascii_art
from pycolab import storytelling

from tvt.pycolab import common
from tvt.pycolab import game
from tvt.pycolab import objects


SYMBOLS_TO_SHUFFLE = ['b', 'c', 'e']

EXPLORE_GRID = [
    '  ppppppp  ',
    '  p     p  ',
    '  p     p  ',
    '  pp   pp  ',
    '  p+++++p  ',
    '  p+++++p  ',
    '  ppppppp  '
]

REWARD_GRID = [
    '###########',
    '# b  c  e #',
    '#         #',
    '#         #',
    '####   ####',
    '   # + #   ',
    '   #####   '
]


class Game(game.AbstractGame):
  """Image Match Passive Game."""

  def __init__(self,
               rng,
               num_apples=10,
               apple_reward=(1, 10),
               fix_apple_reward_in_episode=True,
               final_reward=10.,
               max_frames=common.DEFAULT_MAX_FRAMES_PER_PHASE):
    self._rng = rng
    self._num_apples = num_apples
    self._apple_reward = apple_reward
    self._fix_apple_reward_in_episode = fix_apple_reward_in_episode
    self._final_reward = final_reward
    self._max_frames = max_frames
    self._episode_length = sum(self._max_frames.values())
    self._num_actions = common.NUM_ACTIONS
    self._colours = common.FIXED_COLOURS.copy()
    self._colours.update(
        common.get_shuffled_symbol_colour_map(rng, SYMBOLS_TO_SHUFFLE))

    self._extra_observation_fields = ['chapter_reward_as_string']

  @property
  def extra_observation_fields(self):
    """The field names of extra observations."""
    return self._extra_observation_fields

  @property
  def num_actions(self):
    """Number of possible actions in the game."""
    return self._num_actions

  @property
  def episode_length(self):
    return self._episode_length

  @property
  def colours(self):
    """Symbol to colour map for key to door."""
    return self._colours

  def _make_explore_phase(self, target_char):
    # Keep only one coloured position and one player position.
    grid = common.keep_n_characters_in_grid(EXPLORE_GRID, 'p', 1, common.BORDER)
    grid = common.keep_n_characters_in_grid(grid, 'p', 0, target_char)
    grid = common.keep_n_characters_in_grid(grid, common.PLAYER, 1)

    return ascii_art.ascii_art_to_game(
        grid,
        what_lies_beneath=' ',
        sprites={
            common.PLAYER:
                ascii_art.Partial(
                    common.PlayerSprite,
                    impassable=common.BORDER + target_char),
            target_char:
                objects.ObjectSprite,
            common.TIMER:
                ascii_art.Partial(common.TimerSprite,
                                  self._max_frames['explore']),
        },
        update_schedule=[common.PLAYER, target_char, common.TIMER],
        z_order=[target_char, common.PLAYER, common.TIMER],
    )

  def _make_distractor_phase(self):
    return common.distractor_phase(
        player_sprite=common.PlayerSprite,
        num_apples=self._num_apples,
        max_frames=self._max_frames['distractor'],
        apple_reward=self._apple_reward,
        fix_apple_reward_in_episode=self._fix_apple_reward_in_episode)

  def _make_reward_phase(self, target_char):
    return ascii_art.ascii_art_to_game(
        REWARD_GRID,
        what_lies_beneath=' ',
        sprites={
            common.PLAYER: common.PlayerSprite,
            'b': objects.ObjectSprite,
            'c': objects.ObjectSprite,
            'e': objects.ObjectSprite,
            common.TIMER: ascii_art.Partial(common.TimerSprite,
                                            self._max_frames['reward'],
                                            track_chapter_reward=True),
            target_char: ascii_art.Partial(objects.ObjectSprite,
                                           reward=self._final_reward),
        },
        update_schedule=[common.PLAYER, 'b', 'c', 'e', common.TIMER],
        z_order=[common.PLAYER, 'b', 'c', 'e', common.TIMER],
    )

  def make_episode(self):
    """Factory method for generating new episodes of the game."""
    target_char = self._rng.choice(SYMBOLS_TO_SHUFFLE)
    return storytelling.Story([
        lambda: self._make_explore_phase(target_char),
        self._make_distractor_phase,
        lambda: self._make_reward_phase(target_char),
    ], croppers=common.get_cropper())
