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
"""Key to door task.

The game is split up into three phases:
1. (exploration phase) player can collect a key,
2. (distractor phase) player is collecting apples,
3. (reward phase) player can open the door and get the reward if the key is
    previously collected.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pycolab import ascii_art
from pycolab import storytelling
from pycolab import things as plab_things

from tvt.pycolab import common
from tvt.pycolab import game
from tvt.pycolab import objects


COLOURS = {
    'i': (1000, 1000, 1000),  # Indicator.
}

EXPLORE_GRID = [
    '  #######  ',
    '  #kkkkk#  ',
    '  #kkkkk#  ',
    '  ##   ##  ',
    '  #+++++#  ',
    '  #+++++#  ',
    '  #######  '
]

REWARD_GRID = [
    '           ',
    '   ##d##   ',
    '   #   #   ',
    '   # + #   ',
    '   #   #   ',
    '   #####   ',
    '           ',
]


class KeySprite(plab_things.Sprite):
  """Sprite for the key."""

  def update(self, actions, board, layers, backdrop, things, the_plot):
    player_position = things[common.PLAYER].position
    pick_up = self.position == player_position

    if self.visible and pick_up:
      # Pass information to all phases.
      the_plot['has_key'] = True
      self._visible = False


class DoorSprite(plab_things.Sprite):
  """Sprite for the door."""

  def __init__(self, corner, position, character, pickup_reward):
    super(DoorSprite, self).__init__(corner, position, character)
    self._pickup_reward = pickup_reward

  def update(self, actions, board, layers, backdrop, things, the_plot):
    player_position = things[common.PLAYER].position
    pick_up = self.position == player_position

    if pick_up and the_plot.get('has_key'):
      the_plot.add_reward(self._pickup_reward)
      # The key is lost after the first time opening the door
      # to ensure only one reward per episode.
      the_plot['has_key'] = False


class PlayerSprite(common.PlayerSprite):
  """Sprite for the actor."""

  def __init__(self, corner, position, character):
    super(PlayerSprite, self).__init__(
        corner, position, character,
        impassable=common.BORDER + common.INDICATOR + common.DOOR)

  def update(self, actions, board, layers, backdrop, things, the_plot):

    # Allow moving through the door if key is previously collected.
    if common.DOOR in self.impassable and the_plot.get('has_key'):
      self._impassable.remove(common.DOOR)

    super(PlayerSprite, self).update(actions, board, layers, backdrop, things,
                                     the_plot)


class Game(game.AbstractGame):
  """Key To Door Game."""

  def __init__(self,
               rng,
               num_apples=10,
               apple_reward=(1, 10),
               fix_apple_reward_in_episode=True,
               final_reward=10.,
               crop=True,
               max_frames=common.DEFAULT_MAX_FRAMES_PER_PHASE):
    del rng  # Each episode is identical and colours are not randomised.
    self._num_apples = num_apples
    self._apple_reward = apple_reward
    self._fix_apple_reward_in_episode = fix_apple_reward_in_episode
    self._final_reward = final_reward
    self._crop = crop
    self._max_frames = max_frames
    self._episode_length = sum(self._max_frames.values())
    self._num_actions = common.NUM_ACTIONS
    self._colours = common.FIXED_COLOURS.copy()
    self._colours.update(COLOURS)
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

  def _make_explore_phase(self):
    # Keep only one key and one player position.
    explore_grid = common.keep_n_characters_in_grid(
        EXPLORE_GRID, common.KEY, 1)
    explore_grid = common.keep_n_characters_in_grid(
        explore_grid, common.PLAYER, 1)
    return ascii_art.ascii_art_to_game(
        art=explore_grid,
        what_lies_beneath=' ',
        sprites={
            common.PLAYER: PlayerSprite,
            common.KEY: KeySprite,
            common.INDICATOR: ascii_art.Partial(objects.IndicatorObjectSprite,
                                                char_to_track=common.KEY,
                                                override_position=(0, 5)),
            common.TIMER: ascii_art.Partial(common.TimerSprite,
                                            self._max_frames['explore']),
        },
        update_schedule=[
            common.PLAYER, common.KEY, common.INDICATOR, common.TIMER],
        z_order=[common.KEY, common.INDICATOR, common.PLAYER, common.TIMER],
    )

  def _make_distractor_phase(self):
    return common.distractor_phase(
        player_sprite=PlayerSprite,
        num_apples=self._num_apples,
        max_frames=self._max_frames['distractor'],
        apple_reward=self._apple_reward,
        fix_apple_reward_in_episode=self._fix_apple_reward_in_episode)

  def _make_reward_phase(self):
    return ascii_art.ascii_art_to_game(
        art=REWARD_GRID,
        what_lies_beneath=' ',
        sprites={
            common.PLAYER: PlayerSprite,
            common.DOOR: ascii_art.Partial(DoorSprite,
                                           pickup_reward=self._final_reward),
            common.TIMER: ascii_art.Partial(common.TimerSprite,
                                            self._max_frames['reward'],
                                            track_chapter_reward=True),
        },
        update_schedule=[common.PLAYER, common.DOOR, common.TIMER],
        z_order=[common.PLAYER, common.DOOR, common.TIMER],
    )

  def make_episode(self):
    """Factory method for generating new episodes of the game."""
    if self._crop:
      croppers = common.get_cropper()
    else:
      croppers = None

    return storytelling.Story([
        self._make_explore_phase,
        self._make_distractor_phase,
        self._make_reward_phase,
    ], croppers=croppers)
