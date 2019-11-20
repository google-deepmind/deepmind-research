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
"""Common utilities for Pycolab games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import colorsys
import numpy as np
from pycolab import ascii_art
from pycolab import cropping
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites
from six.moves import zip
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest

# Actions.
# Those with a negative ID are not allowed for the agent.
ACTION_QUIT = -2
ACTION_DELAY = -1
ACTION_NORTH = 0
ACTION_SOUTH = 1
ACTION_WEST = 2
ACTION_EAST = 3

NUM_ACTIONS = 4
DEFAULT_MAX_FRAMES_PER_PHASE = {
    'explore': 15,
    'distractor': 90,
    'reward': 15
}

# Reserved symbols.
PLAYER = '+'
BORDER = '#'
BACKGROUND = ' '
KEY = 'k'
DOOR = 'd'
APPLE = 'a'
TIMER = 't'
INDICATOR = 'i'

FIXED_COLOURS = {
    PLAYER: (898, 584, 430),
    BORDER: (100, 100, 100),
    BACKGROUND: (800, 800, 800),
    KEY: (627, 321, 176),
    DOOR: (529, 808, 922),
    APPLE: (550, 700, 0),
}

APPLE_DISTRACTOR_GRID = [
    '###########',
    '#a a a a a#',
    '# a a a a #',
    '#a a a a a#',
    '# a a a a #',
    '#a a + a a#',
    '###########'
]
DEFAULT_APPLE_RESPAWN_TIME = 20
DEFAULT_APPLE_REWARD = 1.


def get_shuffled_symbol_colour_map(rng_or_seed, symbols,
                                   num_potential_colours=None):
  """Get a randomized mapping between symbols and colours.

  Args:
    rng_or_seed: A random state or random seed.
    symbols: List of symbols.
    num_potential_colours: Number of equally spaced colours to choose from.
      Defaults to number of symbols. Colours are generated deterministically.

  Returns:
    Randomized mapping between symbols and colours.
  """
  num_symbols = len(symbols)
  num_potential_colours = num_potential_colours or num_symbols
  if isinstance(rng_or_seed, np.random.RandomState):
    rng = rng_or_seed
  else:
    rng = np.random.RandomState(rng_or_seed)

  # Generate a range of colours.
  step = 1. / num_potential_colours
  hues = np.arange(0, num_potential_colours) * step
  potential_colours = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues]

  # Randomly draw num_symbols colours without replacement.
  rng.shuffle(potential_colours)
  colours = potential_colours[:num_symbols]

  symbol_to_colour_map = dict(list(zip(symbols, colours)))

  # Multiply each colour value by 1000.
  return nest.map_structure(lambda c: int(c * 1000), symbol_to_colour_map)


def get_cropper():
  return cropping.ScrollingCropper(
      rows=5,
      cols=5,
      to_track=PLAYER,
      pad_char=BACKGROUND,
      scroll_margins=(2, 2))


def distractor_phase(player_sprite, num_apples, max_frames,
                     apple_reward=DEFAULT_APPLE_REWARD,
                     fix_apple_reward_in_episode=False,
                     respawn_every=DEFAULT_APPLE_RESPAWN_TIME):
  """Distractor phase engine factory.

  Args:
    player_sprite: Player sprite class.
    num_apples: Number of apples to sample from the apple distractor grid.
    max_frames: Maximum duration of the distractor phase in frames.
    apple_reward: Can either be a scalar specifying the reward or a reward range
        [min, max), given as a list or tuple, to uniformly sample from.
    fix_apple_reward_in_episode: The apple reward is constant throughout each
        episode.
    respawn_every: respawn frequency of apples.

  Returns:
    Distractor phase engine.
  """
  distractor_grid = keep_n_characters_in_grid(APPLE_DISTRACTOR_GRID, APPLE,
                                              num_apples)

  engine = ascii_art.ascii_art_to_game(
      distractor_grid,
      what_lies_beneath=BACKGROUND,
      sprites={
          PLAYER: player_sprite,
          TIMER: ascii_art.Partial(TimerSprite, max_frames),
      },
      drapes={
          APPLE: ascii_art.Partial(
              AppleDrape,
              reward=apple_reward,
              fix_apple_reward_in_episode=fix_apple_reward_in_episode,
              respawn_every=respawn_every)
      },
      update_schedule=[PLAYER, APPLE, TIMER],
      z_order=[APPLE, PLAYER, TIMER],
  )

  return engine


def replace_grid_symbols(grid, old_to_new_map):
  """Replaces symbols in the grid.

  If mapping is not defined the symbol is not updated.

  Args:
    grid: Represented as a list of strings.
    old_to_new_map: Mapping between symbols.

  Returns:
    Updated grid.
  """
  def symbol_map(x):
    if x in old_to_new_map:
      return old_to_new_map[x]
    return x
  new_grid = []
  for row in grid:
    new_grid.append(''.join(symbol_map(i) for i in row))
  return new_grid


def keep_n_characters_in_grid(grid, character, n, backdrop_char=BACKGROUND):
  """Keeps only a sample of characters `character` in the grid."""
  np_grid = np.array([list(i) for i in grid])
  char_positions = np.argwhere(np_grid == character)

  # Randomly select parts to remove.
  num_empty_positions = char_positions.shape[0] - n
  if num_empty_positions < 0:
    raise ValueError('Not enough characters `{}` in grid.'.format(character))
  empty_pos = np.random.permutation(char_positions)[:num_empty_positions]

  # Remove characters.
  grid = [list(row) for row in grid]
  for (i, j) in empty_pos:
    grid[i][j] = backdrop_char

  return [''.join(row) for row in grid]


class PlayerSprite(prefab_sprites.MazeWalker):
  """Sprite for the actor."""

  def __init__(self, corner, position, character, impassable=BORDER):
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable=impassable,
        confined_to_board=True)

  def update(self, actions, board, layers, backdrop, things, the_plot):

    the_plot.add_reward(0.)

    if actions == ACTION_QUIT:
      the_plot.next_chapter = None
      the_plot.terminate_episode()

    if actions == ACTION_WEST:
      self._west(board, the_plot)
    elif actions == ACTION_EAST:
      self._east(board, the_plot)
    elif actions == ACTION_NORTH:
      self._north(board, the_plot)
    elif actions == ACTION_SOUTH:
      self._south(board, the_plot)


class AppleDrape(plab_things.Drape):
  """Drape for the apples used in the distractor phase."""

  def __init__(self,
               curtain,
               character,
               respawn_every,
               reward,
               fix_apple_reward_in_episode):
    """Constructor.

    Args:
      curtain: Array specifying locations of apples. Obtained from ascii grid.
      character: Character representing the drape.
      respawn_every: respawn frequency of apples.
      reward: Can either be a scalar specifying the reward or a reward range
        [min, max), given as a list or tuple, to uniformly sample from.
      fix_apple_reward_in_episode: If set to True, then only sample the apple's
        reward once in the episode and then fix the value.
    """
    super(AppleDrape, self).__init__(curtain, character)
    self._respawn_every = respawn_every
    if not isinstance(reward, (list, tuple)):
      # Assuming scalar.
      self._reward = [reward, reward]
    else:
      if len(reward) != 2:
        raise ValueError('Reward must be a scalar or a two element list/tuple.')
      self._reward = reward
    self._fix_apple_reward_in_episode = fix_apple_reward_in_episode

    # Grid specifying for each apple the last frame it was picked up.
    # Initialized to inifinity for cells with apples and -1 for cells without.
    self._last_pickup = np.where(curtain,
                                 np.inf * np.ones_like(curtain),
                                 -1. * np.ones_like(curtain))

  def update(self, actions, board, layers, backdrop, things, the_plot):
    player_position = things[PLAYER].position
    # decide the apple_reward
    if (self._fix_apple_reward_in_episode and
        not the_plot.get('sampled_apple_reward', None)):
      the_plot['sampled_apple_reward'] = np.random.choice((self._reward[0],
                                                           self._reward[1]))

    if self.curtain[player_position]:
      self._last_pickup[player_position] = the_plot.frame
      self.curtain[player_position] = False
      if not self._fix_apple_reward_in_episode:
        the_plot.add_reward(np.random.uniform(*self._reward))
      else:
        the_plot.add_reward(the_plot['sampled_apple_reward'])

    if self._respawn_every:
      respawn_cond = the_plot.frame > self._last_pickup + self._respawn_every
      respawn_cond &= self._last_pickup >= 0
      self.curtain[respawn_cond] = True


class TimerSprite(plab_things.Sprite):
  """Sprite for the timer.

  The timer is in charge of stopping the current chapter. Timer sprite should be
  placed last in the update order to make sure everything is updated before the
  chapter terminates.
  """

  def __init__(self, corner, position, character, max_frames,
               track_chapter_reward=False):
    super(TimerSprite, self).__init__(corner, position, character)
    if not isinstance(max_frames, int):
      raise ValueError('max_frames must be of type integer.')
    self._max_frames = max_frames
    self._visible = False
    self._track_chapter_reward = track_chapter_reward
    self._total_chapter_reward = 0.

  def update(self, actions, board, layers, backdrop, things, the_plot):
    directives = the_plot._get_engine_directives()  # pylint: disable=protected-access

    if self._track_chapter_reward:
      self._total_chapter_reward += directives.summed_reward or 0.

    # Every chapter starts at frame = 0.
    if the_plot.frame >= self._max_frames or directives.game_over:
      # Calculate the reward obtained in this phase and send it through the
      # extra observations channel.
      if self._track_chapter_reward:
        the_plot['chapter_reward'] = self._total_chapter_reward
      the_plot.terminate_episode()
