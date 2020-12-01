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

"""Module for generating Mujoban level.

"""


import labmaze


BOX_CHAR = 'B'
TARGET_CHAR = labmaze.defaults.OBJECT_TOKEN

_DEFAULT_LEVEL = """
   #####
   #   #
#### # #
# . .# #
#  .   #
# .## $##
##  #$$ #
 ##   $@#
  ##  ###
   ####"""


# The meaning of symbols here are the same as defined in
# http://sneezingtiger.com/sokoban/levels/sasquatch5Text.html. These are the
# same symbols as used by the Sokoban community.
EMPTY_CELL = ' '
GOAL = '.'
PLAYER = '@'
PLAYER_ON_GOAL = '+'
BOX = '$'
BOX_ON_GOAL = '*'
WALL = '#'
_SOKOBAN_SYMBOLS = [
    EMPTY_CELL, GOAL, PLAYER, PLAYER_ON_GOAL, BOX, BOX_ON_GOAL, WALL
]


def single_level_generator(level=_DEFAULT_LEVEL):
  while True:
    yield level


def _ascii_to_text_grid_level(ascii_level):
  """Goes from official Sokoban ASCII art to string understood by Mujoban.

  Args:
    ascii_level: a multiline string; each character is a location in a
      gridworld.

  Returns:
    A string.
  """
  level = ascii_level
  if level.startswith('\n'):
    level = level[1:]
  level = level.replace('$', BOX_CHAR)
  level = level.replace('.', TARGET_CHAR)
  level = level.replace(' ', '.')
  level = level.replace('#', '*')
  level = level.replace('@', 'P')
  if level[-1] == '\n':
    level = level[:-1]
  # Pad
  all_rows = level.split('\n')
  width = max(len(row) for row in all_rows)
  padded_rows = []
  for row in all_rows:
    row += '*' * (width - len(row))
    padded_rows.append(row)
  level = '\n'.join(padded_rows)
  return level + '\n'


class MujobanLevel(labmaze.BaseMaze):
  """A maze that represents a level in Mujoban."""

  def __init__(self, ascii_level_generator=single_level_generator):
    """Constructor.

    Args:
      ascii_level_generator: a Python generator. At each iteration, this should
      return a string representing a level. The symbols in the string should be
      those of http://sneezingtiger.com/sokoban/levels/sasquatch5Text.html.
      These are the same symbols as used by the Sokoban community.
    """
    self._level_iterator = ascii_level_generator()
    self.regenerate()

  def regenerate(self):
    """Regenerates the maze if required."""
    level = next(self._level_iterator)
    self._entity_layer = labmaze.TextGrid(_ascii_to_text_grid_level(level))
    self._variation_layer = self._entity_layer.copy()
    self._variation_layer[:] = '.'
    self._num_boxes = (self._entity_layer == BOX_CHAR).sum()
    num_targets = (self._entity_layer == TARGET_CHAR).sum()
    if num_targets != self._num_boxes:
      raise ValueError('Number of targets {} should equal number of boxes {}.'
                       .format(num_targets, self._num_boxes))

  @property
  def num_boxes(self):
    return self._num_boxes

  @property
  def num_targets(self):
    return self._num_boxes

  @property
  def entity_layer(self):
    return self._entity_layer

  @property
  def variations_layer(self):
    return self._variation_layer

  @property
  def height(self):
    return self._entity_layer.shape[0]

  @property
  def width(self):
    return self._entity_layer.shape[1]
