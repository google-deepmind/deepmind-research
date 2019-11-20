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
"""Pycolab sprites."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites
import six
from tvt.pycolab import common


class PlayerSprite(prefab_sprites.MazeWalker):
  """Sprite representing the agent."""

  def __init__(self, corner, position, character,
               max_steps_per_act, moving_player):

    """Indicates to the superclass that we can't walk off the board."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable=[common.BORDER],
        confined_to_board=True)

    self._moving_player = moving_player
    self._max_steps_per_act = max_steps_per_act
    self._num_steps = 0

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop  # Unused.

    if actions is not None:
      assert actions in common.ACTIONS

    the_plot.log("Step {} | Action {}".format(self._num_steps, actions))
    the_plot.add_reward(0.0)
    self._num_steps += 1

    if actions == common.ACTION_QUIT:
      the_plot.terminate_episode()

    if  self._moving_player:
      if actions == common.ACTION_WEST:
        self._west(board, the_plot)
      elif actions == common.ACTION_EAST:
        self._east(board, the_plot)
      elif actions == common.ACTION_NORTH:
        self._north(board, the_plot)
      elif actions == common.ACTION_SOUTH:
        self._south(board, the_plot)

    if self._max_steps_per_act == self._num_steps:
      the_plot.terminate_episode()


class ObjectSprite(plab_things.Sprite):
  """Sprite for a generic object which can be collectable."""

  def __init__(self, corner, position, character, reward=0., collectable=True,
               terminate=True):
    super(ObjectSprite, self).__init__(corner, position, character)
    self._reward = reward  # Reward on pickup.
    self._collectable = collectable

  def set_visibility(self, visible):
    self._visible = visible

  def update(self, actions, board, layers, backdrop, things, the_plot):
    player_position = things[common.PLAYER].position
    pick_up = self.position == player_position

    if pick_up and self.visible:
      the_plot.add_reward(self._reward)
      if self._collectable:
        self.set_visibility(False)
        # set all other objects to be invisible
        for v in six.itervalues(things):
          if isinstance(v, ObjectSprite):
            v.set_visibility(False)


class IndicatorObjectSprite(plab_things.Sprite):
  """Sprite for the indicator object.

  The indicator object is an object that spawns at a designated position once
  the player picks up an object defined by the `char_to_track` argument.
  The indicator object is spawned for just a single frame.
  """

  def __init__(self, corner, position, character, char_to_track,
               override_position=None):
    super(IndicatorObjectSprite, self).__init__(corner, position, character)
    if override_position is not None:
      self._position = override_position
    self._char_to_track = char_to_track
    self._visible = False
    self._pickup_frame = None

  def update(self, actions, board, layers, backdrop, things, the_plot):
    player_position = things[common.PLAYER].position
    pick_up = things[self._char_to_track].position == player_position

    if self._pickup_frame:
      self._visible = False

    if pick_up and not self._pickup_frame:
      self._visible = True
      self._pickup_frame = the_plot.frame
