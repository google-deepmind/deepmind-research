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
"""Pycolab human player."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

from absl import app
from absl import flags
import numpy as np
from pycolab import human_ui

from tvt.pycolab import active_visual_match
from tvt.pycolab import common
from tvt.pycolab import key_to_door

FLAGS = flags.FLAGS

flags.DEFINE_enum('game', 'key_to_door',
                  ['key_to_door', 'active_visual_match'],
                  'The name of the game')


def main(unused_argv):

  rng = np.random.RandomState()

  if FLAGS.game == 'key_to_door':
    game = key_to_door.Game(rng)
  elif FLAGS.game == 'active_visual_match':
    game = active_visual_match.Game(rng)
  else:
    raise ValueError('Unsupported game "%s".' % FLAGS.game)
  episode = game.make_episode()

  ui = human_ui.CursesUi(
      keys_to_actions={
          curses.KEY_UP: common.ACTION_NORTH,
          curses.KEY_DOWN: common.ACTION_SOUTH,
          curses.KEY_LEFT: common.ACTION_WEST,
          curses.KEY_RIGHT: common.ACTION_EAST,
          -1: common.ACTION_DELAY,
          'q': common.ACTION_QUIT,
          'Q': common.ACTION_QUIT},
      delay=-1,
      colour_fg=game.colours
      )
  ui.play(episode)

if __name__ == '__main__':
  app.run(main)
