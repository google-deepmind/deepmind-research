# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The pycolab core of the environment for going to the ballet.

This builds the text-based (non-graphical) engine of the environment, and offers
a UI which a human can play (for a fixed level). However, the logic of level
creation, the graphics, and anything that is external to the pycolab engine
itself is contained in ballet_environment.py.
"""
import curses
import enum

from absl import app
from absl import flags

from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites

FLAGS = flags.FLAGS

ROOM_SIZE = (11, 11)  # one square around edge will be wall.
DANCER_POSITIONS = [(2, 2), (2, 5), (2, 8),
                    (5, 2), (5, 8),  # space in center for agent
                    (8, 2), (8, 5), (8, 8)]
AGENT_START = (5, 5)
AGENT_CHAR = "A"
WALL_CHAR = "#"
FLOOR_CHAR = " "
RESERVED_CHARS = [AGENT_CHAR, WALL_CHAR, FLOOR_CHAR]
POSSIBLE_DANCER_CHARS = [
    chr(i) for i in range(65, 91) if chr(i) not in RESERVED_CHARS
]

DANCE_SEQUENCE_LENGTHS = 16


# movement directions for dancers / actions for agent
class DIRECTIONS(enum.IntEnum):
  N = 0
  NE = 1
  E = 2
  SE = 3
  S = 4
  SW = 5
  W = 6
  NW = 7

DANCE_SEQUENCES = {
    "circle_cw": [
        DIRECTIONS.N, DIRECTIONS.E, DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.W,
        DIRECTIONS.W, DIRECTIONS.N, DIRECTIONS.N, DIRECTIONS.E, DIRECTIONS.E,
        DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.W, DIRECTIONS.W, DIRECTIONS.N,
        DIRECTIONS.E
    ],
    "circle_ccw": [
        DIRECTIONS.N, DIRECTIONS.W, DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.E,
        DIRECTIONS.E, DIRECTIONS.N, DIRECTIONS.N, DIRECTIONS.W, DIRECTIONS.W,
        DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.E, DIRECTIONS.E, DIRECTIONS.N,
        DIRECTIONS.W
    ],
    "up_and_down": [
        DIRECTIONS.N, DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.N, DIRECTIONS.N,
        DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.N, DIRECTIONS.N, DIRECTIONS.S,
        DIRECTIONS.S, DIRECTIONS.N, DIRECTIONS.N, DIRECTIONS.S, DIRECTIONS.S,
        DIRECTIONS.N
    ],
    "left_and_right": [
        DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.E,
        DIRECTIONS.W, DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.E, DIRECTIONS.W,
        DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.W,
        DIRECTIONS.E
    ],
    "diagonal_uldr": [
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SE, DIRECTIONS.NW,
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SE, DIRECTIONS.NW,
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SE, DIRECTIONS.NW,
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SE, DIRECTIONS.NW
    ],
    "diagonal_urdl": [
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SW, DIRECTIONS.NE
    ],
    "plus_cw": [
        DIRECTIONS.N, DIRECTIONS.S, DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.S,
        DIRECTIONS.N, DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.N, DIRECTIONS.S,
        DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.S, DIRECTIONS.N, DIRECTIONS.W,
        DIRECTIONS.E
    ],
    "plus_ccw": [
        DIRECTIONS.N, DIRECTIONS.S, DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.S,
        DIRECTIONS.N, DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.N, DIRECTIONS.S,
        DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.S, DIRECTIONS.N, DIRECTIONS.E,
        DIRECTIONS.W
    ],
    "times_cw": [
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SE, DIRECTIONS.NW,
        DIRECTIONS.SW, DIRECTIONS.NE, DIRECTIONS.NW, DIRECTIONS.SE,
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SE, DIRECTIONS.NW,
        DIRECTIONS.SW, DIRECTIONS.NE, DIRECTIONS.NW, DIRECTIONS.SE
    ],
    "times_ccw": [
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.SE, DIRECTIONS.NW, DIRECTIONS.NE, DIRECTIONS.SW,
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.SE, DIRECTIONS.NW, DIRECTIONS.NE, DIRECTIONS.SW
    ],
    "zee": [
        DIRECTIONS.NE, DIRECTIONS.W, DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.E,
        DIRECTIONS.SW, DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SW,
        DIRECTIONS.E, DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.W, DIRECTIONS.NE,
        DIRECTIONS.SW, DIRECTIONS.NE
    ],
    "chevron_down": [
        DIRECTIONS.NW, DIRECTIONS.S, DIRECTIONS.SE, DIRECTIONS.NE, DIRECTIONS.N,
        DIRECTIONS.SW, DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.S, DIRECTIONS.SW, DIRECTIONS.NW, DIRECTIONS.N, DIRECTIONS.SE,
        DIRECTIONS.NW, DIRECTIONS.SE
    ],
    "chevron_up": [
        DIRECTIONS.SE, DIRECTIONS.N, DIRECTIONS.NW, DIRECTIONS.SW, DIRECTIONS.S,
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.NE, DIRECTIONS.SW,
        DIRECTIONS.N, DIRECTIONS.NE, DIRECTIONS.SE, DIRECTIONS.S, DIRECTIONS.NW,
        DIRECTIONS.SE, DIRECTIONS.NW
    ],
}


class DancerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for dancers."""

  def __init__(self, corner, position, character, motion, color, shape,
               value=0.):
    super(DancerSprite, self).__init__(
        corner, position, character, impassable="#")
    self.motion = motion
    self.dance_sequence = DANCE_SEQUENCES[motion].copy()
    self.color = color
    self.shape = shape
    self.value = value
    self.is_dancing = False

  def update(self, actions, board, layers, backdrop, things, the_plot):
    if the_plot["task_phase"] == "dance" and self.is_dancing:
      if not self.dance_sequence:
        raise ValueError(
            "Dance sequence is empty! Was this dancer repeated in the order?")
      dance_move = self.dance_sequence.pop(0)
      if dance_move == DIRECTIONS.N:
        self._north(board, the_plot)
      elif dance_move == DIRECTIONS.NE:
        self._northeast(board, the_plot)
      elif dance_move == DIRECTIONS.E:
        self._east(board, the_plot)
      elif dance_move == DIRECTIONS.SE:
        self._southeast(board, the_plot)
      elif dance_move == DIRECTIONS.S:
        self._south(board, the_plot)
      elif dance_move == DIRECTIONS.SW:
        self._southwest(board, the_plot)
      elif dance_move == DIRECTIONS.W:
        self._west(board, the_plot)
      elif dance_move == DIRECTIONS.NW:
        self._northwest(board, the_plot)

      if not self.dance_sequence:  # done!
        self.is_dancing = False
        the_plot["time_until_next_dance"] = the_plot["dance_delay"]
    else:
      if self.position == things[AGENT_CHAR].position:
        # Award the player the appropriate amount of reward, and end episode.
        the_plot.add_reward(self.value)
        the_plot.terminate_episode()


class PlayerSprite(prefab_sprites.MazeWalker):
  """The player / agent character.

  MazeWalker class methods handle basic movement and collision detection.
  """

  def __init__(self, corner, position, character):
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable="#")

  def update(self, actions, board, layers, backdrop, things, the_plot):
    if the_plot["task_phase"] == "dance":
      # agent's actions are ignored, this logic updates the dance phases.
      if the_plot["time_until_next_dance"] > 0:
        the_plot["time_until_next_dance"] -= 1
        if the_plot["time_until_next_dance"] == 0:  # next phase time!
          if the_plot["dance_order"]:  # start the next dance!
            next_dancer = the_plot["dance_order"].pop(0)
            things[next_dancer].is_dancing = True
          else:  # choice time!
            the_plot["task_phase"] = "choice"
            the_plot["instruction_string"] = the_plot[
                "choice_instruction_string"]
    elif the_plot["task_phase"] == "choice":
      # agent can now move and make its choice
      if actions == DIRECTIONS.N:
        self._north(board, the_plot)
      elif actions == DIRECTIONS.NE:
        self._northeast(board, the_plot)
      elif actions == DIRECTIONS.E:
        self._east(board, the_plot)
      elif actions == DIRECTIONS.SE:
        self._southeast(board, the_plot)
      elif actions == DIRECTIONS.S:
        self._south(board, the_plot)
      elif actions == DIRECTIONS.SW:
        self._southwest(board, the_plot)
      elif actions == DIRECTIONS.W:
        self._west(board, the_plot)
      elif actions == DIRECTIONS.NW:
        self._northwest(board, the_plot)


def make_game(dancers_and_properties, dance_delay=16):
  """Constructs an ascii map, then uses pycolab to make it a game.

  Args:
    dancers_and_properties: list of (character, (row, column), motion, shape,
      color, value), for placing objects in the world.
    dance_delay: how long to wait between dances.

  Returns:
    this_game: Pycolab engine running the specified game.
  """
  num_rows, num_cols = ROOM_SIZE
  level_layout = []
  # upper wall
  level_layout.append("".join([WALL_CHAR] * num_cols))
  # room
  middle_string = "".join([WALL_CHAR] + [" "] * (num_cols - 2) + [WALL_CHAR])
  level_layout.extend([middle_string for _ in range(num_rows - 2)])
  # lower wall
  level_layout.append("".join([WALL_CHAR] * num_cols))

  def _add_to_map(obj, loc):
    """Adds an ascii character to the level at the requested position."""
    obj_row = level_layout[loc[0]]
    pre_string = obj_row[:loc[1]]
    post_string = obj_row[loc[1] + 1:]
    level_layout[loc[0]] = pre_string + obj + post_string

  _add_to_map(AGENT_CHAR, AGENT_START)
  sprites = {AGENT_CHAR: PlayerSprite}
  dance_order = []
  char_to_color_shape = []
  # add dancers to level
  for obj, loc, motion, shape, color, value in dancers_and_properties:
    _add_to_map(obj, loc)
    sprites[obj] = ascii_art.Partial(
        DancerSprite, motion=motion, color=color, shape=shape, value=value)
    char_to_color_shape.append((obj, color + " " + shape))
    dance_order += obj
    if value > 0.:
      choice_instruction_string = motion

  this_game = ascii_art.ascii_art_to_game(
      art=level_layout,
      what_lies_beneath=" ",
      sprites=sprites,
      update_schedule=[[AGENT_CHAR],
                       dance_order])

  this_game.the_plot["task_phase"] = "dance"
  this_game.the_plot["instruction_string"] = "watch"
  this_game.the_plot["choice_instruction_string"] = choice_instruction_string
  this_game.the_plot["dance_order"] = dance_order
  this_game.the_plot["dance_delay"] = dance_delay
  this_game.the_plot["time_until_next_dance"] = 1
  this_game.the_plot["char_to_color_shape"] = char_to_color_shape
  return this_game


def main(argv):
  del argv  # unused
  these_dancers_and_properties = [
      (POSSIBLE_DANCER_CHARS[1], (2, 2), "chevron_up", "triangle", "red", 1),
      (POSSIBLE_DANCER_CHARS[2], (2, 5), "circle_ccw", "triangle", "red", 0),
      (POSSIBLE_DANCER_CHARS[3], (2, 8), "plus_cw", "triangle", "red", 0),
      (POSSIBLE_DANCER_CHARS[4], (5, 2), "plus_ccw", "triangle", "red", 0),
      (POSSIBLE_DANCER_CHARS[5], (5, 8), "times_cw", "triangle", "red", 0),
      (POSSIBLE_DANCER_CHARS[6], (8, 2), "up_and_down", "plus", "blue", 0),
      (POSSIBLE_DANCER_CHARS[7], (8, 5), "left_and_right", "plus", "blue", 0),
      (POSSIBLE_DANCER_CHARS[8], (8, 8), "zee", "plus", "blue", 0),
  ]

  game = make_game(dancers_and_properties=these_dancers_and_properties)

  # Note that these colors are only for human UI
  fg_colours = {
      AGENT_CHAR: (999, 999, 999),  # Agent is white
      WALL_CHAR: (300, 300, 300),  # Wall, dark grey
      FLOOR_CHAR: (0, 0, 0),  # Floor
  }
  for (c, _, _, _, col, _) in these_dancers_and_properties:
    fg_colours[c] = (999, 0, 0) if col == "red" else (0, 0, 999)

  bg_colours = {
      c: (0, 0, 0) for c in RESERVED_CHARS + POSSIBLE_DANCER_CHARS[1:8]
  }

  ui = human_ui.CursesUi(
      keys_to_actions={
          # Basic movement.
          curses.KEY_UP: DIRECTIONS.N,
          curses.KEY_DOWN: DIRECTIONS.S,
          curses.KEY_LEFT: DIRECTIONS.W,
          curses.KEY_RIGHT: DIRECTIONS.E,
          -1: 8,  # Do nothing.
      },
      delay=500,
      colour_fg=fg_colours,
      colour_bg=bg_colours)

  ui.play(game)


if __name__ == "__main__":
  app.run(main)
