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

"""Logic for the Go Game."""

import abc
import collections
import enum
import shutil
import subprocess

from absl import logging
import numpy as np

from dm_control.utils import io as resources
from physics_planning_games.board_games import logic_base
import pyspiel

GNUGO_PATH = '/usr/games/gnugo'
GoMarkerAction = collections.namedtuple('GoMarkerAction',
                                        ['row', 'col', 'pass_action'])

# Note that there is no 'i' in these Go board coordinates
# (cf https://senseis.xmp.net/?Coordinates)
_X_CHARS = 'abcdefghjklmnopqrstuvwxyz'
_X_MAP = {c: x for c, x in zip(_X_CHARS, range(len(_X_CHARS)))}


def _go_marker_to_int(go_marker, board_size):
  """Convert GoMarkerAction into GoPoint integer encoding of move.

  Args:
    go_marker: GoMarkerAction.
    board_size: Board size of the go board (e.g. 9 or 19).

  Returns:
    GoPoint int value.
  """
  if go_marker.pass_action:
    return board_size * board_size
  else:
    return int((go_marker.row) * board_size + go_marker.col)


def _int_to_go_marker(move_int, board_size):
  """Decode the integer move encoding to a GoMarkerAction.

  Args:
    move_int: Integer encoding the go move.
    board_size: Board size of the go board (e.g. 9 or 19).

  Returns:
    GoMarkerAction encoding of move.
  """
  if move_int == board_size * board_size:
    go_marker_action = GoMarkerAction(row=-1, col=-1, pass_action=True)
  else:
    row = move_int // board_size
    col = move_int % board_size
    go_marker_action = GoMarkerAction(row=row, col=col, pass_action=False)

  return go_marker_action


def _go_marker_to_str(go_marker):
  if go_marker.pass_action:
    return 'PASS'
  else:
    move_str = _X_CHARS[go_marker.col] + str(go_marker.row + 1)
    return move_str


def _str_to_go_marker(move_str):
  """Convert from a 2-letter Go move str (e.g.

  a3) to a GoMarker.

  Args:
    move_str: String describing the move (e.g. a3).

  Returns:
    GoMarkerAction encoding of move.
  """
  move_str = move_str.lower()
  if move_str == 'pass':
    action = GoMarkerAction(row=-1, col=-1, pass_action=True)
  elif move_str == 'resign':
    raise NotImplementedError('Not dealing with resign')
  else:
    assert len(move_str) == 2
    col, row = move_str[0], move_str[1]
    col = _X_MAP[col]
    row = int(row) - 1
    action = GoMarkerAction(row=row, col=col, pass_action=False)
  return action


def _get_gnugo_ref_config(level=1, binary_path=None):
  """Reference config for GnuGo.

  Args:
    level: GnuGo level
    binary_path: string pointing to GnuGo binary

  Returns:
    Config dict that can be passed to gtp engine
  """

  try:
    gnugo_binary_path = resources.GetResourceFilename(binary_path)
  except FileNotFoundError:
    gnugo_binary_path = shutil.which('gnugo')
    if not gnugo_binary_path:
      raise FileNotFoundError('Not able to locate gnugo library. ',
                              'Try installing it by:  apt install gnugo')

  gnugo_extra_flags = ['--mode', 'gtp']
  gnugo_extra_flags += ['--chinese-rules', '--capture-all-dead']
  gtp_player_cfg = {
      'name': 'gnugo',
      'binary_path': gnugo_binary_path,
      'level': level,
      'extra_flags': gnugo_extra_flags,
  }
  return gtp_player_cfg


class Stone(enum.Enum):
  EMPTY = 1
  WHITE = 2
  BLACK = 3

  def __lt__(self, other):
    value = int(self.value)
    return value < other.value


def gtp_to_sgf_point(gtp_point, board_size):
  """Format a GTP point according to the SGF format."""
  if gtp_point.lower() == 'pass' or gtp_point.lower() == 'resign':
    return 'tt'
  column, row = gtp_point[0], gtp_point[1:]
  # GTP doesn't use i, but SGF does, so we need to convert.
  gtp_columns = 'abcdefghjklmnopqrstuvwxyz'
  sgf_columns = 'abcdefghijklmnopqrstuvwxyz'
  x = gtp_columns.find(column.lower())
  y = board_size - int(row)
  return '%s%s' % (sgf_columns[x], sgf_columns[y])


class Gtp(object):
  """Wrapper around Go playing program that communicates using GTP."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, checkpoint_file=None):
    self.stones = {
        '.': Stone.EMPTY,
        '+': Stone.EMPTY,
        'O': Stone.WHITE,
        'X': Stone.BLACK
    }
    self.moves = []
    self.comments = []
    self.handicap = 0
    self.board_size = 19
    self.komi = 0
    self.free_handicap = None
    self.byo_yomi_time = None
    self.checkpoint_file = checkpoint_file
    self.stderr = None

  def set_board_size(self, size):
    self.board_size = size
    self.gtp_command('boardsize %d' % size)
    self.gtp_command('clear_board')

  def set_komi(self, komi):
    self.komi = komi
    self.gtp_command('komi %s' % komi)

  def set_free_handicap(self, vertices):
    self.free_handicap = vertices
    self.gtp_command('set_free_handicap %s' % vertices)

  def place_free_handicap(self, n):
    self.free_handicap = self.gtp_command('place_free_handicap %d' % n)
    return self.free_handicap

  def make_move(self, move, record=True):
    self.gtp_command('play %s' % move)
    if record:
      self._record_move(move)

  def set_byo_yomi_time(self, t):
    self.byo_yomi_time = t

  def num_moves(self):
    return len(self.moves)

  def clear_board(self):
    self.moves = []
    self.comments = []
    self.gtp_command('clear_board')

  def generate_move(self, color):
    if self.byo_yomi_time is not None:
      self.gtp_command('time_left %s %d 1' % (color, self.byo_yomi_time))
    move = '%s %s' % (color, self.gtp_command(
        'genmove %s' % color).split(' ')[-1].lower())
    self._record_move(move, stderr=self.stderr)
    return move

  def board(self):
    raw_board = self.gtp_command('showboard', log=False)[1:].strip()
    rows = [line.strip().split('  ')[0] for line in raw_board.split('\n')][1:-1]
    rows = [''.join(row.split(' ')[1:-1]) for row in rows]
    return [[self.stones[cell] for cell in row] for row in rows]

  def quit(self):
    self.gtp_command('quit')

  def final_status(self, status):
    return self.gtp_command('final_status_list %s' % status)[2:].replace(
        '\n', ' ').split(' ')

  def fixed_handicap(self, handicap):
    self.handicap = handicap
    self.gtp_command('fixed_handicap %d' % handicap)

  def undo(self, num_moves):
    self.gtp_command('gg-undo %d' % num_moves)
    for _ in range(num_moves):
      self.moves.pop()
      self.comments.pop()

  def _record_move(self, move, stderr=None):
    self.moves.append(move)
    self.comments.append(stderr)

    if self.checkpoint_file:
      with open(self.checkpoint_file, 'w') as f:
        f.write(self.to_sgf())

  def to_sgf(self):
    sgf = '(;PB[Black]PW[White]KM[%.1f]HA[%d]SZ[19]' % (self.komi,
                                                        self.handicap)
    for i, move in enumerate(self.moves):
      sgf += '\n;' + self._format_sgf_move(move)
      if self.comments[i]:
        sgf += 'C[' + self._sgf_escape(self.comments[i]) + ']'
    return sgf + ')'

  def _format_sgf_move(self, move):
    """Format a move according to the SGF format."""
    color, vertex = str(move).split(' ')
    return '%s[%s]' % (color[0].upper(),
                       gtp_to_sgf_point(vertex, self.board_size))

  def _sgf_escape(self, text):
    return ''.join(['\\' + t if t == ']' or t == '\\' else t for t in text])

  @abc.abstractmethod
  def gtp_command(self, command, log=True):
    """Executes a GTP command and returns its response.

    Args:
      command: The GTP command to run, no trailing newline.
      log: Whether to log command and response to INFO.

    Returns:
      The GTP response.
    Raises:
      GtpError: if the response is not ok (doesn't start with '=').
    """
    pass


class GtpError(Exception):

  def __init__(self, response):
    super(GtpError, self).__init__()
    self.response = response

  def __str__(self):
    return self.response


class GoEngine(Gtp):
  """GTP-based Go engine.

  Supports at least GnuGo and Pachi.

  For GnuGo, at least specify ['--mode', 'gtp'] in extra_flags.
  """

  def __init__(self, command='', checkpoint_file=None, extra_flags=None):
    super(GoEngine, self).__init__(checkpoint_file)
    if extra_flags:
      command = [command] + extra_flags
    self.p = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True)

  def gtp_command(self, command, log=True):
    if log:
      logging.info('GTP: %s', command)
    self.p.stdin.write(command)
    self.p.stdin.write('\n')
    self.p.stdin.flush()

    response = [self.p.stdout.readline()]
    while response[-1] != '\n':
      response.append(self.p.stdout.readline())
    response = ''.join(response).strip()

    if log:
      logging.info('GTP: %s', response)

    if response[0][0] != '=':
      raise GtpError(response)

    return response


class GoGameLogic(logic_base.OpenSpielBasedLogic):
  """Logic for Go game."""

  def __init__(self, board_size, gnugo_level=1, komi=5.5):
    self._board_size = board_size
    self._komi = komi
    gtp_player_cfg = _get_gnugo_ref_config(
        level=gnugo_level,
        binary_path=GNUGO_PATH)

    self._gtp_player = GoEngine(
        command=gtp_player_cfg['binary_path'],
        extra_flags=gtp_player_cfg['extra_flags'])
    self._gtp_player.set_board_size(board_size)
    self.reset()

  def board_size(self):
    return self._board_size

  def get_gtp_player(self):
    return self._gtp_player

  def reset(self):
    """Resets the game state."""
    # For now we always assume we are the starting player and use a random
    # opponent.
    self._gtp_player.gtp_command('clear_board', log=False)
    self._gtp_player.set_board_size(self._board_size)
    self._gtp_player.set_komi(self._komi)
    game = pyspiel.load_game('go', {'board_size': self._board_size})
    self._open_spiel_state = game.new_initial_state()

    self._moves = np.ones(
        (self._board_size * self._board_size * 2,), dtype=np.int32) * -1
    self._move_id = 0

  def show_board(self):
    self._gtp_player.gtp_command('showboard')

  def get_gtp_reward(self):
    self._gtp_player.gtp_command('final_score')

  def get_board_state(self):
    """Returns the logical board state as a numpy array.

    Returns: A boolean array of shape (H, W, C), where H=3, W=3 (height and
      width of the board) and C=4 for the 4 planes. The 4 planes are, in order,
      unmarked, black (player 0), white (player 1) and komi (this layer is
      always all the same value indicating whether white is to play).
    """
    board_state = np.reshape(
        np.array(self._open_spiel_state.observation_tensor(0), dtype=bool),
        [4, self._board_size, self._board_size])
    board_state = np.transpose(board_state, [1, 2, 0])
    board_state = board_state[:, :, [2, 0, 1, 3]]
    return board_state

  def set_state_from_history(self, move_history):
    self.reset()
    move_history = np.squeeze(move_history.numpy())
    for t in range(move_history.size):
      if move_history[t] < 0:
        break
      else:
        self.apply(t % 2, move_history[t])
    # self.show_board()

  def get_move_history(self):
    """Returns the move history as padded numpy array."""
    return self._moves

  def apply(self, player, action):
    """Checks whether action is valid, and if so applies it to the game state.

    Args:
      player: Integer specifying the player ID; either 0 or 1.
      action: A `GoMarkerAction` instance (or numpy.int32) which represent the
        action in the board of size `board_size`.

    Returns:
      True if the action was valid, else False.
    """
    if isinstance(action, GoMarkerAction):
      action = _go_marker_to_int(action, self._board_size)

    if self._open_spiel_state.current_player() != player:
      return False

    legal_actions = self._open_spiel_state.legal_actions()
    if np.isin(action, legal_actions):
      self._open_spiel_state.apply_action(action)
      was_valid_move = True
    else:
      was_valid_move = False

    if not was_valid_move:
      return False

    self._moves[self._move_id] = action
    self._move_id += 1

    # Apply to the Go program
    player_color = 'B' if player == 0 else 'W'
    action_str = _go_marker_to_str(_int_to_go_marker(action, self._board_size))
    self._gtp_player.gtp_command('play {} {}'.format(player_color, action_str))

    return was_valid_move


def gen_move(game_logic, player):
  """Generate move from GTP player and game state defined in game_logic."""
  player_color = 'B' if player == 0 else 'W'
  gtp_player = game_logic.get_gtp_player()
  move_str = gtp_player.gtp_command(
      'reg_genmove {}'.format(player_color), log=True)
  move_str = move_str[2:].lower()
  action = _str_to_go_marker(move_str)
  return action


def gen_random_move(game_logic, random_state):
  """Generate random move for current state in game logic."""
  if game_logic.is_game_over:
    return None
  valid_moves = game_logic.open_spiel_state.legal_actions()
  assert valid_moves
  move = random_state.choice(valid_moves)
  go_action = _int_to_go_marker(move, board_size=game_logic.board_size())
  return go_action


class GoGTPOpponent(logic_base.Opponent):
  """Use external binary Pachi to generate opponent moves."""

  def __init__(self, board_size, mixture_p=0.0):
    """Initialize Go opponent.

    Args:
      board_size: Go board size (int)
      mixture_p: Probability of playing a random move (amongst legal moves).
    """
    self._board_size = board_size
    self._mixture_p = mixture_p

  def reset(self):
    pass

  def policy(self, game_logic, player, random_state):
    """Return policy action.

    Args:
      game_logic: Go game logic state.
      player: Integer specifying the player ID; either 0 or 1.
      random_state: Numpy random state object.

    Returns:
      GoMarkerAction indicating opponent move.
    """
    if random_state.rand() < self._mixture_p:
      return gen_random_move(game_logic, random_state)
    else:
      return gen_move(game_logic, player)


class GoRandomOpponent(logic_base.Opponent):
  """An easy opponent for Go."""

  def __init__(self, board_size):
    self._board_size = board_size

  def reset(self):
    """Resets the opponent's internal state (not implemented)."""
    pass

  def policy(self, game_logic, player, random_state):
    """Return a random, valid move.

    Args:
      game_logic: TicTacToeGameLogic state of the game.
      player: Integer specifying the player ID; either 0 or 1.
      random_state: An instance of `np.random.RandomState`

    Returns:
      GoMarkerAction of opponent.
    """
    return gen_random_move(game_logic, random_state)
