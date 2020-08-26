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

"""Entities representing board game pieces."""


import itertools

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
import numpy as np

_VISIBLE_SITE_GROUP = 0
_INVISIBLE_SITE_GROUP = 3
_RED = (1., 0., 0., 0.5)
_BLUE = (0., 0, 1., 0.5)

_INVALID_PLAYER_ID = '`player_id` must be between 0 and {}, got {}.'
_NO_MORE_MARKERS_AVAILABLE = (
    'All {} markers for player {} have already been placed.')


class Markers(composer.Entity):
  """A collection of non-physical entities for marking board positions."""

  def _build(self,
             num_per_player,
             player_colors=(_RED, _BLUE),
             halfwidth=0.025,
             height=0.01,
             board_size=7):
    """Builds a `Markers` entity.

    Args:
      num_per_player: Integer, the total number of markers to create per player.
      player_colors: Sequence of (R, G, B, A) values specifying the marker
        colors for each player.
      halfwidth: Scalar, the halfwidth of each marker.
      height: Scalar, height of each marker.
      board_size: Integer, optional if using the integer indexing.
    """
    root = mjcf.RootElement(model='markers')
    root.default.site.set_attributes(type='cylinder', size=(halfwidth, height))
    all_markers = []
    for i, color in enumerate(player_colors):
      player_name = 'player_{}'.format(i)
      # TODO(alimuldal): Would look cool if these were textured.
      material = root.asset.add('material', name=player_name, rgba=color)
      player_markers = []
      for j in range(num_per_player):
        player_markers.append(
            root.worldbody.add(
                'site',
                name='player_{}_move_{}'.format(i, j),
                material=material))
      all_markers.append(player_markers)
    self._num_players = len(player_colors)
    self._mjcf_model = root
    self._all_markers = all_markers
    self._move_counts = [0] * self._num_players
    # To go from integer position to marker index in the all_markers array
    self._marker_ids = np.zeros((2, board_size, board_size))
    self._board_size = board_size

  def _build_observables(self):
    return MarkersObservables(self)

  @property
  def mjcf_model(self):
    """`mjcf.RootElement` for this entity."""
    return self._mjcf_model

  @property
  def markers(self):
    """Marker sites belonging to all players.

    Returns:
      A nested list, where `markers[i][j]` contains the `mjcf.Element`
      corresponding to player i's jth marker.
    """
    return self._all_markers

  def initialize_episode(self, physics, random_state):
    """Resets the markers at the start of an episode."""
    del random_state  # Unused.
    self._reset(physics)

  def _reset(self, physics):
    for player_markers in self._all_markers:
      for marker in player_markers:
        bound_marker = physics.bind(marker)
        bound_marker.pos = 0.  # Markers are initially placed at the origin.
        bound_marker.group = _INVISIBLE_SITE_GROUP
    self._move_counts = [0] * self._num_players
    self._marker_ids = np.zeros((2, self._board_size, self._board_size),
                                dtype=np.int32)

  def make_all_invisible(self, physics):
    for player_markers in self._all_markers:
      for marker in player_markers:
        bound_marker = physics.bind(marker)
        bound_marker.group = _INVISIBLE_SITE_GROUP

  def make_visible_by_bpos(self, physics, player_id, all_bpos):
    for bpos in all_bpos:
      marker_id = self._marker_ids[player_id][bpos[0]][bpos[1]]
      marker = self._all_markers[player_id][marker_id]
      bound_marker = physics.bind(marker)
      bound_marker.group = _VISIBLE_SITE_GROUP

  def mark(self, physics, player_id, pos, bpos=None):
    """Enables the visibility of a marker, moves it to the specified position.

    Args:
      physics: `mjcf.Physics` instance.
      player_id: Integer specifying the ID of the player whose marker to use.
      pos: Array-like object specifying the cartesian position of the marker.
      bpos: Board position, optional integer coordinates to index the markers.

    Raises:
      ValueError: If `player_id` is invalid.
      RuntimeError: If `player_id` has no more available markers.
    """
    if not 0 <= player_id < self._num_players:
      raise ValueError(
          _INVALID_PLAYER_ID.format(self._num_players - 1, player_id))
    markers = self._all_markers[player_id]
    move_count = self._move_counts[player_id]
    if move_count >= len(markers):
      raise RuntimeError(
          _NO_MORE_MARKERS_AVAILABLE.format(move_count, player_id))
    bound_marker = physics.bind(markers[move_count])
    bound_marker.pos = pos
    # TODO(alimuldal): Set orientation as well (random? same as contact frame?)
    bound_marker.group = _VISIBLE_SITE_GROUP
    self._move_counts[player_id] += 1

    if bpos:
      self._marker_ids[player_id][bpos[0]][bpos[1]] = move_count


class MarkersObservables(composer.Observables):
  """Observables for a `Markers` entity."""

  @composer.observable
  def position(self):
    """Cartesian positions of all marker sites.

    Returns:
      An `observable.MJCFFeature` instance. When called with an instance of
      `physics` as the argument, this will return a numpy float64 array of shape
      (num_players * num_markers, 3) where each row contains the cartesian
      position of a marker. Unplaced markers will have position (0, 0, 0).
    """
    return observable.MJCFFeature(
        'xpos', list(itertools.chain.from_iterable(self._entity.markers)))
