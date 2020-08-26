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

"""Tests for physics_planning_games.board_games._internal.pieces.py."""

from absl.testing import absltest
from dm_control import mjcf
import numpy as np

from physics_planning_games.board_games._internal import pieces


class MarkersTest(absltest.TestCase):

  def test_position_observable(self):
    num_per_player = 3
    markers = pieces.Markers(num_per_player=num_per_player)
    physics = mjcf.Physics.from_mjcf_model(markers.mjcf_model)
    all_positions = [
        [(0, 1, 2), (3, 4, 5), (6, 7, 8)],  # Player 0
        [(-1, 2, -3), (4, -5, 6)],  # Player 1
    ]
    for player_id, positions in enumerate(all_positions):
      for marker_pos in positions:
        markers.mark(physics=physics, player_id=player_id, pos=marker_pos)
    expected_positions = np.zeros((2, num_per_player, 3), dtype=np.double)
    expected_positions[0, :len(all_positions[0])] = all_positions[0]
    expected_positions[1, :len(all_positions[1])] = all_positions[1]
    observed_positions = markers.observables.position(physics)
    np.testing.assert_array_equal(
        expected_positions.reshape(-1, 3), observed_positions)

  def test_invalid_player_id(self):
    markers = pieces.Markers(num_per_player=5)
    physics = mjcf.Physics.from_mjcf_model(markers.mjcf_model)
    invalid_player_id = 99
    with self.assertRaisesWithLiteralMatch(
        ValueError, pieces._INVALID_PLAYER_ID.format(1, 99)):
      markers.mark(physics=physics, player_id=invalid_player_id, pos=(1, 2, 3))

  def test_too_many_moves(self):
    num_per_player = 5
    player_id = 0
    markers = pieces.Markers(num_per_player=num_per_player)
    physics = mjcf.Physics.from_mjcf_model(markers.mjcf_model)
    for _ in range(num_per_player):
      markers.mark(physics=physics, player_id=player_id, pos=(1, 2, 3))
    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        pieces._NO_MORE_MARKERS_AVAILABLE.format(num_per_player, player_id)):
      markers.mark(physics=physics, player_id=player_id, pos=(1, 2, 3))


if __name__ == '__main__':
  absltest.main()
