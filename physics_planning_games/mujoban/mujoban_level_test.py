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

"""Tests for mujoban_level."""


from absl.testing import absltest

from physics_planning_games.mujoban import mujoban_level


_LEVEL = """
#####
#  @####
#  $.  #
###$.# #
#  $.# #
# #$.  #
#    ###
######"""

_GRID_LEVEL = """********
*..P****
*..BG..*
***BG*.*
*..BG*.*
*.*BG..*
*....***
********
"""


class MujobanLevelTest(absltest.TestCase):

  def test_ascii_to_text_grid_level(self):
    grid_level = mujoban_level._ascii_to_text_grid_level(_LEVEL)
    self.assertEqual(_GRID_LEVEL, grid_level)


if __name__ == '__main__':
  absltest.main()
