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

"""Box props used in Mujoban that the agent pushes.
"""

import itertools

from dm_control import composer
from dm_control.entities import props


class Box(props.Primitive):
  """A class representing a box prop."""

  def _build(self, half_lengths=None, mass=None, name='box'):
    half_lengths = half_lengths or [0.05, 0.1, 0.15]
    super(Box, self)._build(geom_type='box',
                            size=half_lengths,
                            mass=mass,
                            name=name)


class BoxWithSites(Box):
  """A class representing a box prop with sites on the corners."""

  def _build(self, half_lengths=None, mass=None, name='box'):
    half_lengths = half_lengths or [0.05, 0.1, 0.15]
    super(BoxWithSites, self)._build(half_lengths=half_lengths, mass=mass,
                                     name=name)

    corner_positions = itertools.product([half_lengths[0], -half_lengths[0]],
                                         [half_lengths[1], -half_lengths[1]],
                                         [half_lengths[2], -half_lengths[2]])
    corner_sites = []
    for i, corner_pos in enumerate(corner_positions):
      corner_sites.append(
          self.mjcf_model.worldbody.add(
              'site',
              type='sphere',
              name='corner_{}'.format(i),
              size=[0.1],
              pos=corner_pos,
              rgba=[1, 0, 0, 1.0],
              group=composer.SENSOR_SITES_GROUP))
    self._corner_sites = tuple(corner_sites)

  @property
  def corner_sites(self):
    return self._corner_sites
