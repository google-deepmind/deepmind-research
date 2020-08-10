# Copyright 2020 Deepmind Technologies Limited.
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

"""A rectangular pedestal."""

from dm_control import composer
from dm_control import mjcf


class Pedestal(composer.Entity):
  """A rectangular pedestal."""

  def _build(self, size=(.2, .3, .05), rgba=(0, .5, 0, 1), name='pedestal'):
    self._mjcf_root = mjcf.RootElement(model=name)
    self._geom = self._mjcf_root.worldbody.add(
        'geom', type='box', size=size, name='geom', rgba=rgba)

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @property
  def geom(self):
    return self._geom

  def after_compile(self, physics, unused_random_state):
    super(Pedestal, self).after_compile(physics, unused_random_state)
    self._body_geom_ids = set(
        physics.bind(geom).element_id
        for geom in self.mjcf_model.find_all('geom'))

  @property
  def body_geom_ids(self):
    return self._body_geom_ids


class Bucket(composer.Entity):
  """A rectangular bucket."""

  def _build(self, size=(.2, .3, .05), rgba=(0, .5, 0, 1), name='pedestal'):
    self._mjcf_root = mjcf.RootElement(model=name)
    self._geoms = []
    self._geoms.append(self._mjcf_root.worldbody.add(
        'geom', type='box', size=size, name='geom_bottom', rgba=rgba))
    self._geoms.append(self._mjcf_root.worldbody.add(
        'geom', type='box', size=(size[2], size[1], size[0]), name='geom_s1',
        rgba=rgba, pos=[size[0], 0, size[0]]))
    self._geoms.append(self._mjcf_root.worldbody.add(
        'geom', type='box', size=(size[2], size[1], size[0]), name='geom_s2',
        rgba=rgba, pos=[-size[0], 0, size[0]]))
    self._geoms.append(self._mjcf_root.worldbody.add(
        'geom', type='box', size=(size[0], size[2], size[0]), name='geom_s3',
        rgba=rgba, pos=[0, size[1], size[0]]))
    self._geoms.append(self._mjcf_root.worldbody.add(
        'geom', type='box', size=(size[0], size[2], size[0]), name='geom_s4',
        rgba=rgba, pos=[0, -size[1], size[0]]))

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @property
  def geom(self):
    return self._geoms

  def after_compile(self, physics, unused_random_state):
    super(Bucket, self).after_compile(physics, unused_random_state)
    self._body_geom_ids = set(
        physics.bind(geom).element_id
        for geom in self.mjcf_model.find_all('geom'))

  @property
  def body_geom_ids(self):
    return self._body_geom_ids

