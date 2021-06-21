# Copyright 2018 Deepmind Technologies Limited.
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

"""DeepMind Lab textures."""

from dm_control import composer
from dm_control import mjcf
from labmaze import assets as labmaze_assets


class SkyBox(composer.Entity):
  """Represents a texture asset for the sky box."""

  def _build(self, style):
    labmaze_textures = labmaze_assets.get_sky_texture_paths(style)
    self._mjcf_root = mjcf.RootElement(model='dmlab_' + style)
    self._texture = self._mjcf_root.asset.add(
        'texture', type='skybox', name='texture',
        fileleft=labmaze_textures.left, fileright=labmaze_textures.right,
        fileup=labmaze_textures.up, filedown=labmaze_textures.down,
        filefront=labmaze_textures.front, fileback=labmaze_textures.back)

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @property
  def texture(self):
    return self._texture


class WallTextures(composer.Entity):
  """Represents wall texture assets."""

  def _build(self, style):
    labmaze_textures = labmaze_assets.get_wall_texture_paths(style)
    self._mjcf_root = mjcf.RootElement(model='dmlab_' + style)
    self._textures = []
    for texture_name, texture_path in labmaze_textures.items():
      self._textures.append(self._mjcf_root.asset.add(
          'texture', type='2d', name=texture_name,
          file=texture_path.format(texture_name)))

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @property
  def textures(self):
    return self._textures


class FloorTextures(composer.Entity):
  """Represents floor texture assets."""

  def _build(self, style):
    labmaze_textures = labmaze_assets.get_floor_texture_paths(style)
    self._mjcf_root = mjcf.RootElement(model='dmlab_' + style)
    self._textures = []
    for texture_name, texture_path in labmaze_textures.items():
      self._textures.append(self._mjcf_root.asset.add(
          'texture', type='2d', name=texture_name,
          file=texture_path.format(texture_name)))

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @property
  def textures(self):
    return self._textures
