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

"""Board game-specific arena classes."""


from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.mujoco import wrapper

# Robot geoms will be assigned to this group in order to disable their
# visibility to the top-down camera.
ROBOT_GEOM_GROUP = 1


class Standard(composer.Arena):
  """"Board game-specific arena class."""

  def _build(self, name=None):
    """Initializes this arena.

    Args:
      name: (optional) A string, the name of this arena. If `None`, use the
        model name defined in the MJCF file.
    """
    super(Standard, self)._build(name=name)

    # Add visual assets.
    self.mjcf_model.asset.add(
        'texture',
        type='skybox',
        builtin='gradient',
        rgb1=(0.4, 0.6, 0.8),
        rgb2=(0., 0., 0.),
        width=100,
        height=100)
    groundplane_texture = self.mjcf_model.asset.add(
        'texture',
        name='groundplane',
        type='2d',
        builtin='checker',
        rgb1=(0.2, 0.3, 0.4),
        rgb2=(0.1, 0.2, 0.3),
        width=300,
        height=300,
        mark='edge',
        markrgb=(.8, .8, .8))
    groundplane_material = self.mjcf_model.asset.add(
        'material',
        name='groundplane',
        texture=groundplane_texture,
        texrepeat=(5, 5),
        texuniform='true',
        reflectance=0.2)

    # Add ground plane.
    self.mjcf_model.worldbody.add(
        'geom',
        name='ground',
        type='plane',
        material=groundplane_material,
        size=(1, 1, 0.1),
        friction=(0.4,),
        solimp=(0.95, 0.99, 0.001),
        solref=(0.002, 1))

    # Add lighting
    self.mjcf_model.worldbody.add(
        'light',
        pos=(0, 0, 1.5),
        dir=(0, 0, -1),
        diffuse=(0.7, 0.7, 0.7),
        specular=(.3, .3, .3),
        directional='false',
        castshadow='true')

    # Add some fixed cameras to the arena.
    self._front_camera = self.mjcf_model.worldbody.add(
        'camera',
        name='front',
        pos=(0., -0.6, 0.75),
        xyaxes=(1., 0., 0., 0., 0.7, 0.75))

    # Ensures a 7x7 go board fits into the view from camera
    self._front_camera_2 = self.mjcf_model.worldbody.add(
        'camera',
        name='front_2',
        pos=(0., -0.65, 0.85),
        xyaxes=(1., 0., 0., 0., 0.85, 0.6))

    self._top_down_camera = self.mjcf_model.worldbody.add(
        'camera',
        name='top_down',
        pos=(0., 0., 0.5),
        xyaxes=(1., 0., 0., 0., 1., 0.))

    # Always initialize the free camera so that it points at the origin.
    self.mjcf_model.statistic.center = (0., 0., 0.)

  def _build_observables(self):
    return ArenaObservables(self)

  @property
  def front_camera(self):
    return self._front_camera

  @property
  def front_camera_2(self):
    return self._front_camera_2

  @property
  def top_down_camera(self):
    return self._top_down_camera

  def attach_offset(self, entity, offset, attach_site=None):
    """Attaches another entity at a position offset from the attachment site.

    Args:
      entity: The `Entity` to attach.
      offset: A length 3 array-like object representing the XYZ offset.
      attach_site: (optional) The site to which to attach the entity's model.
        If not set, defaults to self.attachment_site.
    Returns:
      The frame of the attached model.
    """
    frame = self.attach(entity, attach_site=attach_site)
    frame.pos = offset
    return frame


class ArenaObservables(composer.Observables):
  """Observables belonging to the arena."""

  @composer.observable
  def front_camera(self):
    return observable.MJCFCamera(mjcf_element=self._entity.front_camera)

  @composer.observable
  def front_camera_2(self):
    return observable.MJCFCamera(mjcf_element=self._entity.front_camera_2)

  @composer.observable
  def top_down_camera(self):
    return observable.MJCFCamera(mjcf_element=self._entity.top_down_camera)

  @composer.observable
  def top_down_camera_invisible_robot(self):
    # Custom scene options for making robot geoms invisible.
    robot_geoms_invisible = wrapper.MjvOption()
    robot_geoms_invisible.geomgroup[ROBOT_GEOM_GROUP] = 0
    return observable.MJCFCamera(mjcf_element=self._entity.top_down_camera,
                                 scene_option=robot_geoms_invisible)
