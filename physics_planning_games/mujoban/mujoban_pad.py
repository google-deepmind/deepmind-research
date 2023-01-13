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

"""A floor pad that is activated through touch."""

import weakref
from dm_control import composer
from dm_control import mjcf
import numpy as np


def _get_activator_box(pad_xpos, pad_size, boxes, tolerance=0.0):
  """Returns the activator box, if any. Otherwise returns None."""
  # Ignore the height
  pad_min = pad_xpos[0:2] - pad_size[0:2]
  pad_max = pad_xpos[0:2] + pad_size[0:2]
  for box in boxes:
    box_xpos = np.array(box.xpos[0:2])
    box_size = np.array(box.size[0:2])

    min_ = pad_min + box_size - tolerance
    max_ = pad_max - box_size + tolerance
    in_range = np.logical_and(box_xpos >= min_, box_xpos <= max_).all()
    if in_range:
      return box
  # No activator box was found
  return None


class MujobanPad(composer.Entity):
  """A less sensitive floor pad for Mujoban."""

  def _build(self, rgba=None, pressed_rgba=None,
             size=1, height=0.02, detection_tolerance=0.0, name='mujoban_pad'):
    rgba = tuple(rgba or (1, 0, 0, 1))
    pressed_rgba = tuple(pressed_rgba or (0.2, 0, 0, 1))
    self._mjcf_root = mjcf.RootElement(model=name)
    self._site = self._mjcf_root.worldbody.add(
        'site', type='box', name='site',
        pos=[0, 0, (height / 2 or -0.001)],
        size=[size / 2, size / 2, (height / 2 or 0.001)], rgba=rgba)
    self._activated = False
    self._rgba = np.array(rgba, dtype=float)
    self._pressed_rgba = np.array(pressed_rgba, dtype=float)
    self._activator = None
    self._detection_tolerance = detection_tolerance
    self._boxes = []

  @property
  def rgba(self):
    return self._rgba

  @property
  def pressed_rgba(self):
    return self._pressed_rgba

  def register_box(self, box_entity):
    self._boxes.append(weakref.proxy(box_entity))

  @property
  def site(self):
    return self._site

  @property
  def boxes(self):
    return self._boxes

  @property
  def activator(self):
    return self._activator if self._activated else None

  @property
  def mjcf_model(self):
    return self._mjcf_root

  def initialize_episode_mjcf(self, unused_random_state):
    self._activated = False

  def initialize_episode(self, physics, unused_random_state):
    self._update_activation(physics)

  def _update_activation(self, physics):
    # Note: we get the physically bound box, not an object from self._boxes.
    # That's because the generator expression below generates bound objects.
    box = _get_activator_box(
        pad_xpos=np.array(physics.bind(self._site).xpos),
        pad_size=np.array(physics.bind(self._site).size),
        boxes=(physics.bind(box.geom) for box in self._boxes),
        tolerance=self._detection_tolerance,)
    if box:
      self._activated = True
      self._activator = box
    else:
      self._activated = False
      self._activator = None
    if self._activated:
      physics.bind(self._site).rgba = self._pressed_rgba
    else:
      physics.bind(self._site).rgba = self._rgba

  def before_step(self, physics, unused_random_state):
    self._update_activation(physics)

  def after_substep(self, physics, unused_random_state):
    self._update_activation(physics)

  @property
  def activated(self):
    """Whether this floor pad is pressed at the moment."""
    return self._activated

  def reset(self, physics):
    self._activated = False
    physics.bind(self._site).rgba = self._rgba
