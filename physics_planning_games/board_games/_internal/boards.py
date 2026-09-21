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

"""Composer entities corresponding to game boards."""

import copy
import os

from dm_control import composer
from dm_control import mjcf
import numpy as np

from dm_control.utils import io as resources

_TOUCH_THRESHOLD = 1e-3  # Activation threshold for touch sensors (N).

# whether to display underlying sensors for Goboard (useful to align texture)
_SHOW_DEBUG_GRID = False
_TEXTURE_PATH = os.path.join(os.path.dirname(__file__), 'goboard_7x7.png')


def _make_checkerboard(rows,
                       columns,
                       square_halfwidth,
                       height=0.01,
                       sensor_size=0.7,
                       name='checkerboard'):
  """Builds a checkerboard with touch sensors centered on each square."""
  root = mjcf.RootElement(model=name)
  black_mat = root.asset.add('material', name='black', rgba=(0.2, 0.2, 0.2, 1))
  white_mat = root.asset.add('material', name='white', rgba=(0.8, 0.8, 0.8, 1))
  sensor_mat = root.asset.add('material', name='sensor', rgba=(0, 1, 0, 0.3))
  root.default.geom.set_attributes(
      type='box', size=(square_halfwidth, square_halfwidth, height))
  root.default.site.set_attributes(
      type='box',
      size=(sensor_size * square_halfwidth,) * 2 + (0.5 * height,),
      material=sensor_mat, group=composer.SENSOR_SITES_GROUP)

  xpos = (np.arange(columns) - 0.5*(columns - 1)) * 2 * square_halfwidth
  ypos = (np.arange(rows) - 0.5*(rows - 1)) * 2 * square_halfwidth
  geoms = []
  touch_sensors = []
  for i in range(rows):
    for j in range(columns):
      geom_mat = black_mat if ((i % 2) == (j % 2)) else white_mat
      name = '{}_{}'.format(i, j)
      geoms.append(
          root.worldbody.add(
              'geom',
              pos=(xpos[j], ypos[i], height),
              name=name,
              material=geom_mat))
      site = root.worldbody.add('site', pos=(xpos[j], ypos[i], 2*height),
                                name=name)
      touch_sensors.append(root.sensor.add('touch', site=site, name=name))

  return root, geoms, touch_sensors


def _make_goboard(boardsize,
                  square_halfwidth,
                  height=0.01,
                  sensor_size=0.7,
                  name='goboard'):
  """Builds a Go with touch sensors centered on each intersection."""
  y_offset = -0.08
  rows = boardsize
  columns = boardsize
  root = mjcf.RootElement(model=name)
  if _SHOW_DEBUG_GRID:
    black_mat = root.asset.add('material', name='black',
                               rgba=(0.2, 0.2, 0.2, 0.5))
    white_mat = root.asset.add('material', name='white',
                               rgba=(0.8, 0.8, 0.8, 0.5))
  else:
    transparent_mat = root.asset.add('material', name='intersection',
                                     rgba=(0, 1, 0, 0.0))

  sensor_mat = root.asset.add('material', name='sensor', rgba=(0, 1, 0, 0.3))

  contents = resources.GetResource(_TEXTURE_PATH)
  root.asset.add('texture', name='goboard', type='2d',
                 file=mjcf.Asset(contents, '.png'))
  board_mat = root.asset.add(
      'material', name='goboard', texture='goboard',
      texrepeat=[0.97, 0.97])

  root.default.geom.set_attributes(
      type='box', size=(square_halfwidth, square_halfwidth, height))
  root.default.site.set_attributes(
      type='box',
      size=(sensor_size * square_halfwidth,) * 2 + (0.5 * height,),
      material=sensor_mat, group=composer.SENSOR_SITES_GROUP)

  board_height = height
  if _SHOW_DEBUG_GRID:
    board_height = 0.5*height

  root.worldbody.add(
      'geom',
      pos=(0, 0+y_offset, height),
      type='box',
      size=(square_halfwidth * boardsize,) * 2 + (board_height,),
      name=name,
      material=board_mat)

  xpos = (np.arange(columns) - 0.5*(columns - 1)) * 2 * square_halfwidth
  ypos = (np.arange(rows) - 0.5*(rows - 1)) * 2 * square_halfwidth + y_offset
  geoms = []
  touch_sensors = []
  for i in range(rows):
    for j in range(columns):
      name = '{}_{}'.format(i, j)
      if _SHOW_DEBUG_GRID:
        transparent_mat = black_mat if ((i % 2) == (j % 2)) else white_mat
      geoms.append(
          root.worldbody.add(
              'geom',
              pos=(xpos[j], ypos[i], height),
              name=name,
              material=transparent_mat))
      site = root.worldbody.add('site', pos=(xpos[j], ypos[i], 2*height),
                                name=name)
      touch_sensors.append(root.sensor.add('touch', site=site, name=name))

  pass_geom = root.worldbody.add(
      'geom',
      pos=(0, y_offset, 0.0),
      size=(square_halfwidth*boardsize*2,
            square_halfwidth*boardsize)  + (0.5 * height,),
      name='pass',
      material=transparent_mat)
  site = root.worldbody.add('site', pos=(0, y_offset, 0.0),
                            size=(square_halfwidth*boardsize*2,
                                  square_halfwidth*boardsize) + (0.5 * height,),
                            name='pass')
  pass_sensor = root.sensor.add('touch', site=site, name='pass')

  return root, geoms, touch_sensors, pass_geom, pass_sensor


class CheckerBoard(composer.Entity):
  """An entity representing a checkerboard."""

  def __init__(self, *args, **kwargs):
    super(CheckerBoard, self).__init__(*args, **kwargs)
    self._contact_from_before_substep = None

  def _build(self, rows=3, columns=3, square_halfwidth=0.05):
    """Builds a `CheckerBoard` entity.

    Args:
      rows: Integer, the number of rows.
      columns: Integer, the number of columns.
      square_halfwidth: Float, the halfwidth of the squares on the board.
    """
    root, geoms, touch_sensors = _make_checkerboard(
        rows=rows, columns=columns, square_halfwidth=square_halfwidth)
    self._mjcf_model = root
    self._geoms = np.array(geoms).reshape(rows, columns)
    self._touch_sensors = np.array(touch_sensors).reshape(rows, columns)

  @property
  def mjcf_model(self):
    return self._mjcf_model

  def before_substep(self, physics, random_state):
    del random_state  # Unused.
    # Cache a copy of the array of active contacts before each substep.
    self._contact_from_before_substep = [
        copy.copy(c) for c in physics.data.contact
    ]

  def validate_finger_touch(self, physics, row, col, hand):
    # Geom for the board square
    geom_id = physics.bind(self._geoms[row, col]).element_id
    # finger geoms
    finger_geoms_ids = set(physics.bind(hand.finger_geoms).element_id)
    contacts = self._contact_from_before_substep

    set1, set2 = set([geom_id]), finger_geoms_ids
    for contact in contacts:
      finger_tile_contact = ((contact.geom1 in set1 and
                              contact.geom2 in set2) or
                             (contact.geom1 in set2 and contact.geom2 in set1))
      if finger_tile_contact:
        return True
    return False

  def get_contact_pos(self, physics, row, col):
    geom_id = physics.bind(self._geoms[row, col]).element_id
    # Here we use the array of active contacts from the previous substep, rather
    # than the current values in `physics.data.contact`. This is because we use
    # touch sensors to detect when a square on the board is being pressed, and
    # the pressure readings are based on forces that were calculated at the end
    # of the previous substep. It's possible that `physics.data.contact` no
    # longer contains any active contacts involving the board geoms, even though
    # the touch sensors are telling us that one of the squares on the board is
    # being pressed.
    contacts = self._contact_from_before_substep
    relevant_contacts = [
        c for c in contacts if c.geom1 == geom_id or c.geom2 == geom_id
    ]
    if relevant_contacts:
      # If there are multiple contacts involving this square of the board, just
      # pick the first one.
      return relevant_contacts[0].pos.copy()
    else:
      print("Touch sensor at ({},{}) doesn't have any active contacts!".format(
          row, col))
      return False

  def get_contact_indices(self, physics):
    pressures = physics.bind(self._touch_sensors.ravel()).sensordata
    # If any of the touch sensors exceed the threshold, return the (row, col)
    # indices of the most strongly activated sensor.
    if np.any(pressures > _TOUCH_THRESHOLD):
      return np.unravel_index(np.argmax(pressures), self._touch_sensors.shape)
    else:
      return None

  def sample_pos_inside_touch_sensor(self, physics, random_state, row, col):
    bound_site = physics.bind(self._touch_sensors[row, col].site)
    jitter = bound_site.size * np.array([1., 1., 0.])
    return bound_site.xpos + random_state.uniform(-jitter, jitter)


class GoBoard(CheckerBoard):
  """An entity representing a Goboard."""

  def _build(self, boardsize=7, square_halfwidth=0.05):  # pytype: disable=signature-mismatch  # overriding-default-value-checks
    """Builds a `GoBoard` entity.

    Args:
      boardsize: Integer, the size of the board (boardsize x boardsize).
      square_halfwidth: Float, the halfwidth of the squares on the board.
    """

    if boardsize != 7:
      raise ValueError('Only boardsize of 7x7 is implemented at the moment')

    root, geoms, touch_sensors, pass_geom, pass_sensor = _make_goboard(
        boardsize=boardsize, square_halfwidth=square_halfwidth)
    self._mjcf_model = root
    self._geoms = np.array(geoms).reshape(boardsize, boardsize)
    self._touch_sensors = np.array(touch_sensors).reshape(boardsize, boardsize)
    self._pass_geom = pass_geom
    self._pass_sensor = pass_sensor

  def get_contact_indices(self, physics):
    pressures = physics.bind(self._touch_sensors.ravel()).sensordata
    # Deal with pass first
    pass_pressure = physics.bind(self._pass_sensor).sensordata
    if pass_pressure > np.max(pressures) and pass_pressure > _TOUCH_THRESHOLD:
      return -1, -1

    # If any of the other touch sensors exceed the threshold, return the
    # (row, col) indices of the most strongly activated sensor.
    if np.any(pressures > _TOUCH_THRESHOLD):
      return np.unravel_index(np.argmax(pressures), self._touch_sensors.shape)
    else:
      return None

  def validate_finger_touch(self, physics, row, col, hand):
    # Geom for the board square
    if row == -1 and col == -1:
      geom_id = physics.bind(self._pass_geom).element_id
    else:
      geom_id = physics.bind(self._geoms[row, col]).element_id
    # finger geoms
    finger_geoms_ids = set(physics.bind(hand.finger_geoms).element_id)
    contacts = self._contact_from_before_substep

    set1, set2 = set([geom_id]), finger_geoms_ids
    for contact in contacts:
      finger_tile_contact = ((contact.geom1 in set1 and
                              contact.geom2 in set2) or
                             (contact.geom1 in set2 and contact.geom2 in set1))
      if finger_tile_contact:
        return True
    return False

  def sample_pos_inside_touch_sensor(self, physics, random_state, row, col):
    bound_site = physics.bind(self._touch_sensors[row, col].site)
    jitter = bound_site.size * np.array([0.25, 0.25, 0.])
    return bound_site.xpos + random_state.uniform(-jitter, jitter)
