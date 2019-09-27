# Copyright 2019 DeepMind Technologies Limited and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""File utilities."""

import math
import os
import numpy as np
from PIL import Image


class FileExporter(object):
  """File exporter utilities."""

  def __init__(self, path, grid_height=None, zoom=1):
    """Constructor.

    Arguments:
      path: The directory to save data to.
      grid_height: How many data elements tall to make the grid, if appropriate.
          The width will be chosen based on height. If None, automatically
          determined.
      zoom: How much to zoom in each data element by, if appropriate.
    """
    if not os.path.exists(path):
      os.makedirs(path)

    self._path = path
    self._zoom = zoom
    self._grid_height = grid_height

  def _reshape(self, data):
    """Reshape given data into image format."""
    batch_size, height, width, n_channels = data.shape
    if self._grid_height:
      grid_height = self._grid_height
    else:
      grid_height = int(math.floor(math.sqrt(batch_size)))

    grid_width = int(math.ceil(batch_size/grid_height))

    if n_channels == 1:
      data = np.tile(data, (1, 1, 1, 3))
      n_channels = 3

    if n_channels != 3:
      raise ValueError('Image batch must have either 1 or 3 channels, but '
                       'was {}'.format(n_channels))

    shape = (height * grid_height, width * grid_width, n_channels)
    buf = np.full(shape, 255, dtype=np.uint8)
    multiplier = 1 if data.dtype in (np.int32, np.int64) else 255

    for k in range(batch_size):
      i = k // grid_width
      j = k % grid_width
      arr = data[k]
      x, y = i * height, j * width
      buf[x:x + height, y:y + width, :] = np.clip(
          multiplier * arr, 0, 255).astype(np.uint8)

    if self._zoom > 1:
      buf = buf.repeat(self._zoom, axis=0).repeat(self._zoom, axis=1)
    return buf

  def save(self, data, name):
    data = self._reshape(data)
    relative_name = '{}_last.png'.format(name)
    target_file = os.path.join(self._path, relative_name)

    img = Image.fromarray(data)
    img.save(target_file, format='PNG')
