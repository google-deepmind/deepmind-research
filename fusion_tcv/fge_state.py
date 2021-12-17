# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A nice python representation of the underlying FGE state."""

from typing import List, Tuple

import numpy as np

from fusion_tcv import shape
from fusion_tcv import shapes_known
from fusion_tcv import tcv_common


class StopSignalException(Exception):  # pylint: disable=g-bad-exception-name
  """This is raised if the FGE environment raises the Stop/Alarm signal."""
  pass


class InvalidSolutionError(RuntimeError):
  """This is raised if returned solution is invalid."""
  pass


class UnhandledOctaveError(Exception):
  """This is raised if some Octave code raises an unhandled error."""
  pass


class FGEState:
  """A nice python representation of the underlying FGE State.

  Given that FGE isn't open source, all of these numbers are made up, and only
  a sketch of what it could look like.
  """

  def __init__(self, num_plasmas):
    self._num_plasmas = num_plasmas

  @property
  def num_plasmas(self) -> int:
    return self._num_plasmas  # Return 1 for singlet, 2 for droplets.

  @property
  def rzip_d(self) -> Tuple[List[float], List[float], List[float]]:
    """Returns the R, Z, and Ip for each plasma domain."""
    if self.num_plasmas == 1:
      return [0.9], [0], [-120000]
    else:
      return [0.9, 0.88], [0.4, -0.4], [-60000, -65000]

  def get_coil_currents_by_type(self, coil_type) -> np.ndarray:
    currents = tcv_common.TCV_ACTION_RANGES.new_random_named_array()
    return currents[coil_type] * tcv_common.ENV_COIL_MAX_CURRENTS[coil_type] / 5

  def get_lcfs_points(self, domain: int) -> shape.ShapePoints:
    del domain  # Should be plasma domain specific
    return shapes_known.SHAPE_70166_0872.canonical().points

  def get_observation_vector(self) -> np.ndarray:
    return tcv_common.TCV_MEASUREMENT_RANGES.new_random_named_array().array

  @property
  def elongation(self) -> List[float]:
    return [1.4] * self.num_plasmas

  @property
  def triangularity(self) -> List[float]:
    return [0.25] * self.num_plasmas

  @property
  def radius(self) -> List[float]:
    return [0.23] * self.num_plasmas

  @property
  def limit_point_d(self) -> List[shape.Point]:
    return [shape.Point(tcv_common.INNER_LIMITER_R, 0.2)] * self.num_plasmas

  @property
  def is_diverted_d(self) -> List[bool]:
    return [False] * self.num_plasmas

  @property
  def x_points(self) -> shape.ShapePoints:
    return []

  @property
  def flux(self) -> np.ndarray:
    """Return the flux at the grid coordinates."""
    return np.random.random((len(self.z_coordinates), len(self.r_coordinates)))

  @property
  def magnetic_axis_flux_strength(self) -> float:
    """The magnetic flux at the center of the plasma."""
    return 2

  @property
  def lcfs_flux_strength(self) -> float:
    """The flux at the LCFS."""
    return 1

  @property
  def r_coordinates(self) -> np.ndarray:
    """The radial coordinates of the simulation."""
    return np.arange(tcv_common.INNER_LIMITER_R, tcv_common.OUTER_LIMITER_R,
                     tcv_common.LIMITER_WIDTH / 10)  # Made up grid resolution.

  @property
  def z_coordinates(self):
    """The vertical coordinates of the simulation."""
    return np.arange(-0.75, 0.75, 1.5 / 30)  # Made up numbers.
