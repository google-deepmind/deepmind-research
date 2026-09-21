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
"""Tools for varying parameters from simulation to simulation."""

from typing import Dict, Optional, Tuple

import dataclasses
import numpy as np

from fusion_tcv import tcv_common

# Pylint does not like variable names like `qA`.
# pylint: disable=invalid-name

RP_DEFAULT = 5e-6
LP_DEFAULT = 2.05e-6
BP_DEFAULT = 0.25
QA_DEFAULT = 1.3


@dataclasses.dataclass
class Settings:
  """Settings to modify solver/plasma model."""
  # Inverse of the resistivity.
  # Plasma circuit equation is roughly
  #  k * dIoh/dt = L * dIp/dt + R * I = Vloop
  # where R is roughly (1 / signeo) or rp.
  # Value is multiplier on the default value.
  # This parameter does not apply to the OhmTor diffusion.
  signeo: Tuple[float, float] = (1, 1)
  # Rp Plasma resistivity. The value is an absolute value.
  rp: float = RP_DEFAULT
  # Plasma self-inductance. The value is an absolute value.
  lp: float = LP_DEFAULT
  # Proportional to the plasma pressure. The value is an absolute value.
  bp: float = BP_DEFAULT
  # Plasma current profile. Value is absolute.
  qA: float = QA_DEFAULT
  # Initial OH coil current. Applied to both coils.
  ioh: Optional[float] = None
  # The voltage offsets for the various coils.
  psu_voltage_offset: Optional[Dict[str, float]] = None

  def _psu_voltage_offset_string(self) -> str:
    """Return a short-ish, readable string of the psu voltage offsets."""
    if not self.psu_voltage_offset:
      return "None"
    if len(self.psu_voltage_offset) < 8:  # Only a few, output individually.
      return ", ".join(
          f"{coil.replace('_00', '')}: {offset:.0f}"
          for coil, offset in self.psu_voltage_offset.items())
    # Otherwise, too long, so output in groups.
    groups = []
    for coil, action_range in tcv_common.TCV_ACTION_RANGES.ranges():
      offsets = [self.psu_voltage_offset.get(tcv_common.TCV_ACTIONS[i], 0)
                 for i in action_range]
      if any(offsets):
        groups.append(f"{coil}: " + ",".join(f"{offset:.0f}"
                                             for offset in offsets))
    return ", ".join(groups)


class ParamGenerator:
  """Varies parameters using uniform/loguniform distributions.

  Absolute parameters are varied using uniform distributions while scaling
  parameters use a loguniform distribution.
  """

  def __init__(self,
               rp_bounds: Optional[Tuple[float, float]] = None,
               lp_bounds: Optional[Tuple[float, float]] = None,
               qA_bounds: Optional[Tuple[float, float]] = None,
               bp_bounds: Optional[Tuple[float, float]] = None,
               rp_mean: float = RP_DEFAULT,
               lp_mean: float = LP_DEFAULT,
               bp_mean: float = BP_DEFAULT,
               qA_mean: float = QA_DEFAULT,
               ioh_bounds: Optional[Tuple[float, float]] = None,
               psu_voltage_offset_bounds: Optional[
                   Dict[str, Tuple[float, float]]] = None):
    # Do not allow Signeo variation as this does not work with OhmTor current
    # diffusion.
    no_scaling = (1, 1)
    self._rp_bounds = rp_bounds if rp_bounds else no_scaling
    self._lp_bounds = lp_bounds if lp_bounds else no_scaling
    self._bp_bounds = bp_bounds if bp_bounds else no_scaling
    self._qA_bounds = qA_bounds if qA_bounds else no_scaling
    self._rp_mean = rp_mean
    self._lp_mean = lp_mean
    self._bp_mean = bp_mean
    self._qA_mean = qA_mean
    self._ioh_bounds = ioh_bounds
    self._psu_voltage_offset_bounds = psu_voltage_offset_bounds

  def generate(self) -> Settings:
    return Settings(
        signeo=(1, 1),
        rp=loguniform_rv(*self._rp_bounds) * self._rp_mean,
        lp=loguniform_rv(*self._lp_bounds) * self._lp_mean,
        bp=loguniform_rv(*self._bp_bounds) * self._bp_mean,
        qA=loguniform_rv(*self._qA_bounds) * self._qA_mean,
        ioh=np.random.uniform(*self._ioh_bounds) if self._ioh_bounds else None,
        psu_voltage_offset=(
            {coil: np.random.uniform(*bounds)
             for coil, bounds in self._psu_voltage_offset_bounds.items()}
            if self._psu_voltage_offset_bounds else None))


def loguniform_rv(lower, upper):
  """Generate loguniform random variable between min and max."""
  if lower == upper:
    return lower
  assert lower < upper
  return np.exp(np.random.uniform(np.log(lower), np.log(upper)))
