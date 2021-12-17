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
"""Terminations for the fusion environment."""

import abc
from typing import List, Optional

import numpy as np

from fusion_tcv import fge_state
from fusion_tcv import tcv_common


class Abstract(abc.ABC):
  """Abstract reward class."""

  @abc.abstractmethod
  def terminate(self, state: fge_state.FGEState) -> Optional[str]:
    """Returns a reason if the situation should be considered a termination."""


class CoilCurrentSaturation(Abstract):
  """Terminates if the coils have saturated their current."""

  def terminate(self, state: fge_state.FGEState) -> Optional[str]:
    # Coil currents are checked by type, independent of the order.
    for coil_type, max_current in tcv_common.ENV_COIL_MAX_CURRENTS.items():
      if coil_type == "DUMMY":
        continue
      currents = state.get_coil_currents_by_type(coil_type)
      if (np.abs(currents) > max_current).any():
        return (f"CoilCurrentSaturation: {coil_type}, max: {max_current}, "
                "real: " + ", ".join(f"{c:.1f}" for c in currents))
    return None


class OHTooDifferent(Abstract):
  """Terminates if the coil currents are too far apart from one another."""

  def __init__(self, max_diff: float):
    self._max_diff = max_diff

  def terminate(self, state: fge_state.FGEState) -> Optional[str]:
    oh_coil_currents = state.get_coil_currents_by_type("OH")
    assert len(oh_coil_currents) == 2
    oh_current_abs = abs(oh_coil_currents[0] - oh_coil_currents[1])
    if oh_current_abs > self._max_diff:
      return ("OHTooDifferent: currents: "
              f"({oh_coil_currents[0]:.0f}, {oh_coil_currents[1]:.0f}), "
              f"diff: {oh_current_abs:.0f}, max: {self._max_diff}")
    return None


class IPTooLow(Abstract):
  """Terminates if the magnitude of Ip in any component is too low."""

  def __init__(self, singlet_threshold: float, droplet_threshold: float):
    self._singlet_threshold = singlet_threshold
    self._droplet_threshold = droplet_threshold

  def terminate(self, state: fge_state.FGEState) -> Optional[str]:
    _, _, ip_d = state.rzip_d
    if len(ip_d) == 1:
      if ip_d[0] > self._singlet_threshold:  # Sign due to negative Ip.
        return f"IPTooLow: Singlet, {ip_d[0]:.0f}"
      return None
    else:
      if max(ip_d) > self._droplet_threshold:  # Sign due to negative Ip.
        return f"IPTooLow: Components: {ip_d[0]:.0f}, {ip_d[1]:.0f}"
      return None


class AnyTermination(Abstract):
  """Terminates if any of conditions are met."""

  def __init__(self, terminators: List[Abstract]):
    self._terminators = terminators

  def terminate(self, state: fge_state.FGEState) -> Optional[str]:
    for terminator in self._terminators:
      term = terminator.terminate(state)
      if term:
        return term
    return None


CURRENT_OH_IP = AnyTermination([
    CoilCurrentSaturation(),
    OHTooDifferent(max_diff=4000),
    IPTooLow(singlet_threshold=-60000, droplet_threshold=-25000),
])
