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
"""Actually interact with FGE via octave."""

from typing import Dict, List

import dataclasses
import numpy as np

from fusion_tcv import fge_state
from fusion_tcv import param_variation

SUBSTEPS = 5


@dataclasses.dataclass
class ShotCondition:
  """Represents a shot and time from a real shot."""
  shot: int
  time: float


class FGESimulatorOctave:
  """Would interact with the FGE solver via Octave.

  Given that FGE isn't open source, this is just a sketch.
  """

  def __init__(
      self,
      shot_condition: ShotCondition,
      power_supply_delays: Dict[str, List[float]]):
    """Initialize the simulator.

    Args:
      shot_condition: A ShotCondition, specifying shot number and time. This
        specifies the machine geometry (eg with or without the baffles), and the
        initial measurements, voltages, current and plasma shape.
      power_supply_delays: A dict with power supply delays (in seconds), keys
        are coil type labels ('E', 'F', 'G', 'OH'). `None` means default delays.
    """
    del power_supply_delays
    # Initialize the simulator:
    # - Use oct2py to load FGE through Octave.
    # - Load the data for the shot_condition.
    # - Set up the reactor geometry from the shot_condition.
    # - Set the timestep to `tcv_common.DT / SUBSTEPS`.
    # - Set up the solver for singlets or droplets based on the shot_condition.
    self._num_plasmas = 2 if shot_condition.shot == 69198 else 1
    # - Set up the power supply, including the limits, initial data, and delays.

  def reset(self, variation: param_variation.Settings) -> fge_state.FGEState:
    """Restarts the simulator with parameters."""
    del variation
    # Update the simulator with the current physics parameters.
    # Reset to the initial state from the shot_condition.
    return fge_state.FGEState(self._num_plasmas)  # Filled with the real state.

  def step(self, voltages: np.ndarray) -> fge_state.FGEState:
    """Run the simulator with `voltages`, returns the state."""
    del voltages
    # for _ in range(SUBSTEPS):
    #   Step the simulator with `voltages`.
    #   raise fge_state.InvalidSolutionError if the solver doesn't converge.
    #   raise fge_state.StopSignalException if an internal termination triggered
    return fge_state.FGEState(self._num_plasmas)  # Filled with the real state.
