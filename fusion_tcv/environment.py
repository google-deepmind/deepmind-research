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
"""Environment API for FGE simulator."""

from typing import Dict, List, Optional

import dm_env
from dm_env import auto_reset_environment
from dm_env import specs
import numpy as np

from fusion_tcv import fge_octave
from fusion_tcv import fge_state
from fusion_tcv import named_array
from fusion_tcv import noise
from fusion_tcv import param_variation
from fusion_tcv import ref_gen
from fusion_tcv import rewards
from fusion_tcv import tcv_common
from fusion_tcv import terminations


# Re-export as fge_octave should be an implementation detail.
ShotCondition = fge_octave.ShotCondition


class Environment(auto_reset_environment.AutoResetEnvironment):
  """An environment using the FGE Solver.

  The simulator will return a flux map, which is the environment's hidden state,
  and some flux measurements, which will be used as observations. The actions
  represent current levels that are passed to the simulator for the next
  flux calculation.
  """

  def __init__(
      self,
      shot_condition: ShotCondition,
      reward: rewards.AbstractReward,
      reference_generator: ref_gen.AbstractReferenceGenerator,
      max_episode_length: int = 10000,
      termination: Optional[terminations.Abstract] = None,
      obs_act_noise: Optional[noise.Noise] = None,
      power_supply_delays: Optional[Dict[str, List[float]]] = None,
      param_generator: Optional[param_variation.ParamGenerator] = None):
    """Initializes an Environment instance.

    Args:
      shot_condition: A ShotCondition, specifying shot number and time. This
        specifies the machine geometry (eg with or without the baffles), and the
        initial measurements, voltages, current and plasma state.
      reward: Function to generate a reward term.
      reference_generator: Generator for the signal to send to references.
      max_episode_length: Maximum number of steps before episode is truncated
        and restarted.
      termination: Decide if the state should be considered a termination.
      obs_act_noise: Type for setting the observation and action noise. If noise
        is set to None then the default noise level is used.
      power_supply_delays: A dict with power supply delays (in seconds), keys
        are coil type labels ('E', 'F', 'G', 'OH'). `None` means default delays.
      param_generator: Generator for Liuqe parameter settings. If None then
        the default settings are used.
    """
    super().__init__()
    if power_supply_delays is None:
      power_supply_delays = tcv_common.TCV_ACTION_DELAYS
    self._simulator = fge_octave.FGESimulatorOctave(
        shot_condition=shot_condition,
        power_supply_delays=power_supply_delays)
    self._reward = reward
    self._reference_generator = reference_generator
    self._max_episode_length = max_episode_length
    self._termination = (termination if termination is not None else
                         terminations.CURRENT_OH_IP)
    self._noise = (obs_act_noise if obs_act_noise is not None else
                   noise.Noise.use_default_noise())
    self._param_generator = (param_generator if param_generator is not None else
                             param_variation.ParamGenerator())
    self._params = None
    self._step_counter = 0
    self._last_observation = None

  def observation_spec(self):
    """Defines the observations provided by the environment."""
    return tcv_common.observation_spec()

  def action_spec(self) -> specs.BoundedArray:
    """Defines the actions that should be provided to `step`."""
    return tcv_common.action_spec()

  def _reset(self) -> dm_env.TimeStep:
    """Starts a new episode."""
    self._step_counter = 0
    self._params = self._param_generator.generate()
    state = self._simulator.reset(self._params)
    references = self._reference_generator.reset()
    zero_act = np.zeros(self.action_spec().shape,
                        dtype=self.action_spec().dtype)
    self._last_observation = self._extract_observation(
        state, references, zero_act)
    return dm_env.restart(self._last_observation)

  def _simulator_voltages_from_voltages(self, voltages):
    voltage_simulator = np.copy(voltages)
    if self._params.psu_voltage_offset is not None:
      for coil, offset in self._params.psu_voltage_offset.items():
        voltage_simulator[tcv_common.TCV_ACTION_INDICES[coil]] += offset
    voltage_simulator = np.clip(
        voltage_simulator,
        self.action_spec().minimum,
        self.action_spec().maximum)
    g_coil = tcv_common.TCV_ACTION_RANGES.index("G")
    if abs(voltage_simulator[g_coil]) < tcv_common.ENV_G_COIL_DEADBAND:
      voltage_simulator[g_coil] = 0
    return voltage_simulator

  def _step(self, action: np.ndarray) -> dm_env.TimeStep:
    """Does one step within TCV."""
    voltages = self._noise.add_action_noise(action)
    voltage_simulator = self._simulator_voltages_from_voltages(voltages)
    try:
      state = self._simulator.step(voltage_simulator)
    except (fge_state.InvalidSolutionError,
            fge_state.StopSignalException):
      return dm_env.termination(
          self._reward.terminal_reward(), self._last_observation)
    references = self._reference_generator.step()
    self._last_observation = self._extract_observation(
        state, references, action)
    term = self._termination.terminate(state)
    if term:
      return dm_env.termination(
          self._reward.terminal_reward(), self._last_observation)
    reward, _ = self._reward.reward(voltages, state, references)
    self._step_counter += 1
    if self._step_counter >= self._max_episode_length:
      return dm_env.truncation(reward, self._last_observation)
    return dm_env.transition(reward, self._last_observation)

  def _extract_observation(
      self, state: fge_state.FGEState,
      references: named_array.NamedArray,
      action: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "references": references.array,
        "measurements": self._noise.add_measurement_noise(
            state.get_observation_vector()),
        "last_action": action,
    }
