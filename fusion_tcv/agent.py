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
"""An agent interface for interacting with the environment."""

import abc

import dm_env
import numpy as np

from fusion_tcv import tcv_common


class AbstractAgent(abc.ABC):
  """Agent base class."""

  def reset(self):
    """Reset to the initial state."""

  @abc.abstractmethod
  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Return the action given the current observations."""


class ZeroAgent(AbstractAgent):
  """An agent that always returns "zero" actions."""

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    del timestep
    return np.zeros(tcv_common.action_spec().shape,
                    tcv_common.action_spec().dtype)
