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
"""A trajectory for an episode."""

from typing import List

import dataclasses
import numpy as np


@dataclasses.dataclass
class Trajectory:
  """A trajectory of actions/obs for an episode."""
  measurements: np.ndarray
  references: np.ndarray
  reward: np.ndarray
  actions: np.ndarray

  @classmethod
  def stack(cls, series: List["Trajectory"]) -> "Trajectory":
    """Stack a series of trajectories, adding a trailing time dimension."""
    values = {k: np.empty(v.shape + (len(series),))
              for k, v in dataclasses.asdict(series[0]).items()
              if v is not None}
    for i, ts in enumerate(series):
      for k, v in values.items():
        v[..., i] = getattr(ts, k)
    out = cls(**values)
    return out
