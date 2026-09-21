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
"""Run an agent on the environment."""

import numpy as np

from fusion_tcv import environment
from fusion_tcv import trajectory


def run_loop(env: environment.Environment, agent,
             max_steps: int = 100000) -> trajectory.Trajectory:
  """Run an agent."""
  results = []
  agent.reset()
  ts = env.reset()
  for _ in range(max_steps):
    obs = ts.observation
    action = agent.step(ts)
    ts = env.step(action)
    results.append(trajectory.Trajectory(
        measurements=obs["measurements"],
        references=obs["references"],
        actions=action,
        reward=np.array(ts.reward)))
    if ts.last():
      break

  return trajectory.Trajectory.stack(results)
