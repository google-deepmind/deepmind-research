# Copyright 2019 DeepMind Technologies Limited.
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
"""Q-learning with side effects penalties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from side_effects_penalties import agent
from side_effects_penalties import side_effects_penalty as sep


class QLearningSE(agent.QLearning):
  """Q-learning agent with side-effects penalties."""

  def __init__(
      self, actions, alpha=0.1, epsilon=0.1, q_initialisation=0.0,
      baseline='start', dev_measure='none', dev_fun='truncation',
      discount=0.99, value_discount=1.0, beta=1.0, num_util_funs=10,
      exact_baseline=False, baseline_env=None, start_timestep=None,
      state_size=None, nonterminal_weight=0.01):
    """Create a Q-learning agent with a side effects penalty.

    Args:
      actions: full discrete action spec.
      alpha: agent learning rate.
      epsilon: agent exploration rate.
      q_initialisation: float, used to initialise the value function.
      baseline: which baseline state to use ('start', 'inaction', 'stepwise').
      dev_measure: deviation measure:
        - "none" for no penalty,
        - "reach" for unreachability,
        - "rel_reach" for relative reachability,
        - "att_util" for attainable utility,
      dev_fun: what function to apply in the deviation measure ('truncation' or
        'absolute' (for 'rel_reach' and 'att_util'), or 'none' (otherwise)).
      discount: discount factor for rewards.
      value_discount: discount factor for value functions in penalties.
      beta: side effects penalty weight.
      num_util_funs: number of random utility functions for attainable utility.
      exact_baseline: whether to use an exact or approximate baseline.
      baseline_env: copy of environment (with noops) for the exact baseline.
      start_timestep: copy of starting timestep for the baseline.
      state_size: the size of each state (flattened) for NN reachability.
      nonterminal_weight: penalty weight on nonterminal states.

    Raises:
      ValueError: for incorrect baseline, dev_measure, or dev_fun
    """

    super(QLearningSE, self).__init__(actions, alpha, epsilon, q_initialisation,
                                      discount)

    # Impact penalty: set dev_fun (f)
    if 'rel_reach' in dev_measure or 'att_util' in dev_measure:
      if dev_fun == 'truncation':
        dev_fun = lambda diff: np.maximum(0, diff)
      elif dev_fun == 'absolute':
        dev_fun = np.abs
      else:
        raise ValueError('Deviation function not recognized')
    else:
      assert dev_fun == 'none'
      dev_fun = None

    # Impact penalty: create deviation measure
    if dev_measure in {'reach', 'rel_reach'}:
      deviation = sep.Reachability(value_discount, dev_fun, discount)
    elif dev_measure == 'uvfa_rel_reach':
      deviation = sep.UVFAReachability(value_discount, dev_fun, discount,
                                       state_size)
    elif dev_measure == 'att_util':
      deviation = sep.AttainableUtility(value_discount, dev_fun, num_util_funs,
                                        discount)
    elif dev_measure == 'none':
      deviation = sep.NoDeviation()
    else:
      raise ValueError('Deviation measure not recognized')

    use_inseparable_rollout = (
        dev_measure == 'reach' and baseline == 'stepwise')

    # Impact penalty: create baseline
    if baseline in {'start', 'inaction', 'stepwise'}:
      baseline_class = getattr(sep, baseline.capitalize() + 'Baseline')
      baseline = baseline_class(start_timestep, exact_baseline, baseline_env,
                                self._timestep_to_state)
    elif baseline == 'step_noroll':
      baseline_class = getattr(sep, 'StepwiseBaseline')
      baseline = baseline_class(start_timestep, exact_baseline, baseline_env,
                                self._timestep_to_state, False)
    else:
      raise ValueError('Baseline not recognized')

    self._impact_penalty = sep.SideEffectPenalty(
        baseline, deviation, beta, nonterminal_weight, use_inseparable_rollout)

  def begin_episode(self):
    """Perform episode initialisation."""
    super(QLearningSE, self).begin_episode()
    self._impact_penalty.reset()

  def _calculate_reward(self, timestep, state):
    reward = super(QLearningSE, self)._calculate_reward(timestep, state)
    return (reward - self._impact_penalty.calculate(
        self._current_state, self._current_action, state))
