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
"""Side Effects Penalties.

Abstract class for implementing a side effects (impact measure) penalty,
and various concrete penalties deriving from it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import copy
import enum
import numpy as np
import six
from six.moves import range
from six.moves import zip


class Actions(enum.IntEnum):
  """Enum for actions the agent can take."""
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3
  NOOP = 4


@six.add_metaclass(abc.ABCMeta)
class Baseline(object):
  """Base class for baseline states."""

  def __init__(self, start_timestep, exact=False, env=None,
               timestep_to_state=None):
    """Create a baseline.

    Args:
      start_timestep: starting state timestep
      exact: whether to use an exact or approximate baseline
      env: a copy of the environment (used to simulate exact baselines)
      timestep_to_state: a function that turns timesteps into states
    """
    self._exact = exact
    self._env = env
    self._timestep_to_state = timestep_to_state
    self._start_timestep = start_timestep
    self._baseline_state = self._timestep_to_state(self._start_timestep)
    self._inaction_next = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0))

  @abc.abstractmethod
  def calculate(self):
    """Update and return the baseline state."""

  def sample(self, state):
    """Sample the outcome of a noop in `state`."""
    d = self._inaction_next[state]
    counts = np.array(list(d.values()))
    index = np.random.choice(a=len(counts), p=counts/sum(counts))
    return list(d.keys())[index]

  def reset(self):
    """Signal start of new episode."""
    self._baseline_state = self._timestep_to_state(self._start_timestep)
    if self._exact:
      self._env.reset()

  @abc.abstractproperty
  def rollout_func(self):
    """Function to compute a rollout chain, or None if n/a."""


class StartBaseline(Baseline):
  """Starting state baseline."""

  def calculate(self, *unused_args):
    return self._baseline_state

  @property
  def rollout_func(self):
    return None


class InactionBaseline(Baseline):
  """Inaction baseline: the state resulting from taking no-ops from start."""

  def calculate(self, prev_state, action, current_state):
    if self._exact:
      self._baseline_state = self._timestep_to_state(
          self._env.step(Actions.NOOP))
    else:
      if action == Actions.NOOP:
        self._inaction_next[prev_state][current_state] += 1
      if self._baseline_state in self._inaction_next:
        self._baseline_state = self.sample(self._baseline_state)
    return self._baseline_state

  @property
  def rollout_func(self):
    return None


class StepwiseBaseline(Baseline):
  """Stepwise baseline: the state one no-op after the previous state."""

  def __init__(self, start_timestep, exact=False, env=None,
               timestep_to_state=None, use_rollouts=True):
    """Create a stepwise baseline.

    Args:
      start_timestep: starting state timestep
      exact: whether to use an exact or approximate baseline
      env: a copy of the environment (used to simulate exact baselines)
      timestep_to_state: a function that turns timesteps into states
      use_rollouts: whether to use inaction rollouts
    """
    super(StepwiseBaseline, self).__init__(
        start_timestep, exact, env, timestep_to_state)
    self._rollouts = use_rollouts

  def calculate(self, prev_state, action, current_state):
    """Update and return the baseline state.

    Args:
      prev_state: the state in which `action` was taken
      action: the action just taken
      current_state: the state resulting from taking `action`
    Returns:
      the baseline state, for computing the penalty for this transition
    """
    if self._exact:
      if prev_state in self._inaction_next:
        self._baseline_state = self.sample(prev_state)
      else:
        inaction_env = copy.deepcopy(self._env)
        timestep_inaction = inaction_env.step(Actions.NOOP)
        self._baseline_state = self._timestep_to_state(timestep_inaction)
        self._inaction_next[prev_state][self._baseline_state] += 1
      timestep_action = self._env.step(action)
      assert current_state == self._timestep_to_state(timestep_action)
    else:
      if action == Actions.NOOP:
        self._inaction_next[prev_state][current_state] += 1
      if prev_state in self._inaction_next:
        self._baseline_state = self.sample(prev_state)
      else:
        self._baseline_state = prev_state
    return self._baseline_state

  def _inaction_rollout(self, state):
    """Compute an (approximate) inaction rollout from a state."""
    chain = []
    st = state
    while st not in chain:
      chain.append(st)
      if st in self._inaction_next:
        st = self.sample(st)
    return chain

  def parallel_inaction_rollouts(self, s1, s2):
    """Compute (approximate) parallel inaction rollouts from two states."""
    chain = []
    states = (s1, s2)
    while states not in chain:
      chain.append(states)
      s1, s2 = states
      states = (self.sample(s1) if s1 in self._inaction_next else s1,
                self.sample(s2) if s2 in self._inaction_next else s2)
    return chain

  @property
  def rollout_func(self):
    return self._inaction_rollout if self._rollouts else None


@six.add_metaclass(abc.ABCMeta)
class DeviationMeasure(object):
  """Base class for deviation measures."""

  @abc.abstractmethod
  def calculate(self):
    """Calculate the deviation between two states."""

  @abc.abstractmethod
  def update(self):
    """Update any models after seeing a state transition."""


class ReachabilityMixin(object):
  """Class for computing reachability deviation measure.

     Computes the relative/un- reachability given a dictionary of
     reachability scores for pairs of states.

     Expects _reachability, _discount, and _dev_fun attributes to exist in the
     inheriting class.
  """

  def calculate(self, current_state, baseline_state, rollout_func=None):
    """Calculate relative/un- reachability between particular states."""
    # relative reachability case
    if self._dev_fun:
      if rollout_func:
        curr_values = self._rollout_values(rollout_func(current_state))
        base_values = self._rollout_values(rollout_func(baseline_state))
      else:
        curr_values = self._reachability[current_state]
        base_values = self._reachability[baseline_state]
      all_s = set(list(curr_values.keys()) + list(base_values.keys()))
      total = 0
      for s in all_s:
        diff = base_values[s] - curr_values[s]
        total += self._dev_fun(diff)
      d = total / len(all_s)
    # unreachability case
    else:
      assert rollout_func is None
      d = 1 - self._reachability[current_state][baseline_state]
    return d

  def _rollout_values(self, chain):
    """Compute stepwise rollout values for the relative reachability penalty.

    Args:
      chain: chain of states in an inaction rollout starting with the state for
        which to compute the rollout values

    Returns:
      a dictionary of the form:
        { s : (1-discount) sum_{k=0}^inf discount^k R_s(S_k) }
       where S_k is the k-th state in the inaction rollout from 'state',
       s is a state, and
       R_s(S_k) is the reachability of s from S_k.
    """
    rollout_values = collections.defaultdict(lambda: 0)
    coeff = 1
    for st in chain:
      for s, rch in six.iteritems(self._reachability[st]):
        rollout_values[s] += coeff * rch * (1.0 - self._discount)
      coeff *= self._discount
    last_state = chain[-1]
    for s, rch in six.iteritems(self._reachability[last_state]):
      rollout_values[s] += coeff * rch
    return rollout_values


class Reachability(ReachabilityMixin, DeviationMeasure):
  """Approximate (relative) (un)reachability deviation measure.

     Unreachability (the default, when `dev_fun=None`) uses the length (say, n)
     of the shortest path (sequence of actions) from the current state to the
     baseline state. The reachability score is value_discount ** n.
     Unreachability is then 1.0 - the reachability score.

     Relative reachability (when `dev_fun` is not `None`) considers instead the
     difference in reachability of all other states from the current state
     versus from the baseline state.

     We approximate reachability by only considering state transitions
     that have been observed. Add transitions using the `update` function.
  """

  def __init__(self, value_discount=1.0, dev_fun=None, discount=None):
    self._value_discount = value_discount
    self._dev_fun = dev_fun
    self._discount = discount
    self._reachability = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0))

  def update(self, prev_state, current_state):
    self._reachability[prev_state][prev_state] = 1
    self._reachability[current_state][current_state] = 1
    if self._reachability[prev_state][current_state] < self._value_discount:
      for s1 in self._reachability.keys():
        if self._reachability[s1][prev_state] > 0:
          for s2 in self._reachability[current_state].keys():
            if self._reachability[current_state][s2] > 0:
              self._reachability[s1][s2] = max(
                  self._reachability[s1][s2],
                  self._reachability[s1][prev_state] * self._value_discount *
                  self._reachability[current_state][s2])

  @property
  def discount(self):
    return self._discount


class AttainableUtilityMixin(object):
  """Class for computing attainable utility measure.

     Computes attainable utility (averaged over a set of utility functions)
     given value functions for each utility function.

     Expects _u_values, _discount, _value_discount, and _dev_fun attributes to
     exist in the inheriting class.
  """

  def calculate(self, current_state, baseline_state, rollout_func=None):
    if rollout_func:
      current_values = self._rollout_values(rollout_func(current_state))
      baseline_values = self._rollout_values(rollout_func(baseline_state))
    else:
      current_values = [u_val[current_state] for u_val in self._u_values]
      baseline_values = [u_val[baseline_state] for u_val in self._u_values]
    penalties = [self._dev_fun(base_val - cur_val) * (1. - self._value_discount)
                 for base_val, cur_val in zip(baseline_values, current_values)]
    return sum(penalties) / len(penalties)

  def _rollout_values(self, chain):
    """Compute stepwise rollout values for the attainable utility penalty.

    Args:
      chain: chain of states in an inaction rollout starting with the state
             for which to compute the rollout values

    Returns:
      a list containing
        (1-discount) sum_{k=0}^inf discount^k V_u(S_k)
      for each utility function u,
      where S_k is the k-th state in the inaction rollout from 'state'.
    """
    rollout_values = [0 for _ in self._u_values]
    coeff = 1
    for st in chain:
      rollout_values = [rv + coeff * u_val[st] * (1.0 - self._discount)
                        for rv, u_val in zip(rollout_values, self._u_values)]
      coeff *= self._discount
    last_state = chain[-1]
    rollout_values = [rv + coeff * u_val[last_state]
                      for rv, u_val in zip(rollout_values, self._u_values)]
    return rollout_values

  def _set_util_funs(self, util_funs):
    """Set up this instance's utility functions.

    Args:
      util_funs: either a number of functions to generate or a list of
                 pre-defined utility functions, represented as dictionaries
                 over states: util_funs[i][s] = u_i(s), the utility of s
                 according to u_i.
    """
    if isinstance(util_funs, int):
      self._util_funs = [
          collections.defaultdict(float) for _ in range(util_funs)
      ]
    else:
      self._util_funs = util_funs

  def _utility(self, u, state):
    """Apply a random utility function, generating its value if necessary."""
    if state not in u:
      u[state] = np.random.random()
    return u[state]


class AttainableUtility(AttainableUtilityMixin, DeviationMeasure):
  """Approximate attainable utility deviation measure."""

  def __init__(self, value_discount=0.99, dev_fun=np.abs, util_funs=10,
               discount=None):
    assert value_discount < 1.0  # AU does not converge otherwise
    self._value_discount = value_discount
    self._dev_fun = dev_fun
    self._discount = discount
    self._set_util_funs(util_funs)
    # u_values[i][s] = V_{u_i}(s), the (approximate) value of s according to u_i
    self._u_values = [
        collections.defaultdict(float) for _ in range(len(self._util_funs))
    ]
    # predecessors[s] = set of states known to lead, by some action, to s
    self._predecessors = collections.defaultdict(set)

  def update(self, prev_state, current_state):
    """Update predecessors and attainable utility estimates."""
    self._predecessors[current_state].add(prev_state)
    seen = set()
    queue = [current_state]
    while queue:
      s_to = queue.pop(0)
      seen.add(s_to)
      for u, u_val in zip(self._util_funs, self._u_values):
        for s_from in self._predecessors[s_to]:
          v = self._utility(u, s_from) + self._value_discount * u_val[s_to]
          if u_val[s_from] < v:
            u_val[s_from] = v
            if s_from not in seen:
              queue.append(s_from)


class NoDeviation(DeviationMeasure):
  """Dummy deviation measure corresponding to no impact penalty."""

  def calculate(self, *unused_args):
    return 0

  def update(self, *unused_args):
    pass


class SideEffectPenalty(object):
  """Impact penalty."""

  def __init__(self, baseline, dev_measure, beta=1.0,
               use_inseparable_rollout=False):
    """Make an object to calculate the impact penalty.

    Args:
      baseline: object for calculating the baseline state
      dev_measure: object for calculating the deviation between states
      beta: weight (scaling factor) for the impact penalty
      use_inseparable_rollout: whether to compute the penalty as the average of
        deviations over parallel inaction rollouts from the current and
        baselines states (True) otherwise just between the current state and
        baseline state (or by whatever rollout value is provided in the
        baseline) (False)
    """
    self._baseline = baseline
    self._dev_measure = dev_measure
    self._beta = beta
    self._use_inseparable_rollout = use_inseparable_rollout

  def calculate(self, prev_state, action, current_state):
    """Calculate the penalty associated with a transition, and update models."""
    if current_state:
      self._dev_measure.update(prev_state, current_state)
      baseline_state = self._baseline.calculate(prev_state, action,
                                                current_state)
      if self._use_inseparable_rollout:
        penalty = self._rollout_value(current_state, baseline_state,
                                      self._dev_measure.discount,
                                      self._dev_measure.calculate)
      else:
        penalty = self._dev_measure.calculate(current_state, baseline_state,
                                              self._baseline.rollout_func)
      return self._beta * penalty
    else:
      return 0

  def reset(self):
    """Signal start of new episode."""
    self._baseline.reset()

  def _rollout_value(self, cur_state, base_state, discount, func):
    """Compute stepwise rollout value for unreachability."""
    # Returns (1-discount) sum_{k=0}^inf discount^k R(S_{t,t+k}, S'_{t,t+k}),
    # where S_{t,t+k} is k-th state in the inaction rollout from current state,
    # S'_{t,t+k} is k-th state in the inaction rollout from baseline state,
    # and R is the reachability function.
    chain = self._baseline.parallel_inaction_rollouts(cur_state, base_state)
    coeff = 1
    rollout_value = 0
    for states in chain:
      rollout_value += (coeff * func(states[0], states[1]) * (1.0 - discount))
      coeff *= discount
    last_states = chain[-1]
    rollout_value += coeff * func(last_states[0], last_states[1])
    return rollout_value

  @property
  def beta(self):
    return self._beta
