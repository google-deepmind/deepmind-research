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
import random
import numpy as np
import six
from six.moves import range
from six.moves import zip
import sonnet as snt
import tensorflow.compat.v1 as tf


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

  @property
  def baseline_state(self):
    return self._baseline_state


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

  def update(self, prev_state, current_state, action=None):
    del action  # Unused.
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


class UVFAReachability(ReachabilityMixin, DeviationMeasure):
  """Approximate relative reachability deviation measure using UVFA.

     We approximate reachability using a neural network only trained on state
     transitions that have been observed. For each (s0, action, s1) transition,
     we update the reachability estimate for (s0, action, s) towards the
     reachability estimate between s1 and s, for each s in a random sample of
     size update_sample_size. In particular, the loss for the neural network
     reachability estimate (NN) is
       sum_s(max_a(NN(s1, a, s)) * value_discount - NN(s0, action, s)),
     where the sum is over all sampled s, the max is taken over all actions a.

     At evaluation time, the reachability difference is calculated with respect
     to a randomly sampled set of states of size calc_sample_size.
  """

  def __init__(
      self,
      value_discount=0.95,
      dev_fun=None,
      discount=0.95,
      state_size=36,  # Sokoban default
      num_actions=5,
      update_sample_size=10,
      calc_sample_size=10,
      hidden_size=50,
      representation_size=5,
      num_layers=1,
      base_loss_coeff=0.1,
      num_stored=100):

    # Create networks to generate state representations. To get a reachability
    # estimate, take the dot product of the origin network output and the goal
    # network output, then pass it through a sigmoid function to constrain it to
    # between 0 and 1.
    output_sizes = [hidden_size] * num_layers + [representation_size]
    self._origin_network = snt.nets.MLP(
        output_sizes=output_sizes,
        activation=tf.nn.relu,
        activate_final=False,
        name='origin_network')
    self._goal_network = snt.nets.MLP(
        output_sizes=output_sizes,
        activation=tf.nn.relu,
        activate_final=False,
        name='goal_network')

    self._value_discount = value_discount
    self._dev_fun = dev_fun
    self._discount = discount
    self._state_size = state_size
    self._num_actions = num_actions
    self._update_sample_size = update_sample_size
    self._calc_sample_size = calc_sample_size
    self._num_stored = num_stored
    self._stored_states = set()

    self._state_0_placeholder = tf.placeholder(tf.float32, shape=(state_size))
    self._state_1_placeholder = tf.placeholder(tf.float32, shape=(state_size))
    self._action_placeholder = tf.placeholder(tf.float32, shape=(num_actions))
    self._update_sample_placeholder = tf.placeholder(
        tf.float32, shape=(update_sample_size, state_size))
    self._calc_sample_placeholder = tf.placeholder(
        tf.float32, shape=(calc_sample_size, state_size))

    # Trained to estimate reachability = value_discount ^ distance.
    self._sample_loss = self._get_state_action_loss(
        self._state_0_placeholder,
        self._state_1_placeholder,
        self._action_placeholder,
        self._update_sample_placeholder)

    # Add additional loss to force observed transitions towards value_discount.
    self._base_reachability = self._get_state_sample_reachability(
        self._state_0_placeholder,
        tf.expand_dims(self._state_1_placeholder, axis=0),
        action=self._action_placeholder)
    self._base_case_loss = tf.keras.losses.MSE(self._value_discount,
                                               self._base_reachability)

    self._opt = tf.train.AdamOptimizer().minimize(self._sample_loss +
                                                  base_loss_coeff *
                                                  self._base_case_loss)

    current_state_reachability = self._get_state_sample_reachability(
        self._state_0_placeholder, self._calc_sample_placeholder)
    baseline_state_reachability = self._get_state_sample_reachability(
        self._state_1_placeholder, self._calc_sample_placeholder)

    self._reachability_calculation = [
        tf.reshape(baseline_state_reachability, [-1]),
        tf.reshape(current_state_reachability, [-1])
    ]

    init = tf.global_variables_initializer()
    self._sess = tf.Session()
    self._sess.run(init)

  def calculate(self, current_state, baseline_state, rollout_func=None):
    """Compute the reachability penalty between two states."""
    current_state = np.array(current_state).flatten()
    baseline_state = np.array(baseline_state).flatten()
    sample = self._sample_n_states(self._calc_sample_size)
    # Run if there are enough states to draw a correctly-sized sample from.
    if sample:
      base, curr = self._sess.run(
          self._reachability_calculation,
          feed_dict={
              self._state_0_placeholder: current_state,
              self._state_1_placeholder: baseline_state,
              self._calc_sample_placeholder: sample
          })
      return sum(map(self._dev_fun, base - curr)) / self._calc_sample_size
    else:
      return 0

  def _sample_n_states(self, n):
    try:
      return random.sample(self._stored_states, n)
    except ValueError:
      return None

  def update(self, prev_state, current_state, action):
    prev_state = np.array(prev_state).flatten()
    current_state = np.array(current_state).flatten()
    one_hot_action = np.zeros(self._num_actions)
    one_hot_action[action] = 1

    sample = self._sample_n_states(self._update_sample_size)
    if self._num_stored is None or len(self._stored_states) < self._num_stored:
      self._stored_states.add(tuple(prev_state))
      self._stored_states.add(tuple(current_state))
    elif (np.random.random() < 0.01 and
          tuple(current_state) not in self._stored_states):
      self._stored_states.pop()
      self._stored_states.add(tuple(current_state))

    # If there aren't enough states to get a full sample, do nothing.
    if sample:
      self._sess.run([self._opt], feed_dict={
          self._state_0_placeholder: prev_state,
          self._state_1_placeholder: current_state,
          self._action_placeholder: one_hot_action,
          self._update_sample_placeholder: sample
      })

  def _get_state_action_loss(self, prev_state, current_state, action, sample):
    """Get the loss from differences in state reachability estimates."""
    # Calculate NN(s0, action, s) for all s in sample.
    prev_state_reachability = self._get_state_sample_reachability(
        prev_state, sample, action=action)
    # Calculate max_a(NN(s1, a, s)) for all s in sample and all actions a.
    current_state_reachability = tf.stop_gradient(
        self._get_state_sample_reachability(current_state, sample))
    # Combine to return loss.
    return tf.keras.losses.MSE(
        current_state_reachability * self._value_discount,
        prev_state_reachability)

  def _get_state_sample_reachability(self, state, sample, action=None):
    """Calculate reachability from a state to each item in a sample."""
    if action is None:
      state_options = self._tile_with_all_actions(state)
    else:
      state_options = tf.expand_dims(tf.concat([state, action], axis=0), axis=0)
    goal_representations = self._goal_network(sample)
    # Reachability of sampled states by taking actions
    reach_result = tf.sigmoid(
        tf.reduce_max(
            tf.matmul(
                goal_representations,
                self._origin_network(state_options),
                transpose_b=True),
            axis=1))
    if action is None:
      # Return 1 if sampled state is already reached (equal to state)
      reach_no_action = tf.cast(tf.reduce_all(tf.equal(sample, state), axis=1),
                                dtype=tf.float32)
      reach_result = tf.maximum(reach_result, reach_no_action)
    return reach_result

  def _tile_with_all_actions(self, state):
    """Returns tensor with all state/action combinations."""
    state_tiled = tf.tile(tf.expand_dims(state, axis=0), [self._num_actions, 1])
    all_actions_tiled = tf.one_hot(
        tf.range(self._num_actions), depth=self._num_actions)
    return tf.concat([state_tiled, all_actions_tiled], axis=1)


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

  def update(self, prev_state, current_state, action=None):
    """Update predecessors and attainable utility estimates."""
    del action  # Unused.
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

  def __init__(
      self, baseline, dev_measure, beta=1.0, nonterminal_weight=0.01,
      use_inseparable_rollout=False):
    """Make an object to calculate the impact penalty.

    Args:
      baseline: object for calculating the baseline state
      dev_measure: object for calculating the deviation between states
      beta: weight (scaling factor) for the impact penalty
      nonterminal_weight: penalty weight on nonterminal states.
      use_inseparable_rollout:
        whether to compute the penalty as the average of deviations over
        parallel inaction rollouts from the current and baseline states (True)
        otherwise just between the current state and baseline state (or by
        whatever rollout value is provided in the baseline) (False)
    """
    self._baseline = baseline
    self._dev_measure = dev_measure
    self._beta = beta
    self._nonterminal_weight = nonterminal_weight
    self._use_inseparable_rollout = use_inseparable_rollout

  def calculate(self, prev_state, action, current_state):
    """Calculate the penalty associated with a transition, and update models."""
    def compute_penalty(current_state, baseline_state):
      """Compute penalty."""
      if self._use_inseparable_rollout:
        penalty = self._rollout_value(current_state, baseline_state,
                                      self._dev_measure.discount,
                                      self._dev_measure.calculate)
      else:
        penalty = self._dev_measure.calculate(current_state, baseline_state,
                                              self._baseline.rollout_func)
      return self._beta * penalty
    if current_state:  # not a terminal state
      self._dev_measure.update(prev_state, current_state, action)
      baseline_state =\
          self._baseline.calculate(prev_state, action, current_state)
      penalty = compute_penalty(current_state, baseline_state)
      return self._nonterminal_weight * penalty
    else:  # terminal state
      penalty = compute_penalty(prev_state, self._baseline.baseline_state)
      return penalty

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
