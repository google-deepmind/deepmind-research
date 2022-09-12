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
"""Vanilla Q-Learning agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import abc
import numpy as np
from six.moves import range


class EpsilonGreedyPolicy(object):
  """Epsilon greedy policy for table value function lookup."""

  def __init__(self, value_function, actions):
    """Construct an epsilon greedy policy object.

    Args:
      value_function: agent value function as a dict.
      actions: list of possible actions.

    Raises:
      ValueError: if `actions` agument is not an iterable.
    """
    if not isinstance(actions, abc.Iterable):
      raise ValueError('`actions` argument must be an iterable.')

    self._value_function = value_function
    self._actions = actions

  def get_action(self, epsilon, state):
    """Get action following the e-greedy policy.

    Args:
      epsilon: probability of selecting a random action
      state: current state of the game as a state/action tuple.

    Returns:
      Chosen action.
    """
    if np.random.random() < epsilon:
      return np.random.choice(self._actions)
    else:
      values = [self._value_function[(state, action)]
                for action in self._actions]

      max_value = max(values)
      max_indices = [i for i, value in enumerate(values) if value == max_value]

      return self._actions[np.random.choice(max_indices)]


class QLearning(object):
  """Q-learning agent."""

  def __init__(self, actions, alpha=0.1, epsilon=0.1, q_initialisation=0.0,
               discount=0.99):
    """Create a Q-learning agent.

    Args:
      actions: a BoundedArraySpec that specifes full discrete action spec.
      alpha: agent learning rate.
      epsilon: agent exploration rate.
      q_initialisation: float, used to initialise the value function.
      discount: discount factor for rewards.
    """

    self._value_function = collections.defaultdict(lambda: q_initialisation)
    self._valid_actions = list(range(actions.minimum, actions.maximum + 1))
    self._policy = EpsilonGreedyPolicy(self._value_function,
                                       self._valid_actions)

    # Hyperparameters.
    self.alpha = alpha
    self.epsilon = epsilon
    self.discount = discount

    # Episode internal variables.
    self._current_action = None
    self._current_state = None

  def begin_episode(self):
    """Perform episode initialisation."""
    self._current_state = None
    self._current_action = None

  def _timestep_to_state(self, timestep):
    return tuple(map(tuple, np.copy(timestep.observation['board'])))

  def step(self, timestep):
    """Perform a single step in the environment."""
    # Get state observations.
    state = self._timestep_to_state(timestep)

    # This is one of the follow up states (i.e. not the initial state).
    if self._current_state is not None:
      self._update(timestep, state)

    self._current_state = state
    # Determine action.
    self._current_action = self._policy.get_action(self.epsilon, state)
    # Emit action.
    return self._current_action

  def _calculate_reward(self, timestep, unused_state):
    """Calculate reward: to be extended when impact penalty is added."""
    reward = timestep.reward
    return reward

  def _update(self, timestep, state):
    """Perform value function update."""

    reward = self._calculate_reward(timestep, state)

    # Terminal state.
    if not state:
      delta = (reward - self._value_function[(self._current_state,
                                              self._current_action)])
    # Non-terminal state.
    else:
      max_action = self._policy.get_action(0, state)
      delta = (
          reward + self.discount * self._value_function[(state, max_action)] -
          self._value_function[(self._current_state, self._current_action)])

    self._value_function[(self._current_state,
                          self._current_action)] += self.alpha * delta

  def end_episode(self, timestep):
    """Performs episode cleanup."""
    # Update for the terminal state.
    self._update(timestep, None)

  @property
  def value_function(self):
    return self._value_function
