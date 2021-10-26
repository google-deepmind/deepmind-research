# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Components for DQN."""

import abc
import collections
import csv
import os
import timeit
from typing import Any, Iterable, Mapping, Optional, Text, Tuple, Union

import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import rlax

from tandem_dqn import networks
from tandem_dqn import processors

Action = int
Network = networks.Network
NetworkParams = networks.Params
PRNGKey = jnp.ndarray  # A size 2 array.


class Agent(abc.ABC):
  """Agent interface."""

  @abc.abstractmethod
  def step(self, timestep: dm_env.TimeStep) -> Action:
    """Selects action given timestep and potentially learns."""

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """

  @abc.abstractmethod
  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves agent state as a dictionary (e.g. for serialization)."""

  @abc.abstractmethod
  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets agent state from a (potentially de-serialized) dictionary."""

  @property
  @abc.abstractmethod
  def statistics(self) -> Mapping[Text, float]:
    """Returns current agent statistics as a dictionary."""


def run_loop(
    agent: Agent,
    environment: dm_env.Environment,
    max_steps_per_episode: int = 0,
    yield_before_reset: bool = False,
) -> Iterable[Tuple[dm_env.Environment, Optional[dm_env.TimeStep], Agent,
                    Optional[Action]]]:
  """Repeatedly alternates step calls on environment and agent.

  At time `t`, `t + 1` environment timesteps and `t + 1` agent steps have been
  seen in the current episode. `t` resets to `0` for the next episode.

  Args:
    agent: Agent to be run, has methods `step(timestep)` and `reset()`.
    environment: Environment to run, has methods `step(action)` and `reset()`.
    max_steps_per_episode: If positive, when time t reaches this value within an
      episode, the episode is truncated.
    yield_before_reset: Whether to additionally yield `(environment, None,
      agent, None)` before the agent and environment is reset at the start of
      each episode.

  Yields:
    Tuple `(environment, timestep_t, agent, a_t)` where
    `a_t = agent.step(timestep_t)`.
  """
  while True:  # For each episode.
    if yield_before_reset:
      yield environment, None, agent, None,

    t = 0
    agent.reset()
    timestep_t = environment.reset()  # timestep_0.

    while True:  # For each step in the current episode.
      a_t = agent.step(timestep_t)
      yield environment, timestep_t, agent, a_t

      # Update t after one environment step and agent step and relabel.
      t += 1
      a_tm1 = a_t
      timestep_t = environment.step(a_tm1)

      if max_steps_per_episode > 0 and t >= max_steps_per_episode:
        assert t == max_steps_per_episode
        timestep_t = timestep_t._replace(step_type=dm_env.StepType.LAST)

      if timestep_t.last():
        unused_a_t = agent.step(timestep_t)  # Extra agent step, action ignored.
        yield environment, timestep_t, agent, None
        break


def generate_statistics(
    trackers: Iterable[Any],
    timestep_action_sequence: Iterable[Tuple[dm_env.Environment,
                                             Optional[dm_env.TimeStep], Agent,
                                             Optional[Action]]]
) -> Mapping[Text, Any]:
  """Generates statistics from a sequence of timestep and actions."""
  # Only reset at the start, not between episodes.
  for tracker in trackers:
    tracker.reset()

  for environment, timestep_t, agent, a_t in timestep_action_sequence:
    for tracker in trackers:
      tracker.step(environment, timestep_t, agent, a_t)

  # Merge all statistics dictionaries into one.
  statistics_dicts = (tracker.get() for tracker in trackers)
  return dict(collections.ChainMap(*statistics_dicts))


class EpisodeTracker:
  """Tracks episode return and other statistics."""

  def __init__(self):
    self._num_steps_since_reset = None
    self._num_steps_over_episodes = None
    self._episode_returns = None
    self._current_episode_rewards = None
    self._current_episode_step = None

  def step(
      self,
      environment: Optional[dm_env.Environment],
      timestep_t: dm_env.TimeStep,
      agent: Optional[Agent],
      a_t: Optional[Action],
  ) -> None:
    """Accumulates statistics from timestep."""
    del (environment, agent, a_t)

    if timestep_t.first():
      if self._current_episode_rewards:
        raise ValueError('Current episode reward list should be empty.')
      if self._current_episode_step != 0:
        raise ValueError('Current episode step should be zero.')
    else:
      # First reward is invalid, all other rewards are appended.
      self._current_episode_rewards.append(timestep_t.reward)

    self._num_steps_since_reset += 1
    self._current_episode_step += 1

    if timestep_t.last():
      self._episode_returns.append(sum(self._current_episode_rewards))
      self._current_episode_rewards = []
      self._num_steps_over_episodes += self._current_episode_step
      self._current_episode_step = 0

  def reset(self) -> None:
    """Resets all gathered statistics, not to be called between episodes."""
    self._num_steps_since_reset = 0
    self._num_steps_over_episodes = 0
    self._episode_returns = []
    self._current_episode_step = 0
    self._current_episode_rewards = []

  def get(self) -> Mapping[Text, Union[int, float, None]]:
    """Aggregates statistics and returns as a dictionary.

    Here the convention is `episode_return` is set to `current_episode_return`
    if a full episode has not been encountered. Otherwise it is set to
    `mean_episode_return` which is the mean return of complete episodes only. If
    no steps have been taken at all, `episode_return` is set to `NaN`.

    Returns:
      A dictionary of aggregated statistics.
    """
    if self._episode_returns:
      mean_episode_return = np.array(self._episode_returns).mean()
      current_episode_return = sum(self._current_episode_rewards)
      episode_return = mean_episode_return
    else:
      mean_episode_return = np.nan
      if self._num_steps_since_reset > 0:
        current_episode_return = sum(self._current_episode_rewards)
      else:
        current_episode_return = np.nan
      episode_return = current_episode_return

    return {
        'mean_episode_return': mean_episode_return,
        'current_episode_return': current_episode_return,
        'episode_return': episode_return,
        'num_episodes': len(self._episode_returns),
        'num_steps_over_episodes': self._num_steps_over_episodes,
        'current_episode_step': self._current_episode_step,
        'num_steps_since_reset': self._num_steps_since_reset,
    }


class StepRateTracker:
  """Tracks step rate, number of steps taken and duration since last reset."""

  def __init__(self):
    self._num_steps_since_reset = None
    self._start = None

  def step(
      self,
      environment: Optional[dm_env.Environment],
      timestep_t: Optional[dm_env.TimeStep],
      agent: Optional[Agent],
      a_t: Optional[Action],
  ) -> None:
    del (environment, timestep_t, agent, a_t)
    self._num_steps_since_reset += 1

  def reset(self) -> None:
    self._num_steps_since_reset = 0
    self._start = timeit.default_timer()

  def get(self) -> Mapping[Text, float]:
    duration = timeit.default_timer() - self._start
    if self._num_steps_since_reset > 0:
      step_rate = self._num_steps_since_reset / duration
    else:
      step_rate = np.nan
    return {
        'step_rate': step_rate,
        'num_steps': self._num_steps_since_reset,
        'duration': duration,
    }


class UnbiasedExponentialWeightedAverageAgentTracker:
  """'Unbiased Constant-Step-Size Trick' from the Sutton and Barto RL book."""

  def __init__(self, step_size: float, initial_agent: Agent):
    self._initial_statistics = dict(initial_agent.statistics)
    self._step_size = step_size
    self.trace = 0.
    self._statistics = dict(self._initial_statistics)

  def step(
      self,
      environment: Optional[dm_env.Environment],
      timestep_t: Optional[dm_env.TimeStep],
      agent: Agent,
      a_t: Optional[Action],
  ) -> None:
    """Accumulates agent statistics."""
    del (environment, timestep_t, a_t)

    self.trace = (1 - self._step_size) * self.trace + self._step_size
    final_step_size = self._step_size / self.trace
    assert 0 <= final_step_size <= 1

    if final_step_size == 1:
      # Since the self._initial_statistics is likely to be NaN and
      # 0 * NaN == NaN just replace self._statistics on the first step.
      self._statistics = dict(agent.statistics)
    else:
      self._statistics = jax.tree_multimap(
          lambda s, x: (1 - final_step_size) * s + final_step_size * x,
          self._statistics, agent.statistics)

  def reset(self) -> None:
    """Resets statistics and internal state."""
    self.trace = 0.
    # get() may be called before step() so ensure statistics are initialized.
    self._statistics = dict(self._initial_statistics)

  def get(self) -> Mapping[Text, float]:
    """Returns current accumulated statistics."""
    return self._statistics


def make_default_trackers(initial_agent: Agent):
  return [
      EpisodeTracker(),
      StepRateTracker(),
      UnbiasedExponentialWeightedAverageAgentTracker(
          step_size=1e-3, initial_agent=initial_agent),
  ]


class EpsilonGreedyActor(Agent):
  """Agent that acts with a given set of Q-network parameters and epsilon.

  Network parameters are set on the actor. The actor can be serialized,
  ensuring determinism of execution (e.g. when checkpointing).
  """

  def __init__(
      self,
      preprocessor: processors.Processor,
      network: Network,
      exploration_epsilon: float,
      rng_key: PRNGKey,
  ):
    self._preprocessor = preprocessor
    self._rng_key = rng_key
    self._action = None
    self.network_params = None  # Nest of arrays (haiku.Params), set externally.

    def select_action(rng_key, network_params, s_t):
      """Samples action from eps-greedy policy wrt Q-values at given state."""
      rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
      q_t = network.apply(network_params, apply_key, s_t[None, ...]).q_values[0]
      a_t = rlax.epsilon_greedy().sample(policy_key, q_t, exploration_epsilon)
      return rng_key, a_t

    self._select_action = jax.jit(select_action)

  def step(self, timestep: dm_env.TimeStep) -> Action:
    """Selects action given a timestep."""
    timestep = self._preprocessor(timestep)

    if timestep is None:  # Repeat action.
      return self._action

    s_t = timestep.observation
    self._rng_key, a_t = self._select_action(self._rng_key, self.network_params,
                                             s_t)
    self._action = Action(jax.device_get(a_t))
    return self._action

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    processors.reset(self._preprocessor)
    self._action = None

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves agent state as a dictionary (e.g. for serialization)."""
    # State contains network params to make agent easy to run from a checkpoint.
    return {
        'rng_key': self._rng_key,
        'network_params': self.network_params,
    }

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets agent state from a (potentially de-serialized) dictionary."""
    self._rng_key = state['rng_key']
    self.network_params = state['network_params']

  @property
  def statistics(self) -> Mapping[Text, float]:
    return {}


class LinearSchedule:
  """Linear schedule, used for exploration epsilon in DQN agents."""

  def __init__(self,
               begin_value,
               end_value,
               begin_t,
               end_t=None,
               decay_steps=None):
    if (end_t is None) == (decay_steps is None):
      raise ValueError('Exactly one of end_t, decay_steps must be provided.')
    self._decay_steps = decay_steps if end_t is None else end_t - begin_t
    self._begin_t = begin_t
    self._begin_value = begin_value
    self._end_value = end_value

  def __call__(self, t):
    """Implements a linear transition from a begin to an end value."""
    frac = min(max(t - self._begin_t, 0), self._decay_steps) / self._decay_steps
    return (1 - frac) * self._begin_value + frac * self._end_value


class NullWriter:
  """A placeholder logging object that does nothing."""

  def write(self, *args, **kwargs) -> None:
    pass

  def close(self) -> None:
    pass


class CsvWriter:
  """A logging object writing to a CSV file.

  Each `write()` takes a `OrderedDict`, creating one column in the CSV file for
  each dictionary key on the first call. Successive calls to `write()` must
  contain the same dictionary keys.
  """

  def __init__(self, fname: Text):
    """Initializes a `CsvWriter`.

    Args:
      fname: File name (path) for file to be written to.
    """
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    self._fname = fname
    self._header_written = False
    self._fieldnames = None

  def write(self, values: collections.OrderedDict) -> None:
    """Appends given values as new row to CSV file."""
    if self._fieldnames is None:
      self._fieldnames = values.keys()
    # Open a file in 'append' mode, so we can continue logging safely to the
    # same file after e.g. restarting from a checkpoint.
    with open(self._fname, 'a') as file:
      # Always use same fieldnames to create writer, this way a consistency
      # check is performed automatically on each write.
      writer = csv.DictWriter(file, fieldnames=self._fieldnames)
      # Write a header if this is the very first write.
      if not self._header_written:
        writer.writeheader()
        self._header_written = True
      writer.writerow(values)

  def close(self) -> None:
    """Closes the `CsvWriter`."""
    pass

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves `CsvWriter` state as a `dict` (e.g. for serialization)."""
    return {
        'header_written': self._header_written,
        'fieldnames': self._fieldnames
    }

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets `CsvWriter` state from a (potentially de-serialized) dictionary."""
    self._header_written = state['header_written']
    self._fieldnames = state['fieldnames']


class NullCheckpoint:
  """A placeholder checkpointing object that does nothing.

  Can be used as a substitute for an actual checkpointing object when
  checkpointing is disabled.
  """

  def __init__(self):
    self.state = AttributeDict()

  def save(self) -> None:
    pass

  def can_be_restored(self) -> bool:
    return False

  def restore(self) -> None:
    pass


class AttributeDict(dict):
  """A `dict` that supports getting, setting, deleting keys via attributes."""

  def __getattr__(self, key):
    return self[key]

  def __setattr__(self, key, value):
    self[key] = value

  def __delattr__(self, key):
    del self[key]
