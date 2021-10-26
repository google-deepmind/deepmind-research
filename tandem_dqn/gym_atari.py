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
"""dm_env environment wrapper around Gym Atari configured to be like Xitari.

Gym Atari is built on the Arcade Learning Environment (ALE), whereas Xitari is
an old fork of the ALE.
"""

# pylint: disable=g-bad-import-order

from typing import Optional, Tuple

import atari_py  # pylint: disable=unused-import for gym to load Atari games.
import dm_env
from dm_env import specs
import gym
import numpy as np

from tandem_dqn import atari_data

_GYM_ID_SUFFIX = '-xitari-v1'
_SA_SUFFIX = '-sa'


def _game_id(game, sticky_actions):
  return game + (_SA_SUFFIX if sticky_actions else '') + _GYM_ID_SUFFIX


def _register_atari_environments():
  """Registers Atari environments in Gym to be as similar to Xitari as possible.

  Main difference from PongNoFrameSkip-v4, etc. is max_episode_steps is unset
  and only the usual 57 Atari games are registered.

  Additionally, sticky-actions variants of the environments are registered
  with an '-sa' suffix.
  """
  for sticky_actions in [False, True]:
    for game in atari_data.ATARI_GAMES:
      repeat_action_probability = 0.25 if sticky_actions else 0.0
      gym.envs.registration.register(
          id=_game_id(game, sticky_actions),
          entry_point='gym.envs.atari:AtariEnv',
          kwargs={  # Explicitly set all known arguments.
              'game': game,
              'mode': None,  # Not necessarily the same as 0.
              'difficulty': None,  # Not necessarily the same as 0.
              'obs_type': 'image',
              'frameskip': 1,  # Get every frame.
              'repeat_action_probability': repeat_action_probability,
              'full_action_space': False,
          },
          max_episode_steps=None,  # No time limit, handled in run loop.
          nondeterministic=False,  # Xitari is deterministic.
      )


_register_atari_environments()


class GymAtari(dm_env.Environment):
  """Gym Atari with a `dm_env.Environment` interface."""

  def __init__(self, game, sticky_actions, seed):
    self._gym_env = gym.make(_game_id(game, sticky_actions))
    self._gym_env.seed(seed)
    self._start_of_episode = True

  def reset(self) -> dm_env.TimeStep:
    """Resets the environment and starts a new episode."""
    observation = self._gym_env.reset()
    lives = np.int32(self._gym_env.ale.lives())
    timestep = dm_env.restart((observation, lives))
    self._start_of_episode = False
    return timestep

  def step(self, action: np.int32) -> dm_env.TimeStep:
    """Updates the environment given an action and returns a timestep."""
    # If the previous timestep was LAST then we call reset() on the Gym
    # environment, otherwise step(). Although Gym environments allow you to step
    # through episode boundaries (similar to dm_env) they emit a warning.
    if self._start_of_episode:
      step_type = dm_env.StepType.FIRST
      observation = self._gym_env.reset()
      discount = None
      reward = None
      done = False
    else:
      observation, reward, done, info = self._gym_env.step(action)
      if done:
        assert 'TimeLimit.truncated' not in info, 'Should never truncate.'
        step_type = dm_env.StepType.LAST
        discount = 0.
      else:
        step_type = dm_env.StepType.MID
        discount = 1.

    lives = np.int32(self._gym_env.ale.lives())
    timestep = dm_env.TimeStep(
        step_type=step_type,
        observation=(observation, lives),
        reward=reward,
        discount=discount,
    )
    self._start_of_episode = done
    return timestep

  def observation_spec(self) -> Tuple[specs.Array, specs.Array]:
    space = self._gym_env.observation_space
    return (specs.Array(shape=space.shape, dtype=space.dtype, name='rgb'),
            specs.Array(shape=(), dtype=np.int32, name='lives'))

  def action_spec(self) -> specs.DiscreteArray:
    space = self._gym_env.action_space
    return specs.DiscreteArray(
        num_values=space.n, dtype=np.int32, name='action')

  def close(self):
    self._gym_env.close()


class RandomNoopsEnvironmentWrapper(dm_env.Environment):
  """Adds a random number of noop actions at the beginning of each episode."""

  def __init__(self,
               environment: dm_env.Environment,
               max_noop_steps: int,
               min_noop_steps: int = 0,
               noop_action: int = 0,
               seed: Optional[int] = None):
    """Initializes the random noops environment wrapper."""
    self._environment = environment
    if max_noop_steps < min_noop_steps:
      raise ValueError('max_noop_steps must be greater or equal min_noop_steps')
    self._min_noop_steps = min_noop_steps
    self._max_noop_steps = max_noop_steps
    self._noop_action = noop_action
    self._rng = np.random.RandomState(seed)

  def reset(self):
    """Begins new episode.

    This method resets the wrapped environment and applies a random number
    of noop actions before returning the last resulting observation
    as the first episode timestep. Intermediate timesteps emitted by the inner
    environment (including all rewards and discounts) are discarded.

    Returns:
      First episode timestep corresponding to the timestep after a random number
      of noop actions are applied to the inner environment.

    Raises:
      RuntimeError: if an episode end occurs while the inner environment
        is being stepped through with the noop action.
    """
    return self._apply_random_noops(initial_timestep=self._environment.reset())

  def step(self, action):
    """Steps environment given action.

    If beginning a new episode then random noops are applied as in `reset()`.

    Args:
      action: action to pass to environment conforming to action spec.

    Returns:
      `Timestep` from the inner environment unless beginning a new episode, in
      which case this is the timestep after a random number of noop actions
      are applied to the inner environment.
    """
    timestep = self._environment.step(action)
    if timestep.first():
      return self._apply_random_noops(initial_timestep=timestep)
    else:
      return timestep

  def _apply_random_noops(self, initial_timestep):
    assert initial_timestep.first()
    num_steps = self._rng.randint(self._min_noop_steps,
                                  self._max_noop_steps + 1)
    timestep = initial_timestep
    for _ in range(num_steps):
      timestep = self._environment.step(self._noop_action)
      if timestep.last():
        raise RuntimeError('Episode ended while applying %s noop actions.' %
                           num_steps)

    # We make sure to return a FIRST timestep, i.e. discard rewards & discounts.
    return dm_env.restart(timestep.observation)

  ## All methods except for reset and step redirect to the underlying env.

  def observation_spec(self):
    return self._environment.observation_spec()

  def action_spec(self):
    return self._environment.action_spec()

  def reward_spec(self):
    return self._environment.reward_spec()

  def discount_spec(self):
    return self._environment.discount_spec()

  def close(self):
    return self._environment.close()
