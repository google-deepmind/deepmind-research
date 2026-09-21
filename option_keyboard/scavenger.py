# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Simple Scavenger environment."""

import copy
import enum
import sys

import dm_env

import numpy as np

from option_keyboard import auto_reset_environment

this_module = sys.modules[__name__]


class Action(enum.IntEnum):
  """Actions available to the player."""
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3


def _one_hot(indices, depth):
  return np.eye(depth)[indices]


def _random_pos(arena_size):
  return tuple(np.random.randint(0, arena_size, size=[2]).tolist())


class Scavenger(auto_reset_environment.Base):
  """Simple Scavenger."""

  def __init__(self,
               arena_size,
               num_channels,
               max_num_steps,
               default_w=None,
               num_init_objects=15,
               object_priors=None,
               egocentric=True,
               rewarder=None,
               aux_tasks_w=None):
    self._arena_size = arena_size
    self._num_channels = num_channels
    self._max_num_steps = max_num_steps
    self._num_init_objects = num_init_objects
    self._egocentric = egocentric
    self._rewarder = (
        getattr(this_module, rewarder)() if rewarder is not None else None)
    self._aux_tasks_w = aux_tasks_w

    if object_priors is None:
      self._object_priors = np.ones(num_channels) / num_channels
    else:
      assert len(object_priors) == num_channels
      self._object_priors = np.array(object_priors) / np.sum(object_priors)

    if default_w is None:
      self._default_w = np.ones(shape=(num_channels,))
    else:
      self._default_w = default_w

    self._num_channels_all = self._num_channels + 2
    self._step_in_episode = None

  @property
  def state(self):
    return copy.deepcopy([
        self._step_in_episode,
        self._walls,
        self._objects,
        self._player_pos,
        self._prev_collected,
    ])

  def set_state(self, state):
    state_ = copy.deepcopy(state)
    self._step_in_episode = state_[0]
    self._walls = state_[1]
    self._objects = state_[2]
    self._player_pos = state_[3]
    self._prev_collected = state_[4]

  @property
  def player_pos(self):
    return self._player_pos

  def _reset(self):
    self._step_in_episode = 0

    # Walls.
    self._walls = []
    for col in range(self._arena_size):
      new_pos = (0, col)
      if new_pos not in self._walls:
        self._walls.append(new_pos)
    for row in range(self._arena_size):
      new_pos = (row, 0)
      if new_pos not in self._walls:
        self._walls.append(new_pos)

    # Objects.
    self._objects = dict()
    for _ in range(self._num_init_objects):
      while True:
        new_pos = _random_pos(self._arena_size)
        if new_pos not in self._objects and new_pos not in self._walls:
          self._objects[new_pos] = np.random.multinomial(1, self._object_priors)
          break

    # Player
    self._player_pos = _random_pos(self._arena_size)
    while self._player_pos in self._objects or self._player_pos in self._walls:
      self._player_pos = _random_pos(self._arena_size)

    self._prev_collected = np.zeros(shape=(self._num_channels,))

    obs = self.observation()

    return dm_env.restart(obs)

  def _step(self, action):
    self._step_in_episode += 1

    if action == Action.UP:
      new_player_pos = (self._player_pos[0], self._player_pos[1] + 1)
    elif action == Action.DOWN:
      new_player_pos = (self._player_pos[0], self._player_pos[1] - 1)
    elif action == Action.LEFT:
      new_player_pos = (self._player_pos[0] - 1, self._player_pos[1])
    elif action == Action.RIGHT:
      new_player_pos = (self._player_pos[0] + 1, self._player_pos[1])
    else:
      raise ValueError("Invalid action `{}`".format(action))

    # Toroidal.
    new_player_pos = (
        (new_player_pos[0] + self._arena_size) % self._arena_size,
        (new_player_pos[1] + self._arena_size) % self._arena_size,
    )

    if new_player_pos not in self._walls:
      self._player_pos = new_player_pos

    # Compute rewards.
    consumed = self._objects.pop(self._player_pos,
                                 np.zeros(shape=(self._num_channels,)))
    if self._rewarder is None:
      reward = np.dot(consumed, np.array(self._default_w))
    else:
      reward = self._rewarder.get_reward(self.state, consumed)
    self._prev_collected = np.copy(consumed)

    assert self._player_pos not in self._objects
    assert self._player_pos not in self._walls

    # Render everything.
    obs = self.observation()

    if self._step_in_episode < self._max_num_steps:
      return dm_env.transition(reward=reward, observation=obs)
    else:
      # termination with discount=1.0
      return dm_env.truncation(reward=reward, observation=obs)

  def observation(self, force_non_egocentric=False):
    arena_shape = [self._arena_size] * 2 + [self._num_channels_all]
    arena = np.zeros(shape=arena_shape, dtype=np.float32)

    def offset_position(pos_):
      use_egocentric = self._egocentric and not force_non_egocentric
      offset = self._player_pos if use_egocentric else (0, 0)
      x = (pos_[0] - offset[0] + self._arena_size) % self._arena_size
      y = (pos_[1] - offset[1] + self._arena_size) % self._arena_size
      return (x, y)

    player_pos = offset_position(self._player_pos)
    arena[player_pos] = _one_hot(self._num_channels, self._num_channels_all)

    for pos, obj in self._objects.items():
      x, y = offset_position(pos)
      arena[x, y, :self._num_channels] = obj

    for pos in self._walls:
      x, y = offset_position(pos)
      arena[x, y] = _one_hot(self._num_channels + 1, self._num_channels_all)

    collected_resources = np.copy(self._prev_collected).astype(np.float32)

    obs = dict(
        arena=arena,
        cumulants=collected_resources,
    )
    if self._aux_tasks_w is not None:
      obs["aux_tasks_reward"] = np.dot(
          np.array(self._aux_tasks_w), self._prev_collected).astype(np.float32)

    return obs

  def observation_spec(self):
    arena = dm_env.specs.BoundedArray(
        shape=(self._arena_size, self._arena_size, self._num_channels_all),
        dtype=np.float32,
        minimum=0.,
        maximum=1.,
        name="arena")
    collected_resources = dm_env.specs.BoundedArray(
        shape=(self._num_channels,),
        dtype=np.float32,
        minimum=-1e9,
        maximum=1e9,
        name="collected_resources")

    obs_spec = dict(
        arena=arena,
        cumulants=collected_resources,
    )
    if self._aux_tasks_w is not None:
      obs_spec["aux_tasks_reward"] = dm_env.specs.BoundedArray(
          shape=(len(self._aux_tasks_w),),
          dtype=np.float32,
          minimum=-1e9,
          maximum=1e9,
          name="aux_tasks_reward")

    return obs_spec

  def action_spec(self):
    return dm_env.specs.DiscreteArray(num_values=len(Action), name="action")


class SequentialCollectionRewarder(object):
  """SequentialCollectionRewarder."""

  def get_reward(self, state, consumed):
    """Get reward."""

    object_counts = sum(list(state[2].values()) + [np.zeros(len(consumed))])

    reward = 0.0
    if np.sum(consumed) > 0:
      for i in range(len(consumed)):
        if np.all(object_counts[:i] <= object_counts[i]):
          reward += consumed[i]
        else:
          reward -= consumed[i]

    return reward


class BalancedCollectionRewarder(object):
  """BalancedCollectionRewarder."""

  def get_reward(self, state, consumed):
    """Get reward."""

    object_counts = sum(list(state[2].values()) + [np.zeros(len(consumed))])

    reward = 0.0
    if np.sum(consumed) > 0:
      for i in range(len(consumed)):
        if (object_counts[i] + consumed[i]) >= np.max(object_counts):
          reward += consumed[i]
        else:
          reward -= consumed[i]

    return reward
