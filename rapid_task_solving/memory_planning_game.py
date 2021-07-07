# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory & Planning Game environment."""
import string

import dm_env
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class MemoryPlanningGame(dm_env.Environment):
  """Memory & Planning Game environment."""

  ACTION_NAMES = ['Up', 'Down', 'Left', 'Right', 'Collect']
  NUM_ACTIONS = len(ACTION_NAMES)
  DIRECTIONS = [
      (0, 1),   # Up
      (0, -1),  # Down
      (-1, 0),  # Left
      (1, 0),   # Right
      (0, 0),   # Collect
  ]

  def __init__(self,
               maze_size=4,
               max_episode_steps=100,
               target_reward=1.,
               per_step_reward=0.,
               random_respawn=False,
               seed=None):
    """The Memory & Planning Game environment.

    Args:
      maze_size: (int) size of the maze dimension.
      max_episode_steps: (int) number of steps per episode.
      target_reward: (float) reward value of the target.
      per_step_reward: (float) reward/cost of taking a step.
      random_respawn: (bool) whether the agent respawns in a random location
        upon collecting the goal.
      seed: (int or None) seed for random number generator.
    """
    self._maze_size = maze_size
    self._num_labels = maze_size * maze_size
    # The graph itself is the same across episodes, but the node labels will be
    # randomly sampled in each episode.
    self._graph = nx.grid_2d_graph(
        self._maze_size, self._maze_size, periodic=True)
    self._max_episode_steps = max_episode_steps
    self._target_reward = target_reward
    self._per_step_reward = per_step_reward
    self._random_respawn = random_respawn
    self._rng = np.random.RandomState(seed)

  def _one_hot(self, node):
    one_hot_vector = np.zeros([self._num_labels], dtype=np.int32)
    one_hot_vector[self._labels[node]] = 1
    return one_hot_vector

  def step(self, action):
    # If previous step was the last step of an episode, reset.
    if self._needs_reset:
      return self.reset()

    # Increment step count and check if it's the last step of the episode.
    self._episode_steps += 1
    if self._episode_steps >= self._max_episode_steps:
      self._needs_reset = True
      transition = dm_env.termination
    else:
      transition = dm_env.transition

    # Recompute agent's position given the selected action.
    direction = self.DIRECTIONS[action]
    self._position = tuple(
        (np.array(self._position) + np.array(direction)) % self._maze_size)
    self._previous_action = self.ACTION_NAMES[action]

    # Get reward if agent is over the goal location and the selected action is
    # `collect`.
    if self._position == self._goal and self.ACTION_NAMES[action] == 'Collect':
      reward = self._target_reward
      self._set_new_goal()
    else:
      reward = self._per_step_reward
    self._episode_reward += reward

    return transition(reward, self._observation())

  def _observation(self):
    return {
        'position': np.array(self._one_hot(self.position), dtype=np.int32),
        'goal': np.array(self._one_hot(self.goal), dtype=np.int32),
    }

  def observation_spec(self):
    return {
        'position': dm_env.specs.Array(
            shape=(self._num_labels,), dtype=np.int32, name='position'),
        'goal': dm_env.specs.Array(
            shape=(self._num_labels,), dtype=np.int32, name='goal'),
    }

  def action_spec(self):
    return dm_env.specs.DiscreteArray(self.NUM_ACTIONS)

  def take_random_action(self):
    return self.step(self._rng.randint(self.NUM_ACTIONS))

  def reset(self):
    self._previous_action = ''
    self._episode_reward = 0.
    self._episode_steps = 0
    self._needs_reset = False
    random_labels = self._rng.permutation(self._num_labels)
    self._labels = {n: random_labels[i]
                    for i, n in enumerate(self._graph.nodes())}
    self._respawn()
    self._set_new_goal()
    return dm_env.restart(self._observation())

  def _respawn(self):
    random_idx = self._rng.randint(self._num_labels)
    self._position = list(self._graph.nodes())[random_idx]

  def _set_new_goal(self):
    if self._random_respawn:
      self._respawn()
    goal = self._position
    while goal == self._position:
      random_idx = self._rng.randint(self._num_labels)
      goal = list(self._graph.nodes())[random_idx]
    self._goal = goal

  @property
  def position(self):
    return self._position

  @property
  def goal(self):
    return self._goal

  @property
  def previous_action(self):
    return self._previous_action

  @property
  def episode_reward(self):
    return self._episode_reward

  def draw_maze(self, ax=None):
    if ax is None:
      plt.figure()
      ax = plt.gca()
    node_positions = {(x, y): (x, y) for x, y in self._graph.nodes()}
    letters = string.ascii_uppercase + string.ascii_lowercase
    labels = {n: letters[self._labels[n]] for n in self._graph.nodes()}
    node_list = list(self._graph.nodes())
    colors = []
    for n in node_list:
      if n == self.position:
        colors.append('lightblue')
      elif n == self.goal:
        colors.append('lightgreen')
      else:
        colors.append('pink')
    nx.draw(self._graph, pos=node_positions, nodelist=node_list, ax=ax,
            node_color=colors, with_labels=True, node_size=200, labels=labels)
    ax.set_title('{}\nEpisode reward={:.1f}'.format(
        self.previous_action, self.episode_reward))
    ax.margins(.1)
    return plt.gcf(), ax
