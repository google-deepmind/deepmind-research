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

"""One-shot StreetLearn environment."""

import dm_env
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def deg_to_rad(x):
  """Convert degrees to radians."""
  return x / 180. * np.pi


def rad_to_deg(x):
  """Convert radians to degrees."""
  return x * 180. / np.pi


class OneShotStreetLearn(dm_env.Environment):
  """One-shot Streetlearn environment."""

  ACTION_NAMES = [
      'Forward',
      'Left',
      'Right',
      'Collect',
  ]
  NUM_ACTIONS = len(ACTION_NAMES)

  def __init__(self, dataset_path, max_episode_steps, num_junctions=8,
               target_reward=1., per_step_reward=0., observation_length=60,
               seed=None):
    self._graph = nx.read_gexf(dataset_path)
    self._node_attrs = self._graph.nodes(data=True)
    self._num_junctions = num_junctions
    self._observation_length = observation_length
    self._max_episode_steps = max_episode_steps
    self._target_reward = target_reward
    self._per_step_reward = per_step_reward
    self._rng = np.random.RandomState(seed)
    self.reset()

  def reset(self):
    self._previous_action = ''
    self._episode_reward = 0.
    self._episode_steps = 0
    self._needs_reset = False
    self._subgraph = self.get_random_subgraph()
    self._observation_map = self.randomize_observations(self._subgraph)
    self._position = self._rng.choice(list(self._subgraph.nodes()))
    neighbours = self._neighbors_bearings(self._subgraph, self._position)
    self._neighbour = neighbours[self._rng.randint(len(neighbours))]
    self._set_new_goal()
    return dm_env.restart(self._observation())

  @property
  def _current_edge(self):
    return (self._position, self._neighbour['neighbour'])

  def _set_new_goal(self):
    goal = None
    edges = list(self._observation_map.keys())
    while goal is None or goal == self._current_edge:
      goal = edges[self._rng.randint(len(edges))]
    self._goal = goal

  def _one_hot(self, edge):
    one_hot_vector = np.zeros([self._observation_length], dtype=np.int32)
    one_hot_vector[self._observation_map[edge]] = 1
    return one_hot_vector

  def _observation(self):
    return {
        'position': np.array(self._one_hot(self._current_edge), dtype=np.int32),
        'goal': np.array(self._one_hot(self._goal), dtype=np.int32),
    }

  def observation_spec(self):
    return {
        'position': dm_env.specs.Array(
            shape=(self._observation_length,), dtype=np.int32, name='position'),
        'goal': dm_env.specs.Array(
            shape=(self._observation_length,), dtype=np.int32, name='goal'),
    }

  def action_spec(self):
    return dm_env.specs.DiscreteArray(self.NUM_ACTIONS)

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

    # Recompute agent's position
    self._move(action)
    self._previous_action = self.ACTION_NAMES[action]

    # Get reward if agent is at the goal location and the selected action is
    # `collect`.
    if (self._current_edge == self._goal and
        self.ACTION_NAMES[action] == 'Collect'):
      reward = self._target_reward
      self._set_new_goal()
    else:
      reward = self._per_step_reward
    self._episode_reward += reward

    return transition(reward, self._observation())

  def randomize_observations(self, subgraph):
    edges = list(subgraph.edges())
    edges.extend([(y, x) for (x, y) in edges])
    obs_permutation = self._rng.permutation(self._observation_length)
    return {e: obs_permutation[i] for i, e in enumerate(edges)}

  def _calculate_bearing(self, node, neighbor):
    lat1 = deg_to_rad(self._node_attrs[node]['lat'])
    lng1 = deg_to_rad(self._node_attrs[node]['lng'])
    lat2 = deg_to_rad(self._node_attrs[neighbor]['lat'])
    lng2 = deg_to_rad(self._node_attrs[neighbor]['lng'])
    delta_lng = lng2 - lng1
    theta = np.arctan2(
        np.sin(delta_lng) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) -
        np.sin(lat1) * np.cos(lat2) * np.cos(delta_lng))
    return theta

  def _neighbors_bearings(self, subgraph, node):
    bearings = []
    for neighbor in list(subgraph[node]):
      orientation = self._calculate_bearing(node, neighbor)
      bearings.append({'neighbour': neighbor, 'orientation': orientation})
    bearings.sort(key=lambda x: x['orientation'])
    return bearings

  def _sort_neighbors(self, node, neighbour):
    bearings = self._neighbors_bearings(self._subgraph, node)
    bs = [x['orientation'] for x in bearings]
    idx = np.argmin(np.abs(bs - neighbour['orientation']))
    return {
        'forward': bearings[idx],
        'right': bearings[idx-1],
        'left': bearings[(idx+1) % len(bearings)],
    }

  def _move(self, action):
    neighbours = self._sort_neighbors(self._position, self._neighbour)
    if action == 0:
      new_node = self._neighbour['neighbour']
      neighbours = self._sort_neighbors(new_node, neighbours['forward'])
      new_neighbour = neighbours['forward']
    else:
      new_node = self._position
      if action == 1:
        new_neighbour = neighbours['left']
      elif action == 2:
        new_neighbour = neighbours['right']
      else:
        new_neighbour = self._neighbour
    self._position = new_node
    self._neighbour = new_neighbour

  def _all_next_junctions(self, subgraph, node):
    neighbors = list(subgraph[node])
    edges = [self._get_next_junction(subgraph, node, nb) for nb in neighbors]
    nodes = [y for (_, y) in edges]
    return nodes, edges

  def _get_next_junction(self, subgraph, initial_node, next_node):
    node = initial_node
    while subgraph.degree(next_node) == 2:
      neighbours = list(subgraph.neighbors(next_node))
      neighbours.remove(node)
      node = next_node
      next_node = neighbours.pop()
    return (initial_node, next_node)

  def get_random_subgraph(self):
    graph = self._graph
    num_nodes = len(graph)
    rnd_index = self._rng.randint(num_nodes)
    center_node = list(graph.nodes())[rnd_index]
    while graph.degree(center_node) <= 2:
      rnd_index = self._rng.randint(num_nodes)
      center_node = list(graph.nodes())[rnd_index]
    to_visit = [center_node]
    visited = []
    subgraph = nx.Graph()
    while to_visit:
      node = to_visit.pop(0)
      visited.append(node)
      new_nodes, new_edges = self._all_next_junctions(graph, node)
      subgraph.add_edges_from(new_edges)
      node_degrees = [subgraph.degree(n) for n in subgraph.nodes()]
      count_junctions = len(list(filter(lambda x: x > 2, node_degrees)))
      if count_junctions >= self._num_junctions:
        break
      new_nodes = filter(lambda x: x not in visited + to_visit, new_nodes)
      to_visit.extend(new_nodes)
    return subgraph

  def draw_subgraph(self, ax=None):
    if ax is None:
      _ = plt.figure(figsize=(3, 3))
      ax = plt.gca()
    node_ids = list(self._subgraph.nodes())
    pos = {
        x: (self._node_attrs[x]['lat'], self._node_attrs[x]['lng'])
        for x in node_ids
    }
    labels = {}
    nc = 'pink'
    ec = 'black'
    ns = 50
    nshape = 'o'
    # Draw the current subgraph
    nx.draw(self._subgraph, pos=pos, node_color=nc, with_labels=False,
            node_size=ns, labels=labels, edgecolors=ec, node_shape=nshape,
            ax=ax)
    max_xy = np.array([np.array(x) for x in pos.values()]).max(0)
    min_xy = np.array([np.array(x) for x in pos.values()]).min(0)
    delta_xy = (max_xy - min_xy) / 6.
    ax.set_xlim([min_xy[0] - delta_xy[0], max_xy[0] + delta_xy[0]])
    ax.set_ylim([min_xy[1] - delta_xy[1], max_xy[1] + delta_xy[1]])
    # Draw goal position and orientation
    x = self._node_attrs[self._goal[0]]['lat']
    y = self._node_attrs[self._goal[0]]['lng']
    rotation = rad_to_deg(self._calculate_bearing(*self._goal))
    _ = ax.plot(x, y, marker=(3, 0, rotation - 90), color=(0, 0, 0),
                markersize=14, markerfacecolor='white')
    _ = ax.plot(x, y, marker=(2, 0, rotation - 90), color=(0, 0, 0),
                markersize=12, markerfacecolor='None')
    # Draw current position and orientation
    x = self._node_attrs[self._position]['lat']
    y = self._node_attrs[self._position]['lng']
    rotation = rad_to_deg(self._neighbour['orientation'])
    _ = ax.plot(x, y, marker=(3, 0, rotation - 90), color=(0, 0, 0),
                markersize=14, markerfacecolor='lightgreen')
    _ = ax.plot(x, y, marker=(2, 0, rotation - 90), color=(0, 0, 0),
                markersize=12, markerfacecolor='None')
    ax.set_title('{}\nEpisode reward = {}'.format(
        self._previous_action, self._episode_reward))
    return plt.gcf(), ax
