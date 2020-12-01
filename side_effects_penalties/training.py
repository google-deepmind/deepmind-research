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
"""Training loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ai_safety_gridworlds.helpers import factory
import numpy as np
from six.moves import range


def get_env(env_name, noops,
            movement_reward=-1, goal_reward=1, side_effect_reward=-1):
  """Get a copy of the environment for simulating the baseline."""
  if env_name == 'box' or 'sokocoin' in env_name:
    levels = {'box': 0, 'sokocoin1': 1, 'sokocoin2': 2, 'sokocoin3': 3}
    sizes = {'box': 36, 'sokocoin1': 100, 'sokocoin2': 72, 'sokocoin3': 100}
    env = factory.get_environment_obj(
        'side_effects_sokoban', noops=noops, movement_reward=movement_reward,
        goal_reward=goal_reward, wall_reward=side_effect_reward,
        corner_reward=side_effect_reward, level=levels[env_name])
    size = sizes[env_name]
  elif 'sushi' in env_name or env_name == 'vase':
    env = factory.get_environment_obj(
        'conveyor_belt', variant=env_name, noops=noops, goal_reward=goal_reward)
    size = 49
  else:
    env = factory.get_environment_obj(env_name)
    size = None
  return env, size


def run_loop(agent, env, number_episodes, anneal):
  """Training agent."""
  episodic_returns = []
  episodic_performances = []
  if anneal:
    agent.epsilon = 1.0
    eps_unit = 1.0 / number_episodes
  for episode in range(number_episodes):
    # Get the initial set of observations from the environment.
    timestep = env.reset()
    # Prepare agent for a new episode.
    agent.begin_episode()
    while True:
      action = agent.step(timestep)
      timestep = env.step(action)
      if timestep.last():
        agent.end_episode(timestep)
        episodic_returns.append(env.episode_return)
        episodic_performances.append(env.get_last_performance())
        break
    if anneal:
      agent.epsilon = max(0, agent.epsilon - eps_unit)
    if episode % 500 == 0:
      print('Episode', episode)
  return episodic_returns, episodic_performances


def run_agent(baseline, dev_measure, dev_fun, discount, value_discount, beta,
              nonterminal_weight, exact_baseline, anneal, num_episodes,
              num_episodes_noexp, seed, env_name, noops, movement_reward,
              goal_reward, side_effect_reward, agent_class):
  """Run agent.

  Create an agent with the given parameters for the side effects penalty.
  Run the agent for `num_episodes' episodes with an exploration rate that is
  either annealed from 1 to 0 (`anneal=True') or constant (`anneal=False').
  Then run the agent with no exploration for `num_episodes_noexp' episodes.

  Args:
    baseline: baseline state
    dev_measure: deviation measure
    dev_fun: summary function for the deviation measure
    discount: discount factor
    value_discount: discount factor for deviation measure value function.
    beta: weight for side effects penalty
    nonterminal_weight: penalty weight for nonterminal states.
    exact_baseline: whether to use an exact or approximate baseline
    anneal: whether to anneal the exploration rate from 1 to 0 or use a constant
      exploration rate
    num_episodes: number of episodes
    num_episodes_noexp: number of episodes with no exploration
    seed: random seed
    env_name: environment name
    noops: whether the environment has noop actions
    movement_reward: movement reward
    goal_reward: reward for reaching a goal state
    side_effect_reward: hidden reward for causing side effects
    agent_class: Q-learning agent class: QLearning (regular) or QLearningSE
      (with side effects penalty)

  Returns:
    returns: return for each episode
    performances: safety performance for each episode
  """
  np.random.seed(seed)
  env, state_size = get_env(env_name=env_name,
                            noops=noops,
                            movement_reward=movement_reward,
                            goal_reward=goal_reward,
                            side_effect_reward=side_effect_reward)
  start_timestep = env.reset()
  if exact_baseline:
    baseline_env, _ = get_env(env_name=env_name,
                              noops=True,
                              movement_reward=movement_reward,
                              goal_reward=goal_reward,
                              side_effect_reward=side_effect_reward)
  else:
    baseline_env = None
  agent = agent_class(
      actions=env.action_spec(), baseline=baseline, dev_measure=dev_measure,
      dev_fun=dev_fun, discount=discount, value_discount=value_discount,
      beta=beta, exact_baseline=exact_baseline, baseline_env=baseline_env,
      start_timestep=start_timestep, state_size=state_size,
      nonterminal_weight=nonterminal_weight)
  returns, performances = run_loop(
      agent, env, number_episodes=num_episodes, anneal=anneal)
  if num_episodes_noexp > 0:
    agent.epsilon = 0
    returns_noexp, performances_noexp = run_loop(
        agent, env, number_episodes=num_episodes_noexp, anneal=False)
    returns.extend(returns_noexp)
    performances.extend(performances_noexp)
  return returns, performances
