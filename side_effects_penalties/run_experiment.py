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
"""Run a Q-learning agent with a side effects penalty."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import pandas as pd
from six.moves import range
from six.moves import zip
from side_effects_penalties import agent_with_penalties
from side_effects_penalties import training
from side_effects_penalties.file_loading import filename


FLAGS = flags.FLAGS

if __name__ == '__main__':  # Avoid defining flags when used as a library.
  # Side effects penalty settings
  flags.DEFINE_enum('baseline', 'inaction',
                    ['start', 'inaction', 'stepwise', 'step_noroll'],
                    'Baseline.')
  flags.DEFINE_enum('dev_measure', 'rel_reach',
                    ['none', 'reach', 'rel_reach',
                     'uvfa_rel_reach', 'att_util'],
                    'Deviation measure.')
  flags.DEFINE_enum('dev_fun', 'truncation', ['truncation', 'absolute'],
                    'Summary function for the deviation measure.')
  flags.DEFINE_float('discount', 0.99, 'Discount factor for rewards.')
  flags.DEFINE_float('value_discount', 0.99,
                     'Discount factor for deviation measure value function.')
  flags.DEFINE_float('beta', 30.0, 'Weight for side effects penalty.')
  flags.DEFINE_string('nonterminal', 'disc',
                      'Penalty for nonterminal states relative to terminal'
                      'states: none (0), full (1), or disc (1-discount).')
  flags.DEFINE_bool('exact_baseline', False,
                    'Compute the exact baseline using an environment copy.')
  # Agent settings
  flags.DEFINE_bool('anneal', True,
                    'Whether to anneal the exploration rate from 1 to 0.')
  flags.DEFINE_integer('num_episodes', 10000, 'Number of episodes.')
  flags.DEFINE_integer('num_episodes_noexp', 0,
                       'Number of episodes with no exploration.')
  flags.DEFINE_integer('seed', 1, 'Random seed.')
  # Environment settings
  flags.DEFINE_string('env_name', 'box', 'Environment name.')
  flags.DEFINE_bool('noops', True, 'Whether the environment includes noops.')
  flags.DEFINE_integer('movement_reward', 0, 'Movement reward.')
  flags.DEFINE_integer('goal_reward', 1, 'Reward for reaching a goal state.')
  flags.DEFINE_integer('side_effect_reward', -1,
                       'Hidden reward for causing side effects.')
  # Settings for outputting results
  flags.DEFINE_enum('mode', 'save', ['print', 'save'],
                    'Print results or save to file.')
  flags.DEFINE_string('path', '', 'File path.')
  flags.DEFINE_string('suffix', '', 'Filename suffix.')


def run_experiment(
    baseline, dev_measure, dev_fun, discount, value_discount, beta, nonterminal,
    exact_baseline, anneal, num_episodes, num_episodes_noexp, seed,
    env_name, noops, movement_reward, goal_reward, side_effect_reward,
    mode, path, suffix):
  """Run agent and save or print the results."""
  performances = []
  rewards = []
  seeds = []
  episodes = []
  if 'rel_reach' not in dev_measure and 'att_util' not in dev_measure:
    dev_fun = 'none'
  nonterminal_weights = {'none': 0.0, 'disc': 1.0-discount, 'full': 1.0}
  nonterminal_weight = nonterminal_weights[nonterminal]
  reward, performance = training.run_agent(
      baseline=baseline,
      dev_measure=dev_measure,
      dev_fun=dev_fun,
      discount=discount,
      value_discount=value_discount,
      beta=beta,
      nonterminal_weight=nonterminal_weight,
      exact_baseline=exact_baseline,
      anneal=anneal,
      num_episodes=num_episodes,
      num_episodes_noexp=num_episodes_noexp,
      seed=seed,
      env_name=env_name,
      noops=noops,
      movement_reward=movement_reward,
      goal_reward=goal_reward,
      side_effect_reward=side_effect_reward,
      agent_class=agent_with_penalties.QLearningSE)
  rewards.extend(reward)
  performances.extend(performance)
  seeds.extend([seed] * (num_episodes + num_episodes_noexp))
  episodes.extend(list(range(num_episodes + num_episodes_noexp)))
  if mode == 'save':
    d = {'reward': rewards, 'performance': performances,
         'seed': seeds, 'episode': episodes}
    df = pd.DataFrame(d)
    df1 = add_smoothed_data(df)
    f = filename(env_name, noops, dev_measure, dev_fun, baseline, beta,
                 value_discount, path=path, suffix=suffix, seed=seed)
    df1.to_csv(f)
  return reward, performance


def _smooth(values, window=100):
  return values.rolling(window,).mean()


def add_smoothed_data(df, groupby='seed', window=100):
  grouped = df.groupby(groupby)[['reward', 'performance']]
  grouped = grouped.apply(_smooth, window=window).rename(columns={
      'performance': 'performance_smooth', 'reward': 'reward_smooth'})
  temp = pd.concat([df, grouped], axis=1)
  return temp


def main(unused_argv):
  reward, performance = run_experiment(
      baseline=FLAGS.baseline,
      dev_measure=FLAGS.dev_measure,
      dev_fun=FLAGS.dev_fun,
      discount=FLAGS.discount,
      value_discount=FLAGS.value_discount,
      beta=FLAGS.beta,
      nonterminal=FLAGS.nonterminal,
      exact_baseline=FLAGS.exact_baseline,
      anneal=FLAGS.anneal,
      num_episodes=FLAGS.num_episodes,
      num_episodes_noexp=FLAGS.num_episodes_noexp,
      seed=FLAGS.seed,
      env_name=FLAGS.env_name,
      noops=FLAGS.noops,
      movement_reward=FLAGS.movement_reward,
      goal_reward=FLAGS.goal_reward,
      side_effect_reward=FLAGS.side_effect_reward,
      mode=FLAGS.mode,
      path=FLAGS.path,
      suffix=FLAGS.suffix)
  if FLAGS.mode == 'print':
    print('Performance and reward in the last 10 steps:')
    print(list(zip(performance, reward))[-10:-1])


if __name__ == '__main__':
  app.run(main)
