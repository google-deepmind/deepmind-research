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
"""A simple training loop."""

import csv

from absl import logging
from tensorflow.compat.v1.io import gfile


def _ema(base, val, decay=0.995):
  return base * decay + (1 - decay) * val


def run(env, agent, num_episodes, report_every=200, num_eval_reps=1):
  """Runs an agent on an environment.

  Args:
    env: The environment.
    agent: The agent.
    num_episodes: Number of episodes to train for.
    report_every: Frequency at which training progress are reported (episodes).
    num_eval_reps: Number of eval episodes to run per training episode.

  Returns:
    A list of dicts containing training and evaluation returns, and a list of
    reported returns smoothed by EMA.
  """

  returns = []
  logged_returns = []
  train_return_ema = 0.
  eval_return_ema = 0.
  for episode in range(num_episodes):
    returns.append(dict(episode=episode))

    # Run a training episode.
    train_episode_return = run_episode(env, agent, is_training=True)
    train_return_ema = _ema(train_return_ema, train_episode_return)
    returns[-1]["train"] = train_episode_return

    # Run an evaluation episode.
    returns[-1]["eval"] = []
    for _ in range(num_eval_reps):
      eval_episode_return = run_episode(env, agent, is_training=False)
      eval_return_ema = _ema(eval_return_ema, eval_episode_return)
      returns[-1]["eval"].append(eval_episode_return)

    if ((episode + 1) % report_every) == 0 or episode == 0:
      logged_returns.append(
          dict(episode=episode, train=train_return_ema, eval=[eval_return_ema]))
      logging.info("Episode %s, avg train return %.3f, avg eval return %.3f",
                   episode + 1, train_return_ema, eval_return_ema)
      if hasattr(agent, "get_logs"):
        logging.info("Episode %s, agent logs: %s", episode + 1,
                     agent.get_logs())

  return returns, logged_returns


def run_episode(environment, agent, is_training=False):
  """Run a single episode."""

  timestep = environment.reset()

  while not timestep.last():
    action = agent.step(timestep, is_training)
    new_timestep = environment.step(action)

    if is_training:
      agent.update(timestep, action, new_timestep)

    timestep = new_timestep

  episode_return = environment.episode_return

  return episode_return


def write_returns_to_file(path, returns):
  """Write returns to file."""

  with gfile.GFile(path, "w") as file:
    writer = csv.writer(file, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["episode", "train"] +
                    [f"eval_{idx}" for idx in range(len(returns[0]["eval"]))])
    for row in returns:
      writer.writerow([row["episode"], row["train"]] + row["eval"])
