# Copyright 2020 DeepMind Technologies Limited.
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

"""Visualize physical planning games in Domain Explorer.
"""

import functools

from absl import app
from absl import flags
from dm_control import composer
from dm_control import viewer
from dm_control.locomotion import walkers

from physics_planning_games import board_games
from physics_planning_games.mujoban.boxoban import boxoban_level_generator
from physics_planning_games.mujoban.mujoban import Mujoban
from physics_planning_games.mujoban.mujoban_level import MujobanLevel

flags.DEFINE_enum('environment_name', 'mujoban', [
    'mujoban', 'go_7x7', 'tic_tac_toe_markers_features',
    'tic_tac_toe_mixture_opponent_markers_features',
    'tic_tac_toe_optimal_opponent_markers_features'],
                  'Name of an environment to load.')
FLAGS = flags.FLAGS

TIME_LIMIT = 1000
CONTROL_TIMESTEP = .1


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  environment_name = FLAGS.environment_name
  if environment_name == 'mujoban':
    walker = walkers.JumpingBallWithHead(add_ears=True, camera_height=0.25)
    arena = MujobanLevel(boxoban_level_generator)
    task = Mujoban(
        walker=walker,
        maze=arena,
        control_timestep=CONTROL_TIMESTEP,
        top_camera_height=64,
        top_camera_width=48)
    env = composer.Environment(
        time_limit=TIME_LIMIT, task=task, strip_singleton_obs_buffer_dim=True)
  else:
    env = functools.partial(
        board_games.load, environment_name=environment_name)

  viewer.launch(env)

if __name__ == '__main__':
  app.run(main)
