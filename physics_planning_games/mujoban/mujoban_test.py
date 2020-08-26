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

"""Tests for Mujoban."""


from absl.testing import absltest
from dm_control import composer
from dm_control.locomotion import walkers
import dm_env as environment
import numpy as np

from physics_planning_games.mujoban.mujoban import Mujoban
from physics_planning_games.mujoban.mujoban_level import MujobanLevel


TIME_LIMIT = 5
CONTROL_TIMESTEP = .1


class MujobanTest(absltest.TestCase):

  def test(self):
    walker = walkers.JumpingBallWithHead(add_ears=True, camera_height=0.25)
    arena = MujobanLevel()
    task = Mujoban(
        walker=walker,
        maze=arena,
        control_timestep=CONTROL_TIMESTEP,
        top_camera_height=64,
        top_camera_width=48)
    env = composer.Environment(
        time_limit=TIME_LIMIT,
        task=task,
        strip_singleton_obs_buffer_dim=True)
    time_step = env.reset()
    self.assertEqual(
        set([
            'pixel_layer', 'full_entity_layer', 'top_camera',
            'walker/body_height', 'walker/end_effectors_pos',
            'walker/joints_pos', 'walker/joints_vel',
            'walker/sensors_accelerometer', 'walker/sensors_gyro',
            'walker/sensors_touch', 'walker/sensors_velocimeter',
            'walker/world_zaxis', 'walker/orientation',
        ]), set(time_step.observation.keys()))
    top_camera = time_step.observation['top_camera']
    self.assertEqual(np.uint8, top_camera.dtype)
    self.assertEqual((64, 48, 3), top_camera.shape)
    all_step_types = []
    # Run enough actions that we are guaranteed to have restarted the
    # episode at least once.
    for _ in range(int(2*TIME_LIMIT/CONTROL_TIMESTEP)):
      action = 2*np.random.random(env.action_spec().shape) - 1
      time_step = env.step(action)
      all_step_types.append(time_step.step_type)
    self.assertEqual(set([environment.StepType.FIRST,
                          environment.StepType.MID,
                          environment.StepType.LAST]),
                     set(all_step_types))


if __name__ == '__main__':
  absltest.main()
