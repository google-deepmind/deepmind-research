# Copyright 2020 Deepmind Technologies Limited.
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

"""Tests for box_arrangement.predicate_task."""

from absl.testing import absltest
from dm_control import composer
from dm_control.entities import props
from dm_control.locomotion import arenas
from dm_control.locomotion import walkers
import numpy as np

from box_arrangement import predicate_task
from box_arrangement import predicates


_EGOCENTRIC_OBSERVABLES = [
    "walker/body_height",
    "walker/end_effectors_pos",
    "walker/joints_pos",
    "walker/joints_vel",
    "walker/sensors_accelerometer",
    "walker/sensors_gyro",
    "walker/sensors_velocimeter",
    "walker/world_zaxis",
]


class PredicateTaskTest(absltest.TestCase):

  def _setup_basic_gtt_task(self, num_targets=1, reward_scale=1.0):
    walker = walkers.Ant()
    text_maze = arenas.padded_room.PaddedRoom(
        room_size=8, num_objects=2, pad_with_walls=True)
    maze_arena = arenas.MazeWithTargets(maze=text_maze)
    targets = []
    for _ in range(num_targets):
      targets.append(
          props.PositionDetector(
              pos=[0, 0, 0.5],
              size=[0.5, 0.5, 0.5],
              inverted=False,
              visible=True))
    test_predicates = [predicates.MoveWalkerToRandomTarget(walker, targets)]
    self._task = predicate_task.PredicateTask(
        walker=walker,
        maze_arena=maze_arena,
        predicates=test_predicates,
        targets=targets,
        randomize_num_predicates=False,
        reward_scale=reward_scale,
        terminating_reward_bonus=2.0,
        )
    random_state = np.random.RandomState(12345)
    self._env = composer.Environment(self._task, random_state=random_state)
    self._walker = walker
    self._targets = targets

  def test_observables(self):
    self._setup_basic_gtt_task()
    timestep = self._env.reset()
    self.assertIn("predicate_0", timestep.observation)
    self.assertIn("walker/target_positions", timestep.observation)
    for observable in _EGOCENTRIC_OBSERVABLES:
      self.assertIn(observable, timestep.observation)

  def test_termination_and_discount(self):
    self._setup_basic_gtt_task()
    self._env.reset()
    target_pos = (0, 0, 0.5)
    # Initialize the walker away from the target.
    self._walker.set_pose(
        self._env.physics, position=(-2, 0, 0.0), quaternion=(1, 0, 0, 0))
    self._targets[0].set_position(
        self._env.physics,
        target_pos)
    self._env.physics.forward()
    zero_action = np.zeros_like(self._env.physics.data.ctrl)
    for _ in range(10):
      timestep = self._env.step(zero_action)
      self.assertEqual(timestep.discount, 1.0)
      self.assertEqual(timestep.reward, 0.0)

    walker_pos = (0, 0, 0.0)
    self._walker.set_pose(
        self._env.physics,
        position=walker_pos)
    self._env.physics.forward()

    # For a single predicate, first the reward is +1.0 for activating the
    # predicate
    timestep = self._env.step(zero_action)
    self.assertEqual(timestep.discount, 1.0)
    self.assertEqual(timestep.reward, 1.0)
    # If the predicate is active and *remains* active, the discount gets to 0.0
    # and the terminating reward bonus is given.
    timestep = self._env.step(zero_action)
    self.assertEqual(timestep.discount, 0.0)
    self.assertEqual(timestep.reward, 2.0)
    # Make sure this is a termination step.
    self.assertTrue(timestep.last())

  def test_reward_scaling(self):
    self._setup_basic_gtt_task(reward_scale=10.0)
    self._env.reset()
    zero_action = np.zeros_like(self._env.physics.data.ctrl)
    target_pos = (0, 0, 0.5)
    walker_pos = (0, 0, 0.0)
    self._targets[0].set_position(self._env.physics, target_pos)
    self._walker.set_pose(self._env.physics, position=walker_pos)
    self._env.physics.forward()

    # For a single predicate, first the reward is +1.0 for activating the
    # predicate
    timestep = self._env.step(zero_action)
    self.assertEqual(timestep.discount, 1.0)
    self.assertEqual(timestep.reward, 10.0)
    # If the predicate is active and *remains* active, the discount gets to 0.0
    # and the terminating reward bonus is given.
    timestep = self._env.step(zero_action)
    self.assertEqual(timestep.discount, 0.0)
    self.assertEqual(timestep.reward, 20.0)
    # Make sure this is a termination step.
    self.assertTrue(timestep.last())

  def test_too_few_predicates_raises_exception(self):
    walker = walkers.Ant()
    num_targets = 1
    text_maze = arenas.padded_room.PaddedRoom(
        room_size=8, num_objects=2, pad_with_walls=True)
    maze_arena = arenas.MazeWithTargets(maze=text_maze)
    targets = []
    for _ in range(num_targets):
      targets.append(
          props.PositionDetector(
              pos=[0, 0, 0.5],
              size=[0.5, 0.5, 0.5],
              inverted=False,
              visible=True))
    test_predicates = []
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Not enough predicates for task."
        " The maximum number of "
        "predicates can be "
        "1 but only 0 predicates provided."):
      predicate_task.PredicateTask(
          walker=walker,
          maze_arena=maze_arena,
          predicates=test_predicates,
          targets=targets,
          randomize_num_predicates=False,
          reward_scale=1.0,
          terminating_reward_bonus=2.0,
          )

  def test_error_too_few_targets(self):
    walker = walkers.Ant()
    num_targets = 5
    text_maze = arenas.padded_room.PaddedRoom(
        room_size=8, num_objects=2, pad_with_walls=True)
    maze_arena = arenas.MazeWithTargets(maze=text_maze)
    targets = []
    for _ in range(num_targets):
      targets.append(
          props.PositionDetector(
              pos=[0, 0, 0.5],
              size=[0.5, 0.5, 0.5],
              inverted=False,
              visible=True))
    test_predicates = [predicates.MoveWalkerToRandomTarget(walker, targets)]
    task = predicate_task.PredicateTask(
        walker=walker,
        maze_arena=maze_arena,
        predicates=test_predicates,
        targets=targets,
        randomize_num_predicates=False,
        reward_scale=1.0,
        terminating_reward_bonus=2.0,
    )
    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    with self.assertRaisesWithLiteralMatch(
        RuntimeError, "The generated maze does not contain enough target "
        "positions for the requested number of props (0) and targets (5): "
        "got 2."
    ):
      env.reset()

  def test_error_if_no_predicates_found(self):
    walker = walkers.Ant()
    num_targets = 2
    text_maze = arenas.padded_room.PaddedRoom(
        room_size=8, num_objects=6, pad_with_walls=True)
    maze_arena = arenas.MazeWithTargets(maze=text_maze)
    targets = []
    for _ in range(num_targets):
      targets.append(
          props.PositionDetector(
              pos=[0, 0, 0.5],
              size=[0.5, 0.5, 0.5],
              inverted=False,
              visible=True))
    # Moving the walker to two targets is not possible since the walker is a
    # shared object in use.
    test_predicates = [predicates.MoveWalkerToTarget(walker, targets[0]),
                       predicates.MoveWalkerToTarget(walker, targets[1])]
    task = predicate_task.PredicateTask(
        walker=walker,
        maze_arena=maze_arena,
        predicates=test_predicates,
        targets=targets[1:],
        randomize_num_predicates=False,
        max_num_predicates=2,
        reward_scale=1.0,
        terminating_reward_bonus=2.0,
    )
    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Could not find set of active predicates"
        " with unique objects are after 1000 iterations."):
      env.reset()

    # However moving to one of the two targets is fine.
    walker = walkers.Ant()
    num_targets = 2
    text_maze = arenas.padded_room.PaddedRoom(
        room_size=8, num_objects=6, pad_with_walls=True)
    maze_arena = arenas.MazeWithTargets(maze=text_maze)
    targets = []
    for _ in range(num_targets):
      targets.append(
          props.PositionDetector(
              pos=[0, 0, 0.5],
              size=[0.5, 0.5, 0.5],
              inverted=False,
              visible=True))
    test_predicates = [predicates.MoveWalkerToTarget(walker, targets[0]),
                       predicates.MoveWalkerToTarget(walker, targets[1])]
    task = predicate_task.PredicateTask(
        walker=walker,
        maze_arena=maze_arena,
        predicates=test_predicates,
        targets=targets[1:],
        randomize_num_predicates=False,
        max_num_predicates=1,
        reward_scale=1.0,
        terminating_reward_bonus=2.0,
    )
    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    env.reset()


if __name__ == "__main__":
  absltest.main()
