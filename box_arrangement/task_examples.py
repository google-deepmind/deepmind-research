# Copyright 2019 Deepmind Technologies Limited.
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

"""Example tasks used in publications."""

from dm_control import composer
from dm_control.entities import props
from dm_control.locomotion import arenas as locomotion_arenas
from dm_control.locomotion import walkers
from dm_control.manipulation import props as manipulation_props


from box_arrangement import dmlab_assets
from box_arrangement import predicates as predicates_module
from box_arrangement.predicate_task import PredicateTask

DEFAULT_TIME_LIMIT = 20.0
DEFAULT_CONTROL_TIMESTEP = 0.05
MIN_ROOM_SIZE = 3


def _make_predicate_task(n_boxes, n_targets,
                         include_gtt_predicates, include_move_box_predicates,
                         max_num_predicates, control_timestep, time_limit):
  """Auxiliary function to construct different predicates tasks."""
  walker = walkers.Ant()
  skybox = dmlab_assets.SkyBox(style='sky_03')
  wall = dmlab_assets.WallTextures(style='style_03')
  floor = dmlab_assets.FloorTextures(style='style_03')

  # Make room size become bigger once the number of objects become larger.
  num_objects = n_boxes + n_targets
  room_size = max(MIN_ROOM_SIZE, num_objects)
  text_maze = locomotion_arenas.padded_room.PaddedRoom(
      room_size=room_size, num_objects=num_objects, pad_with_walls=True)
  arena = locomotion_arenas.MazeWithTargets(
      maze=text_maze,
      skybox_texture=skybox,
      wall_textures=wall,
      floor_textures=floor)

  boxes = []
  for _ in range(n_boxes):
    boxes.append(
        manipulation_props.BoxWithSites(mass=1.5, half_lengths=[0.5, 0.5, 0.5]))

  targets = []
  for _ in range(n_targets):
    targets.append(
        props.PositionDetector(
            pos=[0, 0, 0.5], size=[0.5, 0.5, 0.5], inverted=False,
            visible=True))

  predicates = []
  if include_gtt_predicates:
    predicates.append(
        predicates_module.MoveWalkerToRandomTarget(
            walker=walker, targets=targets))
  if include_move_box_predicates:
    for box_idx in range(len(boxes)):
      predicates.append(
          predicates_module.MoveBoxToRandomTarget(
              walker=walker,
              box=boxes[box_idx],
              box_index=box_idx,
              targets=targets))

  task = PredicateTask(
      walker=walker,
      maze_arena=arena,
      predicates=predicates,
      props=boxes,
      targets=targets,
      max_num_predicates=max_num_predicates,
      randomize_num_predicates=False,
      reward_scale=10.,
      regenerate_predicates=False,
      physics_timestep=0.005,
      control_timestep=control_timestep)
  env = composer.Environment(task=task, time_limit=time_limit)

  return env


def go_to_k_targets(n_targets=3,
                    time_limit=DEFAULT_TIME_LIMIT,
                    control_timestep=DEFAULT_CONTROL_TIMESTEP):
  """Loads `go_to_k_targets` task."""
  return _make_predicate_task(
      n_boxes=0,
      n_targets=n_targets,
      include_gtt_predicates=True,
      include_move_box_predicates=False,
      max_num_predicates=1,
      control_timestep=control_timestep,
      time_limit=time_limit)


def move_box(n_targets=3,
             time_limit=DEFAULT_TIME_LIMIT,
             control_timestep=DEFAULT_CONTROL_TIMESTEP):
  """Loads `move_box` task."""
  return _make_predicate_task(
      n_boxes=1,
      n_targets=n_targets,
      include_gtt_predicates=False,
      include_move_box_predicates=True,
      max_num_predicates=1,
      control_timestep=control_timestep,
      time_limit=time_limit)


def move_box_or_gtt(n_targets=3,
                    time_limit=DEFAULT_TIME_LIMIT,
                    control_timestep=DEFAULT_CONTROL_TIMESTEP):
  """Loads `move_box_or_gtt` task."""
  return _make_predicate_task(
      n_boxes=1,
      n_targets=n_targets,
      include_gtt_predicates=True,
      include_move_box_predicates=True,
      max_num_predicates=1,
      control_timestep=control_timestep,
      time_limit=time_limit)


def move_box_and_gtt(n_targets=3,
                     time_limit=DEFAULT_TIME_LIMIT,
                     control_timestep=DEFAULT_CONTROL_TIMESTEP):
  """Loads `move_box_or_gtt` task."""
  return _make_predicate_task(
      n_boxes=1,
      n_targets=n_targets,
      include_gtt_predicates=True,
      include_move_box_predicates=True,
      max_num_predicates=2,
      control_timestep=control_timestep,
      time_limit=time_limit)
