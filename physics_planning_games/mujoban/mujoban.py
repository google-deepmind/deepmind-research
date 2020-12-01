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

"""MuJoban task.

Mujoban is a single player puzzle-solving game embedded in the MuJoCo
simulation environment. The puzzle is based on the 2D game of Sokoban,
where an agent situated on a grid has to push boxes onto target locations.
"""

import collections

from dm_control import composer
from dm_control.composer.observation import observable as observable_lib
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas.mazes import MazeWithTargets
from dm_env import specs
import numpy as np
from six.moves import range
from six.moves import zip

from physics_planning_games.mujoban import mujoban_level
from physics_planning_games.mujoban.mujoban_pad import MujobanPad
from physics_planning_games.mujoban.props import BoxWithSites

_FLOOR_GAP_CHAR = '#'
_AMBIENT_HEADLIGHT = 0.8
_BOX_SIZE = 0.4
_BOX_HEIGHT = 0.15
_BOX_MASS = 2.5
_BOX_FRICTION = [0.5, 0.005, 0.0001]

_BOX_RGBA = [173. / 255., 179. / 255., 60. / 255., 1.]
_BOX_PRESSED_RGBA = [0, 0, 1, 1]
_TARGET_RGBA = [1.0, 0., 0., 1.]
_PRESSED_TARGET_RGBA = [0., 1., 0., 1.]

_PEG_SIZE = 0.05
_PEG_HEIGHT = 0.25
_PEG_RGBA = [0.5, 0.5, 0.5, 1]
_PEG_ANGLE = np.pi / 4

# Aliveness in [-1., 0.].
_ALIVE_THRESHOLD = -0.5

# Constants used by the full entity layer
_WALL_LAYER = 0
_TARGET_LAYER = 1
_SOKOBAN_LAYER = 2
_BOX_LAYER = 3


def _round_positions(boxes, walker, last_round_walker):
  """Round float positions to snap objects to grid."""
  round_walker = np.round(walker).astype('int32')
  round_boxes = [np.round(box).astype('int32') for box in boxes]
  for box in round_boxes:
    if np.array_equal(box, round_walker):
      round_walker = last_round_walker
  return round_boxes, round_walker


class Mujoban(composer.Task):
  """Requires objects to be moved onto matching-colored floor pads.

  Agent only receives instantaneous rewards of +1 for the
  timestep in which a box first enters a target, and -1 for the
  timestep in which a box leaves the target. There is an additional reward of
  +10 when all the boxes are put on targets, at which point the episode
  terminates.
  """

  def __init__(self,
               walker,
               maze,
               target_height=0,
               box_prop=None,
               box_size=None,
               box_mass=None,
               with_grid_pegs=False,
               detection_tolerance=0.0,
               physics_timestep=0.001,
               control_timestep=0.025,
               top_camera_height=128,
               top_camera_width=128,
               box_on_target_reward=1.0,
               level_solved_reward=10.0):
    """Initializes this task.

    Args:
      walker: A `Walker` object.
      maze: A `BaseMaze` object.
      target_height: The height of the target pads above the ground, in meters.
      box_prop: An optional `Primitive` prop to use as the box.
      box_size: An optional three element sequence defining the half lengths of
        the sides of the box.
      box_mass: Box mass. If this is a list or tuple, a random value is sampled
        from the truncated exponential distribution in [a, b) where a =
        box_mass[0] and b = box_mass[1], with scale factor box_mass[2] * (b -
        a).
      with_grid_pegs: Whether to add solid pegs at the corners of the maze
        grid cells. This helps to enforce the usual Sokoban rules where
        diagonal movements are forbidden.
      detection_tolerance: A maximum length scale (in metres) within which a
        box is allowed to stick outside a target pad while still activating it.
        For example, if this is set to 0.1 then a box will activate a pad if it
        sticks out of the pad by no more than 10 centimetres.
      physics_timestep: The time step of the physics simulation.
      control_timestep: Should be an integer multiple of the physics time step.
      top_camera_height: An int; the height of the top camera in the
        observation. Setting this to 0 will disable the top camera.
      top_camera_width: An int; the width of the top camera in the observation.
        Setting this to 0 will disable the top camera.
      box_on_target_reward: A float; reward for putting a box on a target.
      level_solved_reward: A float: reward for solving the level.
    """
    skybox_texture = labmaze_textures.SkyBox(style='sky_03')
    wall_textures = labmaze_textures.WallTextures(style='style_01')
    floor_textures = labmaze_textures.FloorTextures(style='style_01')

    self._detection_tolerance = detection_tolerance
    self._box_prop = box_prop
    self._box_on_target_reward = box_on_target_reward
    self._level_solved_reward = level_solved_reward

    self._maze = maze
    self._arena = MazeWithTargets(
        maze=maze,
        xy_scale=1,
        z_height=1,
        skybox_texture=skybox_texture,
        wall_textures=wall_textures,
        floor_textures=floor_textures)
    self._walker = walker
    self._arena.mjcf_model.visual.headlight.ambient = [_AMBIENT_HEADLIGHT] * 3
    self._arena.text_maze_regenerated_hook = self._regenerate_positions
    self._first_step = True

    # Targets.
    self._targets = []
    self._target_positions = []

    # Boxes.
    self._box_size = box_size or [_BOX_SIZE] * 2 + [_BOX_HEIGHT]
    self._box_mass = box_mass or _BOX_MASS
    self._boxes = []
    self._box_positions = []
    self._with_grid_pegs = with_grid_pegs
    self._peg_body = None
    self._last_walker_position = None

    # Create walkers and corresponding observables.
    self._walker.create_root_joints(self._arena.attach(self._walker))
    enabled_observables = [self._walker.observables.sensors_touch,
                           self._walker.observables.orientation]
    enabled_observables += self._walker.observables.proprioception
    enabled_observables += self._walker.observables.kinematic_sensors
    for observable in enabled_observables:
      observable.enabled = True
    if top_camera_width and top_camera_height:
      self._arena.observables.top_camera.enabled = True
      self._arena.observables.top_camera.width = top_camera_width
      self._arena.observables.top_camera.height = top_camera_height
    # symbolic entity repenstaion in labyrinth format.
    self._entity_layer = self._maze.entity_layer
    # pixel layer is same as pixel rendering of symbolic sokoban.
    self._pixel_layer = np.zeros(self._entity_layer.shape + (3,), dtype='uint8')
    self._full_entity_layer = np.zeros(self._entity_layer.shape + (4,),
                                       dtype='bool')
    pixel_layer_obs = observable_lib.Generic(lambda _: self._pixel_layer)
    pixel_layer_obs.enabled = True
    full_entity_layer_obs = observable_lib.Generic(
        lambda _: self._full_entity_layer)
    full_entity_layer_obs.enabled = True
    self._task_observables = collections.OrderedDict({
        'pixel_layer': pixel_layer_obs,
        'full_entity_layer': full_entity_layer_obs,
    })
    # Set time steps.
    self.set_timesteps(
        physics_timestep=physics_timestep, control_timestep=control_timestep)
    self._discount = 1.

  @property
  def name(self):
    return 'Mujoban'

  @property
  def root_entity(self):
    return self._arena

  def _regenerate_positions(self):
    self._object_positions = self._arena.find_token_grid_positions(
        [mujoban_level.TARGET_CHAR, mujoban_level.BOX_CHAR])
    self._box_positions = self._arena.grid_to_world_positions(
        self._object_positions[mujoban_level.BOX_CHAR])
    target_grid_positions = self._object_positions[mujoban_level.TARGET_CHAR]
    self._target_positions = self._arena.grid_to_world_positions(
        target_grid_positions)

    for idx in range(len(self._target_positions)):
      target_grid_position = target_grid_positions[idx]
      grid_y, grid_x = target_grid_position
      self._arena.maze.variations_layer[grid_y, grid_x] = _FLOOR_GAP_CHAR

  def initialize_episode_mjcf(self, random_state):
    self._arena.regenerate()

    # Clear existing targets and boxes
    for target in self._targets:
      target.detach()
    self._targets = []
    for box in self._boxes:
      box.detach()
    self._boxes = []
    self._arena.mjcf_model.contact.remove('pair')

    for _ in range(self._maze.num_targets):
      target = MujobanPad(
          size=self._arena.xy_scale,
          height=0,
          detection_tolerance=self._detection_tolerance)
      self._arena.attach(target)
      self._targets.append(target)

    for _ in range(self._maze.num_boxes):
      box = self._box_prop
      if not box:
        box = BoxWithSites(half_lengths=self._box_size)
        box.geom.mass = _BOX_MASS
      box.geom.rgba = [0, 0, 0, 1]  # Will be randomized for each episode.
      frame = self._arena.attach(box)
      frame.add('joint', type='slide', axis=[1, 0, 0], name='x_slider')
      frame.add('joint', type='slide', axis=[0, 1, 0], name='y_slider')
      frame.add('joint', type='slide', axis=[0, 0, 1], name='z_slider')
      self._boxes.append(box)
      for target in self._targets:
        target.register_box(box)

      # Reduce the friction between box and ground.
      ground_geom = self._arena.mjcf_model.find('geom', 'ground')
      self._arena.mjcf_model.contact.add(
          'pair',
          geom1=box.geom,
          geom2=ground_geom,
          condim=6,
          friction=[
              _BOX_FRICTION[0], _BOX_FRICTION[0], _BOX_FRICTION[1],
              _BOX_FRICTION[2], _BOX_FRICTION[2]
          ])

    # Set box masses.
    for box in self._boxes:
      box.geom.mass = _BOX_MASS
      box.geom.rgba[:] = _BOX_RGBA

    for target in self._targets:
      target.rgba[:] = _TARGET_RGBA
      target.pressed_rgba[:] = _PRESSED_TARGET_RGBA

    if self._with_grid_pegs:
      if self._peg_body is not None:
        self._peg_body.remove()

      self._peg_body = self._arena.mjcf_model.worldbody.add('body')
      for y in range(self._arena.maze.height - 1):
        for x in range(self._arena.maze.width - 1):
          peg_x, peg_y, _ = self._arena.grid_to_world_positions(
              [[x + 0.5, y + 0.5]])[0]
          self._peg_body.add(
              'geom', type='box',
              size=[_PEG_SIZE / np.sqrt(2),
                    _PEG_SIZE / np.sqrt(2),
                    _PEG_HEIGHT / 2],
              pos=[peg_x, peg_y, _PEG_HEIGHT / 2],
              quat=[np.cos(_PEG_ANGLE / 2), 0, 0, np.sin(_PEG_ANGLE / 2)],
              rgba=_PEG_RGBA)

  def initialize_episode(self, physics, random_state):
    self._first_step = True
    self._was_activated = [False] * len(self._targets)
    self._is_solved = False
    self._discount = 1.

    self._walker.reinitialize_pose(physics, random_state)
    spawn_position = self._arena.spawn_positions[0]
    spawn_rotation = random_state.uniform(-np.pi, np.pi)
    spawn_quat = np.array(
        [np.cos(spawn_rotation / 2), 0, 0,
         np.sin(spawn_rotation / 2)])
    self._walker.shift_pose(
        physics, [spawn_position[0], spawn_position[1], 0.0], spawn_quat)

    for box, box_xy_position in zip(self._boxes, self._box_positions):
      # Position at the middle of a maze cell.
      box_position = np.array(
          [box_xy_position[0], box_xy_position[1], self._box_size[2]])

      # Commit the box's final pose.
      box.set_pose(physics, position=box_position, quaternion=[1., 0., 0., 0.])

    for target, target_position in zip(self._targets, self._target_positions):
      target.set_pose(physics, position=target_position)
      target.reset(physics)

    self._update_entity_pixel_layers(physics)

  def before_step(self, physics, actions, random_state):
    if isinstance(actions, list):
      actions = np.concatenate(actions)
    super(Mujoban, self).before_step(physics, actions, random_state)
    if self._first_step:
      self._first_step = False
    else:
      self._was_activated = [target.activated for target in self._targets]

  def _get_object_positions_in_grid(self, physics):
    box_positions = self._arena.world_to_grid_positions(
        [physics.bind(box.geom).xpos for box in self._boxes])
    walker_position = self._arena.world_to_grid_positions(
        [physics.bind(self._walker.root_body).xpos])[0]

    return box_positions, walker_position

  def _update_entity_pixel_layers(self, physics):
    """Updates the pixel observation and both layered representations.

    Mujoban offers 3 grid representations of the world:
    * the pixel layer: this is a grid representations with an RGB value at
      each grid point;
    * the entity layer: this is a grid representation with a character at
      each grid point. This representation hides information since if Sokoban
      or a box are over a target, then the target is occluded. This is the
      official entity layer used by arenas which is based on dm_control labmaze;
    * the full entity layer: this is a grid represention with a boolean vector
      of length 4 at each grid point. The first value is `True` iff there is a
      wall at this location. The second value is `True` iff there is a target at
      this location. The third value is for Sokoban, and fourth value is for
      boxes. Note that this is not a one-hot encoding since Sokoban or a box
      can share the same location as a target.

    Args:
      physics: a Mujoco physics object.

    Raises:
      RuntimeError: if a box or walker are overlapping with a wall.
    """
    # The entity layer from the maze is a string that shows the maze at the
    # *beginning* of the level. This is fixed throughout an episode.
    entity_layer = self._maze.entity_layer.copy()
    box_positions, walker_position = self._get_object_positions_in_grid(physics)
    # round positions to snap to grid.
    box_positions, walker_position = _round_positions(
        box_positions, walker_position, self._last_walker_position)

    # setup pixel layer
    map_size = entity_layer.shape
    pixel_layer = np.ndarray(map_size + (3,), dtype='uint8')
    pixel_layer.fill(128)
    # setup full entity layer
    full_entity_layer = np.zeros(map_size + (4,), dtype='bool')
    # remove boxes and agent
    entity_layer[entity_layer == mujoban_level.BOX_CHAR] = '.'
    entity_layer[entity_layer == 'P'] = '.'
    # draw empty space and goals
    pixel_layer[entity_layer == '.'] = [0, 0, 0]
    pixel_layer[entity_layer == 'G'] = [255, 0, 0]
    full_entity_layer[:, :, _WALL_LAYER] = True
    full_entity_layer[:, :, _WALL_LAYER][entity_layer == '.'] = False
    full_entity_layer[:, :, _WALL_LAYER][entity_layer == 'G'] = False
    full_entity_layer[:, :, _TARGET_LAYER][entity_layer == 'G'] = True

    # update boxes
    for pos in box_positions:
      # to ensure we are not changing the walls.
      if entity_layer[pos[0], pos[1]] == '*':
        raise RuntimeError('Box and wall positions are overlapping and this ',
                           'should not happen. It requires investigation and ',
                           'and fixing.')
      # the entity layer has no representation of box on goal.
      entity_layer[pos[0], pos[1]] = mujoban_level.BOX_CHAR
      if np.array_equal(pixel_layer[pos[0], pos[1]], [255, 0, 0]):
        pixel_layer[pos[0], pos[1]] = [0, 255, 0]  # box on goal
      else:
        pixel_layer[pos[0], pos[1]] = [255, 255, 0]
      full_entity_layer[pos[0], pos[1], _BOX_LAYER] = True

    # update player
    if entity_layer[walker_position[0], walker_position[1]] == '*':
      raise RuntimeError('Walker and wall positions are overlapping and this ',
                         'should have not happen. It requires investigation ',
                         'and fixing.')

    entity_layer[walker_position[0], walker_position[1]] = 'P'
    pixel_layer[walker_position[0], walker_position[1]] = 0, 0, 255
    full_entity_layer[
        walker_position[0], walker_position[1], _SOKOBAN_LAYER] = True

    self._last_walker_position = walker_position
    self._entity_layer = entity_layer
    self._pixel_layer = pixel_layer
    self._full_entity_layer = full_entity_layer

  def after_step(self, physics, random_state):
    super(Mujoban, self).after_step(physics, random_state)
    for box in self._boxes:
      physics.bind(box.geom).rgba = _BOX_RGBA
    for target in self._targets:
      if target.activated:
        target.activator.rgba = _BOX_PRESSED_RGBA
    self._update_entity_pixel_layers(physics)
    self._is_solved = all([target.activated for target in self._targets])
    if self._is_solved:
      self._discount = 0.

  def get_reward(self, physics):
    reward = 0.0
    for target, was_activated in zip(self._targets, self._was_activated):
      if target.activated and not was_activated:
        reward += self._box_on_target_reward
      elif was_activated and not target.activated:
        reward -= self._box_on_target_reward
    if self._is_solved:
      reward += self._level_solved_reward
    return reward

  def get_discount(self, physics):
    return self._discount

  def should_terminate_episode(self, physics):
    is_dead = self._walker.aliveness(physics) < _ALIVE_THRESHOLD
    return self._is_solved or is_dead

  def get_reward_spec(self):
    return specs.ArraySpec(shape=[], dtype=np.float32)

  @property
  def task_observables(self):
    return self._task_observables
