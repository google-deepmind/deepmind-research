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

"""Functions that build representative tasks."""

from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.mocap import loader as mocap_loader
from dm_control.locomotion.walkers import cmu_humanoid

from catch_carry import ball_toss
from catch_carry import warehouse


def build_vision_warehouse(random_state=None):
  """Build canonical 4-pedestal, 2-prop task."""

  # Build a position-controlled CMU humanoid walker.
  walker = cmu_humanoid.CMUHumanoidPositionControlled(
      observable_options={'egocentric_camera': dict(enabled=True)})

  # Build the task.
  size_distribution = distributions.Uniform(low=0.75, high=1.25)
  mass_distribution = distributions.Uniform(low=2, high=7)
  prop_resizer = mocap_loader.PropResizer(size_factor=size_distribution,
                                          mass=mass_distribution)
  task = warehouse.PhasedBoxCarry(
      walker=walker,
      num_props=2,
      num_pedestals=4,
      proto_modifier=prop_resizer,
      negative_reward_on_failure_termination=True)

  # return the environment
  return composer.Environment(
      time_limit=15,
      task=task,
      random_state=random_state,
      strip_singleton_obs_buffer_dim=True,
      max_reset_attempts=float('inf'))


def build_vision_toss(random_state=None):
  """Build canonical ball tossing task."""

  # Build a position-controlled CMU humanoid walker.
  walker = cmu_humanoid.CMUHumanoidPositionControlled(
      observable_options={'egocentric_camera': dict(enabled=True)})

  # Build the task.
  size_distribution = distributions.Uniform(low=0.95, high=1.5)
  mass_distribution = distributions.Uniform(low=2, high=4)
  prop_resizer = mocap_loader.PropResizer(size_factor=size_distribution,
                                          mass=mass_distribution)
  task = ball_toss.BallToss(
      walker=walker,
      proto_modifier=prop_resizer,
      negative_reward_on_failure_termination=True,
      priority_friction=True,
      bucket_offset=3.,
      y_range=0.5,
      toss_delay=1.5,
      randomize_init=True)

  # return the environment
  return composer.Environment(
      time_limit=6,
      task=task,
      random_state=random_state,
      strip_singleton_obs_buffer_dim=True,
      max_reset_attempts=float('inf'))
