# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The environment definitions used for our experiments."""

from fusion_tcv import environment
from fusion_tcv import references
from fusion_tcv import rewards_used


# Used in TCV#70915
def fundamental_capability() -> environment.Environment:
  return environment.Environment(
      shot_condition=environment.ShotCondition(70166, 0.0872),
      reward=rewards_used.FUNDAMENTAL_CAPABILITY,
      reference_generator=references.fundamental_capability(),
      max_episode_length=10000)


# Used in TCV#70920
def elongation() -> environment.Environment:
  return environment.Environment(
      shot_condition=environment.ShotCondition(70166, 0.45),
      reward=rewards_used.ELONGATION,
      reference_generator=references.elongation(),
      max_episode_length=5000)


# Used in TCV#70600
def iter() -> environment.Environment:  # pylint: disable=redefined-builtin
  return environment.Environment(
      shot_condition=environment.ShotCondition(70392, 0.0872),
      reward=rewards_used.ITER,
      reference_generator=references.iter(),
      max_episode_length=1000)


# Used in TCV#70457
def negative_triangularity() -> environment.Environment:
  return environment.Environment(
      shot_condition=environment.ShotCondition(70166, 0.45),
      reward=rewards_used.NEGATIVE_TRIANGULARITY,
      reference_generator=references.negative_triangularity(),
      max_episode_length=5000)


# Used in TCV#70755
def snowflake() -> environment.Environment:
  return environment.Environment(
      shot_condition=environment.ShotCondition(70166, 0.0872),
      reward=rewards_used.SNOWFLAKE,
      reference_generator=references.snowflake(),
      max_episode_length=10000)


# Used in TCV#69545
def droplet() -> environment.Environment:
  return environment.Environment(
      shot_condition=environment.ShotCondition(69198, 0.418),
      reward=rewards_used.DROPLETS,
      reference_generator=references.droplet(),
      max_episode_length=2000)
