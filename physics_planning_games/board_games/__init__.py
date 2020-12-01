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

"""Physically-grounded board game environments."""

from dm_control import composer as _composer

from physics_planning_games.board_games import go as _go
from physics_planning_games.board_games import tic_tac_toe as _tic_tac_toe
from physics_planning_games.board_games._internal import registry as _registry

_registry.done_importing_tasks()

ALL = tuple(_registry.get_all_names())
TAGS = tuple(_registry.get_tags())


def get_environments_by_tag(tag):
  """Returns the names of all environments matching a given tag.

  Args:
    tag: A string from `TAGS`.

  Returns:
    A tuple of environment names.
  """
  return tuple(_registry.get_names_by_tag(tag))


def load(environment_name,
         env_kwargs=None,
         seed=None,
         time_limit=float('inf'),
         strip_singleton_obs_buffer_dim=False):
  """Loads an environment from board_games.

  Args:
    environment_name: String, the name of the environment to load. Must be in
      `ALL`.
    env_kwargs: extra params to pass to task creation.
    seed: Optional, either an int seed or an `np.random.RandomState`
      object. If None (default), the random number generator will self-seed
      from a platform-dependent source of entropy.
    time_limit: (optional) A float, the time limit in seconds beyond which an
      episode is forced to terminate.
    strip_singleton_obs_buffer_dim: (optional) A boolean, if `True`,
      the array shape of observations with `buffer_size == 1` will not have a
      leading buffer dimension.

  Returns:
    An instance of `composer.Environment`.
  """
  if env_kwargs is not None:
    task = _registry.get_constructor(environment_name)(**env_kwargs)
  else:
    task = _registry.get_constructor(environment_name)()
  return _composer.Environment(
      task=task,
      time_limit=time_limit,
      strip_singleton_obs_buffer_dim=strip_singleton_obs_buffer_dim,
      random_state=seed)
