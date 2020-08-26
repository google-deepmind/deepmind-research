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

"""A global registry of constructors for board game environments."""


from dm_control.utils import containers

_ALL_CONSTRUCTORS = containers.TaggedTasks(allow_overriding_keys=False)

add = _ALL_CONSTRUCTORS.add
get_constructor = _ALL_CONSTRUCTORS.__getitem__
get_all_names = _ALL_CONSTRUCTORS.keys
get_tags = _ALL_CONSTRUCTORS.tags
get_names_by_tag = _ALL_CONSTRUCTORS.tagged

# This disables the check that prevents the same task constructor name from
# being added to the container more than once. This is done in order to allow
# individual task modules to be reloaded without also reloading `registry.py`
# first (e.g. when "hot-reloading" environments in domain explorer).


def done_importing_tasks():
  _ALL_CONSTRUCTORS.allow_overriding_keys = True
