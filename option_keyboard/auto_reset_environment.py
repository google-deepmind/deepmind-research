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
"""Auto-resetting environment base class.

The environment API states that stepping an environment after a LAST timestep
should return the first timestep of a new episode.

However, environment authors sometimes don't spot this part or find it awkward
to implement. This module contains a class that helps implement the reset
behaviour.
"""

import abc
import dm_env


class Base(dm_env.Environment):
  """This class implements the required `step()` and `reset()` methods.

  It instead requires users to implement `_step()` and `_reset()`. This class
  handles the reset behaviour automatically when it detects a LAST timestep.
  """

  def __init__(self):
    self._reset_next_step = True

  @abc.abstractmethod
  def _reset(self):
    """Returns a `timestep` namedtuple as per the regular `reset()` method."""

  @abc.abstractmethod
  def _step(self, action):
    """Returns a `timestep` namedtuple as per the regular `step()` method."""

  def reset(self):
    self._reset_next_step = False
    return self._reset()

  def step(self, action):
    if self._reset_next_step:
      return self.reset()
    timestep = self._step(action)
    self._reset_next_step = timestep.last()
    return timestep
