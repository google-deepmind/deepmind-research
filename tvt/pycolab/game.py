# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Pycolab Game interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class AbstractGame(object):
  """Abstract base class for Pycolab games."""

  @abc.abstractmethod
  def __init__(self, rng, **settings):
    """Initialize the game."""

  @abc.abstractproperty
  def num_actions(self):
    """Number of possible actions in the game."""

  @abc.abstractproperty
  def colours(self):
    """Symbol to colour map for the game."""

  @abc.abstractmethod
  def make_episode(self):
    """Factory method for generating new episodes of the game."""
