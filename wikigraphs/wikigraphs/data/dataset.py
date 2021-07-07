# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
#
# WikiGraphs is licensed under the terms of the Creative Commons
# Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.
#
# WikiText-103 data (unchanged) is licensed by Salesforce.com, Inc. under the
# terms of the Creative Commons Attribution-ShareAlike 4.0 International
# (CC BY-SA 4.0) license. You can find details about CC BY-SA 4.0 at:
#
#     https://creativecommons.org/licenses/by-sa/4.0/legalcode
#
# Freebase data is licensed by Google LLC under the terms of the Creative
# Commons CC BY 4.0 license. You may obtain a copy of the License at:
#
#     https://creativecommons.org/licenses/by/4.0/legalcode
#
# ==============================================================================
"""Base class of the datasets."""

import abc
from typing import Any, Iterator


class Dataset(abc.ABC):
  """Base class for all datasets.

  All sub-classes should define `_load_data()` where an iterator
  `self._data_iter` should be instantiated that iterates over the dataset.
  """

  def __init__(self):
    """Constructor."""
    self._data_iter = None  # An iterator produced by `self._load_data`.

  @abc.abstractmethod
  def _load_data(self) -> Iterator[Any]:
    """Prepare data for another pass through the dataset.

    This method should return a generator in a child class.
    """

  def __next__(self):
    return next(self._data_iter)

  def __iter__(self):
    self._data_iter = self._load_data()
    return self
