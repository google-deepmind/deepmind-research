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
"""Give names to parts of a numpy array."""

from typing import Iterable, List, Mapping, MutableMapping, Tuple, Union

import numpy as np


def lengths_to_ranges(
    lengths: Mapping[str, int]) -> MutableMapping[str, List[int]]:
  """Eg: {a: 2, b: 3} -> {a: [0, 1], b: [2, 3, 4]} ."""
  ranges = {}
  start = 0
  for key, length in lengths.items():
    ranges[key] = list(range(start, start + length))
    start += length
  return ranges


class NamedRanges:
  """Given a map of {key: count}, give various views into it."""

  def __init__(self, counts: Mapping[str, int]):
    self._ranges = lengths_to_ranges(counts)
    self._size = sum(counts.values())

  def __getitem__(self, name) -> List[int]:
    return self._ranges[name]

  def __contains__(self, name) -> bool:
    return name in self._ranges

  def set_range(self, name: str, value: List[int]):
    """Overwrite or create a custom range, which may intersect with others."""
    self._ranges[name] = value

  def range(self, name: str) -> List[int]:
    return self[name]

  def index(self, name: str) -> int:
    rng = self[name]
    if len(rng) != 1:
      raise ValueError(f"{name} has multiple values")
    return rng[0]

  def count(self, name: str) -> int:
    return len(self[name])

  def names(self) -> Iterable[str]:
    return self._ranges.keys()

  def ranges(self) -> Iterable[Tuple[str, List[int]]]:
    return self._ranges.items()

  def counts(self) -> Mapping[str, int]:
    return {k: len(v) for k, v in self._ranges.items()}

  @property
  def size(self) -> int:
    return self._size

  def named_array(self, array: np.ndarray) -> "NamedArray":
    return NamedArray(array, self)

  def new_named_array(self) -> "NamedArray":
    return NamedArray(np.zeros((self.size,)), self)

  def new_random_named_array(self) -> "NamedArray":
    return NamedArray(np.random.uniform(size=(self.size,)), self)


class NamedArray:
  """Given a numpy array and a NamedRange, access slices by name."""

  def __init__(self, array: np.ndarray, names: NamedRanges):
    if array.shape != (names.size,):
      raise ValueError(f"Wrong sizes: {array.shape} != ({names.size},)")
    self._array = array
    self._names = names

  def __getitem__(
      self, name: Union[str, Tuple[str, Union[int, List[int],
                                              slice]]]) -> np.ndarray:
    """Return a read-only view into the array by name."""
    if isinstance(name, str):
      arr = self._array[self._names[name]]
    else:
      name, i = name
      arr = self._array[np.array(self._names[name])[i]]
    if not np.isscalar(arr):
      # Read-only because it's indexed by an array of potentially non-contiguous
      # indices, which isn't representable as a normal tensor, which forces a
      # copy and therefore writes don't modify the underlying array as expected.
      arr.flags.writeable = False
    return arr

  def __setitem__(
      self, name: Union[str, Tuple[str, Union[int, List[int], slice]]], value):
    """Set one or more values of a range to a value."""
    if isinstance(name, str):
      self._array[self._names[name]] = value
    else:
      name, i = name
      self._array[np.array(self._names[name])[i]] = value

  @property
  def array(self) -> np.ndarray:
    return self._array

  @property
  def names(self) -> NamedRanges:
    return self._names

  def to_dict(self) -> Mapping[str, np.ndarray]:
    return {k: self[k] for k in self._names.names()}
