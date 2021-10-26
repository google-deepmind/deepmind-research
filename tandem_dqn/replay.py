# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay components for DQN-type agents."""

import collections
import typing
from typing import Any, Callable, Generic, Iterable, List, Mapping, Optional, Sequence, Text, Tuple, TypeVar

import dm_env
import numpy as np
import snappy

from tandem_dqn import parts

CompressedArray = Tuple[bytes, Tuple, np.dtype]

# Generic replay structure: Any flat named tuple.
ReplayStructure = TypeVar('ReplayStructure', bound=Tuple[Any, ...])


class Transition(typing.NamedTuple):
  s_tm1: Optional[np.ndarray]
  a_tm1: Optional[parts.Action]
  r_t: Optional[float]
  discount_t: Optional[float]
  s_t: Optional[np.ndarray]
  a_t: Optional[parts.Action] = None
  mc_return_tm1: Optional[float] = None


class TransitionReplay(Generic[ReplayStructure]):
  """Uniform replay, with circular buffer storage for flat named tuples."""

  def __init__(self,
               capacity: int,
               structure: ReplayStructure,
               random_state: np.random.RandomState,
               encoder: Optional[Callable[[ReplayStructure], Any]] = None,
               decoder: Optional[Callable[[Any], ReplayStructure]] = None):
    self._capacity = capacity
    self._structure = structure
    self._random_state = random_state
    self._encoder = encoder or (lambda s: s)
    self._decoder = decoder or (lambda s: s)

    self._storage = [None] * capacity
    self._num_added = 0

  def add(self, item: ReplayStructure) -> None:
    """Adds single item to replay."""
    self._storage[self._num_added % self._capacity] = self._encoder(item)
    self._num_added += 1

  def get(self, indices: Sequence[int]) -> List[ReplayStructure]:
    """Retrieves items by indices."""
    return [self._decoder(self._storage[i]) for i in indices]

  def sample(self, size: int) -> ReplayStructure:
    """Samples batch of items from replay uniformly, with replacement."""
    indices = self._random_state.choice(self.size, size=size, replace=True)
    samples = self.get(indices)
    transposed = zip(*samples)
    stacked = [np.stack(xs, axis=0) for xs in transposed]
    return type(self._structure)(*stacked)  # pytype: disable=not-callable

  @property
  def size(self) -> int:
    """Number of items currently contained in replay."""
    return min(self._num_added, self._capacity)

  @property
  def capacity(self) -> int:
    """Total capacity of replay (max number of items stored at any one time)."""
    return self._capacity

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves replay state as a dictionary (e.g. for serialization)."""
    return {
        'storage': self._storage,
        'num_added': self._num_added,
    }

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets replay state from a (potentially de-serialized) dictionary."""
    self._storage = state['storage']
    self._num_added = state['num_added']


class TransitionAccumulatorWithMCReturn:
  """Accumulates timesteps to transitions with MC returns."""

  def __init__(self):
    self._transitions = collections.deque()
    self.reset()

  def step(self, timestep_t: dm_env.TimeStep,
           a_t: parts.Action) -> Iterable[Transition]:
    """Accumulates timestep and resulting action, maybe yields transitions."""
    if timestep_t.first():
      self.reset()

    # There are no transitions on the first timestep.
    if self._timestep_tm1 is None:
      assert self._a_tm1 is None
      if not timestep_t.first():
        raise ValueError('Expected FIRST timestep, got %s.' % str(timestep_t))
      self._timestep_tm1 = timestep_t
      self._a_tm1 = a_t
      return  # Empty iterable.

    self._transitions.append(
        Transition(
            s_tm1=self._timestep_tm1.observation,
            a_tm1=self._a_tm1,
            r_t=timestep_t.reward,
            discount_t=timestep_t.discount,
            s_t=timestep_t.observation,
            a_t=a_t,
            mc_return_tm1=None,
        ))

    self._timestep_tm1 = timestep_t
    self._a_tm1 = a_t

    if timestep_t.last():
      # Annotate all episode transitions with their MC returns.
      mc_return = 0
      mc_transitions = []
      while self._transitions:
        transition = self._transitions.pop()
        mc_return = transition.discount_t * mc_return + transition.r_t
        mc_transitions.append(transition._replace(mc_return_tm1=mc_return))
      for transition in reversed(mc_transitions):
        yield transition

    else:
      # Wait for episode end before yielding anything.
      return

  def reset(self) -> None:
    """Resets the accumulator. Following timestep is expected to be FIRST."""
    self._transitions.clear()
    self._timestep_tm1 = None
    self._a_tm1 = None


def compress_array(array: np.ndarray) -> CompressedArray:
  """Compresses a numpy array with snappy."""
  return snappy.compress(array), array.shape, array.dtype


def uncompress_array(compressed: CompressedArray) -> np.ndarray:
  """Uncompresses a numpy array with snappy given its shape and dtype."""
  compressed_array, shape, dtype = compressed
  byte_string = snappy.uncompress(compressed_array)
  return np.frombuffer(byte_string, dtype=dtype).reshape(shape)
