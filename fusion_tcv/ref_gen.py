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
"""Generators for References vector."""

import abc
import copy
from typing import List, Optional

import dataclasses

from fusion_tcv import named_array
from fusion_tcv import shape as shape_lib
from fusion_tcv import tcv_common


class AbstractReferenceGenerator(abc.ABC):
  """Abstract class for generating the reference signal."""

  @abc.abstractmethod
  def reset(self) -> named_array.NamedArray:
    """Resets the class for a new episode and returns the first reference."""

  @abc.abstractmethod
  def step(self) -> named_array.NamedArray:
    """Returns the reference signal."""


@dataclasses.dataclass
class LinearTransition:
  reference: named_array.NamedArray  # Reference at which to end the transition.
  transition_steps: int  # Number of intermediate steps between the shapes.
  steady_steps: int  # Number of steps in the steady state.


class LinearTransitionReferenceGenerator(AbstractReferenceGenerator):
  """A base class for generating references that are a series of transitions."""

  def __init__(self, start_offset: int = 0):
    self._last_ref = None
    self._reset_counters()
    self._start_offset = start_offset

  @abc.abstractmethod
  def _next_transition(self) -> LinearTransition:
    """Override this in the subclass."""

  def reset(self) -> named_array.NamedArray:
    self._last_ref = None
    self._reset_counters()
    for _ in range(self._start_offset):
      self.step()
    return self.step()

  def _reset_counters(self):
    self._steady_step = 0
    self._transition_step = 0
    self._transition = None

  def step(self) -> named_array.NamedArray:
    if (self._transition is None or
        self._steady_step == self._transition.steady_steps):
      if self._transition is not None:
        self._last_ref = self._transition.reference
      self._reset_counters()
      self._transition = self._next_transition()
      # Ensure at least one steady step in middle transitions.
      # If we would like this to not have to be true, we need to change the
      # logic below which assumes there is at least one step in the steady
      # phase.
      assert self._transition.steady_steps > 0

    assert self._transition is not None  # to make pytype happy
    transition_steps = self._transition.transition_steps
    if self._last_ref is None:  # No transition at beginning of episode.
      transition_steps = 0

    if self._transition_step < transition_steps:  # In transition phase.
      self._transition_step += 1
      a = self._transition_step / (self._transition.transition_steps + 1)  # pytype: disable=attribute-error
      return self._last_ref.names.named_array(
          self._last_ref.array * (1 - a) + self._transition.reference.array * a)  # pytype: disable=attribute-error
    else:  # In steady phase.
      self._steady_step += 1
      return copy.deepcopy(self._transition.reference)


class FixedReferenceGenerator(LinearTransitionReferenceGenerator):
  """Generates linear transitions from a fixed set of references."""

  def __init__(self, transitions: List[LinearTransition],
               start_offset: int = 0):
    self._transitions = transitions
    self._current_transition = 0
    super().__init__(start_offset=start_offset)

  def reset(self) -> named_array.NamedArray:
    self._current_transition = 0
    return super().reset()

  def _next_transition(self) -> LinearTransition:
    if self._current_transition == len(self._transitions):
      # Have gone through all of the transitions. Return the final reference
      # for a very long time.
      return LinearTransition(steady_steps=50000, transition_steps=0,
                              reference=self._transitions[-1].reference)
    self._current_transition += 1
    return copy.deepcopy(self._transitions[self._current_transition - 1])


@dataclasses.dataclass
class TimedTransition:
  steady_steps: int  # Number of steps to hold the shape.
  transition_steps: int  # Number of steps to transition.


@dataclasses.dataclass
class ParametrizedShapeTimedTarget:
  """RZIP condition with a timestep attached."""
  shape: shape_lib.Shape
  timing: TimedTransition


class PresetShapePointsReferenceGenerator(FixedReferenceGenerator):
  """Generates a fixed set of shape points."""

  def __init__(
      self, targets: List[ParametrizedShapeTimedTarget], start_offset: int = 0):
    if targets[0].timing.transition_steps != 0:
      raise ValueError("Invalid first timing, transition must be 0, not "
                       f"{targets[0].timing.transition_steps}")
    transitions = []
    for target in targets:
      transitions.append(LinearTransition(
          steady_steps=target.timing.steady_steps,
          transition_steps=target.timing.transition_steps,
          reference=target.shape.canonical().gen_references()))
    super().__init__(transitions, start_offset=start_offset)


class ShapeFromShot(PresetShapePointsReferenceGenerator):
  """Generate shapes from EPFL references."""

  def __init__(
      self, time_slices: List[shape_lib.ReferenceTimeSlice],
      start: Optional[float] = None):
    """Given a series of time slices, start from time_slice.time==start."""
    if start is None:
      start = time_slices[0].time
    dt = 1e-4
    targets = []
    time_slices = shape_lib.canonicalize_reference_series(time_slices)
    prev = None
    for i, ref in enumerate(time_slices):
      assert prev is None or prev.hold < ref.time
      if ref.time < start:
        continue
      if prev is None and start != ref.time:
        raise ValueError("start must be one of the time slice times.")

      steady = (max(1, int((ref.hold - ref.time) / dt))
                if i < len(time_slices) - 1 else 100000)
      transition = (0 if prev is None else
                    (int((ref.time - prev.time) / dt) -
                     max(1, int((prev.hold - prev.time) / dt))))

      targets.append(ParametrizedShapeTimedTarget(
          shape=ref.shape,
          timing=TimedTransition(
              steady_steps=steady, transition_steps=transition)))
      prev = ref

    assert targets
    super().__init__(targets)


@dataclasses.dataclass
class RZIpTarget:
  r: float
  z: float
  ip: float


def make_symmetric_multidomain_rzip_reference(
    target: RZIpTarget) -> named_array.NamedArray:
  """Generate multi-domain rzip references."""
  refs = tcv_common.REF_RANGES.new_named_array()
  refs["R"] = (target.r, target.r)
  refs["Z"] = (target.z, -target.z)
  refs["Ip"] = (target.ip, target.ip)
  return refs

