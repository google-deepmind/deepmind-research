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
"""Reward function for the fusion environment."""

import abc
import collections
import functools
from typing import Callable, Dict, List, Optional, Text, Tuple, Union

from absl import logging
import dataclasses
import numpy as np

from fusion_tcv import combiners
from fusion_tcv import fge_state
from fusion_tcv import named_array
from fusion_tcv import targets as targets_lib
from fusion_tcv import transforms


class AbstractMeasure(abc.ABC):

  @abc.abstractmethod
  def __call__(self, targets: List[targets_lib.Target]) -> List[float]:
    """Returns a list of error measures."""


class AbsDist(AbstractMeasure):
  """Return the absolute distance between the actual and target."""

  @staticmethod
  def __call__(targets: List[targets_lib.Target]) -> List[float]:
    return [abs(t.actual - t.target) for t in targets]


@dataclasses.dataclass(frozen=True)
class MeasureDetails:
  min: float
  mean: float
  max: float


@dataclasses.dataclass
class RewardDetails:
  reward: float  # 0-1 reward value.
  weighted: float  # Should sum to < 0-1.
  weight: float
  measure: Optional[MeasureDetails] = None


class AbstractReward(abc.ABC):
  """Abstract reward class."""

  @abc.abstractmethod
  def reward(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray,
      ) -> Tuple[float, Dict[Text, List[RewardDetails]]]:
    """Returns the reward and log dict as a function of the penalty term."""

  @abc.abstractmethod
  def terminal_reward(self) -> float:
    """Returns the reward if the simulator crashed."""


WeightFn = Callable[[named_array.NamedArray], float]
WeightOrFn = Union[float, WeightFn]


@dataclasses.dataclass
class Component:
  target: targets_lib.AbstractTarget
  transforms: List[transforms.AbstractTransform]
  measure: AbstractMeasure = dataclasses.field(default_factory=AbsDist)
  weight: Union[WeightOrFn, List[WeightOrFn]] = 1
  name: Optional[str] = None


class Reward(AbstractReward):
  """Combines a bunch of reward components into a single reward.

  The component parts are applied in the order: target, measure, transform.
  - Targets represent some error value as one or more pair of values
    (target, actual), usually with some meaningful physical unit (eg distance,
    volts, etc).
  - Measures combine the (target, actual) into a single float, for example
    absolute distance, for each error value.
  - Transforms can make arbitrary conversions, but one of them must change from
    the arbitrary (often meaningful) scale to a reward in the 0-1 range.
  - Combiners are a special type of transform that reduces a vector of values
    down to a single value. The combiner can be skipped if the target only
    outputs a single value, or if you want a vector of outputs for the final
    combiner.
  - The component weights are passed to the final combiner, and must match the
    number of outputs for that component.
  """

  def __init__(self,
               components: List[Component],
               combiner: combiners.AbstractCombiner,
               terminal_reward: float = -5,
               reward_scale: float = 0.01):
    self._components = components
    self._combiner = combiner
    self._terminal_reward = terminal_reward
    self._reward_scale = reward_scale

    self._weights = []
    component_count = collections.Counter()
    for component in self._components:
      num_outputs = component.target.outputs
      for transform in component.transforms:
        if transform.outputs is not None:
          num_outputs = transform.outputs
      if not isinstance(component.weight, list):
        component.weight = [component.weight]
      if len(component.weight) != num_outputs:
        name = component.name or component.target.name
        raise ValueError(f"Wrong number of weights for '{name}': got:"
                         f" {len(component.weight)}, expected: {num_outputs}")
      self._weights.extend(component.weight)

  def terminal_reward(self) -> float:
    return self._terminal_reward * self._reward_scale

  def reward(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray,
      ) -> Tuple[float, Dict[Text, List[RewardDetails]]]:
    values = []
    weights = [weight(references) if callable(weight) else weight
               for weight in self._weights]
    reward_dict = collections.defaultdict(list)
    for component in self._components:
      name = component.name or component.target.name
      num_outputs = len(component.weight)
      component_weights = weights[len(values):(len(values) + num_outputs)]
      try:
        target = component.target(voltages, state, references)
      except targets_lib.TargetError:
        logging.exception("Target failed.")
        # Failed turns into minimum reward.
        measure = [987654321] * num_outputs
        transformed = [0] * num_outputs
      else:
        measure = component.measure(target)
        transformed = functools.reduce(
            (lambda e, fn: fn(e)), component.transforms, measure)
      assert len(transformed) == num_outputs
      for v in transformed:
        if not np.isnan(v) and not 0 <= v <= 1:
          raise ValueError(f"The transformed value in {name} is invalid: {v}")
      values.extend(transformed)
      for weight, value in zip(component_weights, transformed):
        measure = [m for m in measure if not np.isnan(m)] or [float("nan")]
        reward_dict[name].append(RewardDetails(
            value, weight * value * self._reward_scale,
            weight if not np.isnan(value) else 0,
            MeasureDetails(
                min(measure), sum(measure) / len(measure), max(measure))))

    sum_weights = sum(sum(d.weight for d in detail)
                      for detail in reward_dict.values())
    for reward_details in reward_dict.values():
      for detail in reward_details:
        detail.weighted /= sum_weights

    final_combined = self._combiner(values, weights)
    assert len(final_combined) == 1
    return final_combined[0] * self._reward_scale, reward_dict
