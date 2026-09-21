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
"""Reward combiners."""

import abc
import math
from typing import List, Optional, Tuple

import dataclasses
import numpy as np
from scipy import special

from fusion_tcv import targets


class AbstractCombiner(targets.AbstractTarget):
  """Combines a set of rewards, possibly weighted."""

  @abc.abstractmethod
  def __call__(self, values: List[float],  # pytype: disable=signature-mismatch  # overriding-return-type-checks
               weights: Optional[List[float]] = None) -> List[float]:
    """Combines a set of rewards, possibly weighted."""

  @property
  def outputs(self) -> int:
    """All combiners return exactly one value, even if it's NaN."""
    return 1

  @staticmethod
  def _clean_values_weights(
      values: List[float],
      weights: Optional[List[float]] = None) -> Tuple[List[float], List[float]]:
    """Validate the values and weights, and if no weights, return equal."""
    if weights is None:
      weights = [1] * len(values)
    else:
      if len(values) != len(weights):
        raise ValueError("Number of weights don't match values. "
                         f"values: {len(values)}, weights: {len(weights)}")
      for w in weights:
        if w < 0:
          raise ValueError(f"Weights must be >=0: {w}")

    new_values_weights = [(v, w) for v, w in zip(values, weights)
                          if not np.isnan(v) and w > 0]
    return tuple(zip(*new_values_weights)) if new_values_weights else ([], [])


class Mean(AbstractCombiner):
  """Take the weighted mean of the values.

  Ignores NaNs and values with weight 0.
  """

  def __call__(self, values: List[float],
               weights: Optional[List[float]] = None) -> List[float]:
    values, weights = self._clean_values_weights(values, weights)
    if not values:
      return [float("nan")]
    return [sum(r * w for r, w in zip(values, weights)) / sum(weights)]


def _multiply(values, weights, mean):
  """Multiplies the values taking care to validate the weights.

  Defines 0^0 = 1 so a reward with no weight is "off" even if the value is 0.

  Args:
    values: The reward values.
    weights: The reward weights.
    mean: If true, divides by the sum of the weights (computes the geometric
      mean).

  Returns:
    Product of v^w across the components.
  """
  # If weight and value are both zero, set the value to 1 so that 0^0 = 1.
  values = [1 if (v == 0 and w == 0) else v for (v, w) in zip(values, weights)]
  if any(v == 0 for v in values):
    return [0]
  den = sum(weights) if mean else 1
  return [math.exp(sum(np.log(values) * weights) / den)]


class Multiply(AbstractCombiner):
  """Combine by multiplying the (weighted) values together.

  This is the same as Geometric mean, but without the n^th root taken at the
  end. This means doing poorly on several rewards compounds, rather than
  averages. As such it likely only makes sense after the non-linearities, ie
  where the values are in the 0-1 range, otherwise it'll cause them to increase.
  This is even harsher than Min or SmoothMax(-inf).

  Ignores NaNs and values with weight 0.
  """

  def __call__(self, values: List[float],
               weights: Optional[List[float]] = None) -> List[float]:
    values, weights = self._clean_values_weights(values, weights)
    if not values:
      return [float("nan")]
    return _multiply(values, weights, mean=False)


class GeometricMean(AbstractCombiner):
  """Take the weighted geometric mean of the values.

  Pushes values towards 0, so likely only makes sense after the non-linear
  transforms.

  Ignores NaNs and values with weight 0.
  """

  def __call__(self, values: List[float],
               weights: Optional[List[float]] = None) -> List[float]:
    values, weights = self._clean_values_weights(values, weights)
    if not values:
      return [float("nan")]
    return _multiply(values, weights, mean=True)


class Min(AbstractCombiner):
  """Take the min of the values. Ignores NaNs and values with weight 0."""

  def __call__(self, values: List[float],
               weights: Optional[List[float]] = None) -> List[float]:
    values, _ = self._clean_values_weights(values, weights)
    if not values:
      return [float("nan")]
    return [min(values)]


class Max(AbstractCombiner):
  """Take the max of the values. Ignores NaNs and values with weight 0."""

  def __call__(self, values: List[float],
               weights: Optional[List[float]] = None) -> List[float]:
    values, _ = self._clean_values_weights(values, weights)
    if not values:
      return [float("nan")]
    return [max(values)]


@dataclasses.dataclass(frozen=True)
class LNorm(AbstractCombiner):
  """Take the l-norm of the values.

  Reasonable norm values (assuming normalized):
  - 1: avg of the values
  - 2: euclidean distance metric
  - inf: max value

  Values in between go between the average and max. As the l-norm goes up, the
  result gets closer to the max.

  Normalized means dividing by the max possible distance, such that the units
  still make sense.

  This likely only makes sense before the non-linear transforms. SmoothMax is
  similar but more flexible and understandable.

  Ignores NaNs and values with weight 0.
  """
  norm: float
  normalized: bool = True

  def __call__(self, values: List[float],
               weights: Optional[List[float]] = None) -> List[float]:
    values, _ = self._clean_values_weights(values, weights)
    if not values:
      return [float("nan")]
    lnorm = np.linalg.norm(values, ord=self.norm)
    if self.normalized:
      lnorm /= np.linalg.norm(np.ones(len(values)), ord=self.norm)
    return [float(lnorm)]


@dataclasses.dataclass(frozen=True)
class SmoothMax(AbstractCombiner):
  """Combines component rewards using a smooth maximum.

  https://en.wikipedia.org/wiki/Smooth_maximum
  alpha is the exponent for the smooth max.
  - alpha -> inf: returns the maximum
  - alpha == 0: returns the weighted average
  - alpha -> -inf: returns the minimum
  alpha in between returns values in between.

  Since this varies between min, mean and max, it keeps the existing scale.

  Alpha >= 0 make sense before converting to 0-1, alpha <= 0 make sense after.

  Ignores NaNs and values with weight 0.
  """
  alpha: float

  def __call__(self, values: List[float],
               weights: Optional[List[float]] = None) -> List[float]:
    values, weights = self._clean_values_weights(values, weights)
    if not values:
      return [float("nan")]
    if math.isinf(self.alpha):
      return [max(values) if self.alpha > 0 else min(values)]
    # Compute weights in a numerically-friendly way.
    log_soft_weights = [np.log(w) + c * self.alpha
                        for w, c in zip(weights, values)]
    log_soft_weights -= special.logsumexp(log_soft_weights)
    soft_weights = np.exp(log_soft_weights)
    return Mean()(values, soft_weights)  # weighted mean
