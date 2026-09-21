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
"""Transforms from actual/target to rewards."""

import abc
import math
from typing import List, Optional

import dataclasses


# Comparison of some of the transforms:
# Order is NegExp, SoftPlus, Sigmoid.
# Over range of good to bad:
# https://www.wolframalpha.com/input/?i=plot+e%5E%28x*ln%280.1%29%29%2C2%2F%281%2Be%5E%28x*ln%2819%29%29%29%2C1%2F%281%2Be%5E%28-+%28-ln%2819%29+-+%28x-1%29*%282*ln%2819%29%29%29%29%29+from+x%3D0+to+2
# When close to good:
# https://www.wolframalpha.com/input/?i=plot+e%5E%28x*ln%280.1%29%29%2C2%2F%281%2Be%5E%28x*ln%2819%29%29%29%2C1%2F%281%2Be%5E%28-+%28-ln%2819%29+-+%28x-1%29*%282*ln%2819%29%29%29%29%29+from+x%3D0+to+0.2


class AbstractTransform(abc.ABC):

  @abc.abstractmethod
  def __call__(self, errors: List[float]) -> List[float]:
    """Transforms target errors into rewards."""

  @property
  def outputs(self) -> Optional[int]:
    return None


def clip(value: float, low: float, high: float) -> float:
  """Clip a value to the range of low - high."""
  if math.isnan(value):
    return value
  assert low <= high
  return max(low, min(high, value))


def scale(v: float, a: float, b: float, c: float, d: float) -> float:
  """Scale a value, v on a line with anchor points a,b to new anchors c,d."""
  v01 = (v - a) / (b - a)
  return c - v01 * (c - d)


def logistic(v: float) -> float:
  """Standard logistic, asymptoting to 0 and 1."""
  v = clip(v, -50, 50)  # Improve numerical stability.
  return 1 / (1 + math.exp(-v))


@dataclasses.dataclass(frozen=True)
class Equal(AbstractTransform):
  """Returns 1 if the error is 0 and not_equal_val otherwise."""
  not_equal_val: float = 0

  def __call__(self, errors: List[float]) -> List[float]:
    out = []
    for err in errors:
      if math.isnan(err):
        out.append(err)
      elif err == 0:
        out.append(1)
      else:
        out.append(self.not_equal_val)
    return out


class Abs(AbstractTransform):
  """Take the absolue value of the error. Does not guarantee 0-1."""

  @staticmethod
  def __call__(errors: List[float]) -> List[float]:
    return [abs(err) for err in errors]


class Neg(AbstractTransform):
  """Negate the error. Does not guarantee 0-1."""

  @staticmethod
  def __call__(errors: List[float]) -> List[float]:
    return [-err for err in errors]


@dataclasses.dataclass(frozen=True)
class Pow(AbstractTransform):
  """Return a power of the error. Does not guarantee 0-1."""
  pow: float

  def __call__(self, errors: List[float]) -> List[float]:
    return [err**self.pow for err in errors]


@dataclasses.dataclass(frozen=True)
class Log(AbstractTransform):
  """Return a log of the error. Does not guarantee 0-1."""
  eps: float = 1e-4

  def __call__(self, errors: List[float]) -> List[float]:
    return [math.log(err + self.eps) for err in errors]


@dataclasses.dataclass(frozen=True)
class ClippedLinear(AbstractTransform):
  """Scales and clips errors, bad to 0, good to 1. If good=0, this is a relu."""
  bad: float
  good: float = 0

  def __call__(self, errors: List[float]) -> List[float]:
    return [clip(scale(err, self.bad, self.good, 0, 1), 0, 1)
            for err in errors]


@dataclasses.dataclass(frozen=True)
class SoftPlus(AbstractTransform):
  """Scales and clips errors, bad to 0.1, good to 1, asymptoting to 0.

  Based on the lower half of the logistic instead of the standard softplus as
  we want it to be bounded from 0 to 1, with the good value being exactly 1.
  Various constants can be chosen to get the softplus to give the desired
  properties, but this is much simpler.
  """
  bad: float
  good: float = 0

  # Constant to set the sharpness/slope of the softplus.
  # Default was chosen such that the good/bad have 1 and 0.1 reward:
  # https://www.wolframalpha.com/input/?i=plot+2%2F%281%2Be%5E%28x*ln%2819%29%29%29+from+x%3D0+to+2
  low: float = -math.log(19)  # -2.9444389791664403

  def __call__(self, errors: List[float]) -> List[float]:
    return [clip(2 * logistic(scale(e, self.bad, self.good, self.low, 0)), 0, 1)
            for e in errors]


@dataclasses.dataclass(frozen=True)
class NegExp(AbstractTransform):
  """Scales and clips errors, bad to 0.1, good to 1, asymptoting to 0.

  This scales the reward in an exponential space. This means there is a sharp
  gradient toward reaching the value of good, flattening out at the value of
  bad. This can be useful for a reward that gives meaningful signal far away,
  but still have a sharp gradient near the true target.
  """
  bad: float
  good: float = 0

  # Constant to set the sharpness/slope of the exponential.
  # Default was chosen such that the good/bad have 1 and 0.1 reward:
  # https://www.wolframalpha.com/input/?i=plot+e%5E%28x*ln%280.1%29%29+from+x%3D0+to+2
  low: float = -math.log(0.1)

  def __call__(self, errors: List[float]) -> List[float]:
    return [clip(math.exp(-scale(e, self.bad, self.good, self.low, 0)), 0, 1)
            for e in errors]


@dataclasses.dataclass(frozen=True)
class Sigmoid(AbstractTransform):
  """Scales and clips errors, bad to 0.05, good to 0.95, asymptoting to 0-1."""
  good: float
  bad: float

  # Constants to set the sharpness/slope of the sigmoid.
  # Defaults were chosen such that the good/bad have 0.95 and 0.05 reward:
  # https://www.wolframalpha.com/input/?i=plot+1%2F%281%2Be%5E%28-+%28-ln%2819%29+-+%28x-1%29*%282*ln%2819%29%29%29%29%29+from+x%3D0+to+2
  high: float = math.log(19)  # +2.9444389791664403
  low: float = -math.log(19)  # -2.9444389791664403

  def __call__(self, errors: List[float]) -> List[float]:
    return [logistic(scale(err, self.bad, self.good, self.low, self.high))
            for err in errors]

