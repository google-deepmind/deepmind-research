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

"""Transforms errors into rewards for reinforcement learning."""

import abc
import math
from typing import List

# Utility Functions
def clip_value(value: float, min_val: float = 0, max_val: float = 1) -> float:
    """Clips a value between min_val and max_val, preserving NaN."""
    return value if math.isnan(value) else max(min_val, min(max_val, value))

def scale_value(value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    """Linearly scales value from [old_min, old_max] to [new_min, new_max]."""
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

def logistic_value(value: float) -> float:
    """Computes the logistic function, clipped for stability."""
    clipped = clip_value(value, -50, 50)
    return 1 / (1 + math.exp(-clipped))

# Abstract Base Class
class RewardTransform(abc.ABC):
    """Base class for transforming errors into rewards."""
    
    @abc.abstractmethod
    def __call__(self, errors: List[float]) -> List[float]:
        """Converts a list of errors into a list of rewards."""

# Transformation Classes
class EqualReward(RewardTransform):
    """Rewards 1 for zero error, else a specified value."""
    def __init__(self, nonzero_value: float = 0):
        self.nonzero_value = nonzero_value
    
    def __call__(self, errors: List[float]) -> List[float]:
        return [error if math.isnan(error) else (1 if error == 0 else self.nonzero_value) 
                for error in errors]

class AbsoluteReward(RewardTransform):
    """Returns the absolute value of errors."""
    def __call__(self, errors: List[float]) -> List[float]:
        return [abs(error) for error in errors]

class NegatedReward(RewardTransform):
    """Negates the errors."""
    def __call__(self, errors: List[float]) -> List[float]:
        return [-error for error in errors]

class PowerReward(RewardTransform):
    """Raises errors to a specified power."""
    def __init__(self, exponent: float):
        self.exponent = exponent
    
    def __call__(self, errors: List[float]) -> List[float]:
        return [error ** self.exponent for error in errors]

class LogReward(RewardTransform):
    """Computes the natural log of errors plus a small offset."""
    def __init__(self, offset: float = 1e-4):
        self.offset = offset
    
    def __call__(self, errors: List[float]) -> List[float]:
        return [math.log(error + self.offset) for error in errors]

class LinearClippedReward(RewardTransform):
    """Scales errors linearly from bad (0) to good (1), clipped to [0, 1]."""
    def __init__(self, bad_value: float, good_value: float = 0):
        self.bad_value = bad_value
        self.good_value = good_value
    
    def __call__(self, errors: List[float]) -> List[float]:
        return [clip_value(scale_value(error, self.bad_value, self.good_value, 0, 1)) 
                for error in errors]

class SoftPlusReward(RewardTransform):
    """Smoothly scales errors from bad (~0.1) to good (1), clipped to [0, 1]."""
    def __init__(self, bad_value: float, good_value: float = 0, sharpness: float = -math.log(19)):
        self.bad_value = bad_value
        self.good_value = good_value
        self.sharpness = sharpness
    
    def __call__(self, errors: List[float]) -> List[float]:
        return [clip_value(2 * logistic_value(scale_value(error, self.bad_value, self.good_value, self.sharpness, 0))) 
                for error in errors]

class NegExpReward(RewardTransform):
    """Exponentially scales errors from bad (~0.1) to good (1), clipped to [0, 1]."""
    def __init__(self, bad_value: float, good_value: float = 0, sharpness: float = -math.log(0.1)):
        self.bad_value = bad_value
        self.good_value = good_value
        self.sharpness = sharpness
    
    def __call__(self, errors: List[float]) -> List[float]:
        return [clip_value(math.exp(-scale_value(error, self.bad_value, self.good_value, self.sharpness, 0))) 
                for error in errors]

class SigmoidReward(RewardTransform):
    """Scales errors sigmoidally from bad (~0.05) to good (~0.95)."""
    def __init__(self, bad_value: float, good_value: float, 
                 low_sharpness: float = -math.log(19), high_sharpness: float = math.log(19)):
        self.bad_value = bad_value
        self.good_value = good_value
        self.low_sharpness = low_sharpness
        self.high_sharpness = high_sharpness
    
    def __call__(self, errors: List[float]) -> List[float]:
        return [logistic_value(scale_value(error, self.bad_value, self.good_value, self.low_sharpness, self.high_sharpness)) 
                for error in errors]
