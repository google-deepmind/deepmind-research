# Copyright 2020 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type Aliases."""

from typing import Callable, Tuple, Union

import jax.numpy as jnp
import numpy as np
import optax

TensorLike = Union[np.ndarray, jnp.DeviceArray]

ActivationFn = Callable[[TensorLike], TensorLike]
GatingFn = Callable[[TensorLike], TensorLike]
NetworkFn = Callable[[TensorLike], TensorLike]

# Callable doesn't allow kwargs to be used, and we often want to
# pass in is_training=..., so ignore the arguments for the sake of pytype.
NormalizeFn = Callable[..., TensorLike]

OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]


