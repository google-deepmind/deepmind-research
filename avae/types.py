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

"""Useful dataclasses types used across the code."""

from typing import Optional

import dataclasses
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class ELBOOutputs:
  elbo: jnp.ndarray
  data_fidelity: jnp.ndarray
  kl: jnp.ndarray


@dataclasses.dataclass(frozen=True)
class LabelledData:
  """A batch of labelled examples.

  Attributes:
    data: Array of shape (batch_size, ...).
    label: Array of shape (batch_size, ...).
  """
  data: np.ndarray
  label: Optional[np.ndarray]


@dataclasses.dataclass(frozen=True)
class NormalParams:
  """Parameters of a normal distribution.

  Attributes:
    mean: Array of shape (batch_size, latent_dim).
    variance: Array of shape (batch_size, latent_dim).
  """
  mean: jnp.ndarray
  variance: jnp.ndarray
