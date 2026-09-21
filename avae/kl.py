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

"""Various KL implementations in JAX."""

import jax.numpy as jnp


def kl_p_with_uniform_normal(mean: jnp.ndarray,
                             variance: jnp.ndarray) -> jnp.ndarray:
  r"""KL between p_dist with uniform normal prior.

  Args:
   mean: Mean of the gaussian distribution, shape (latent_dims,)
   variance: Variance of the gaussian distribution, shape (latent_dims,)
  Returns:
   KL divergence KL(P||N(0, 1)) shape ()
  """

  if len(variance.shape) == 2:
    # If `variance` is a full covariance matrix
    variance_trace = jnp.trace(variance)
    _, ldet1 = jnp.linalg.slogdet(variance)
  else:
    variance_trace = jnp.sum(variance)
    ldet1 = jnp.sum(jnp.log(variance))

  mean_contribution = jnp.sum(jnp.square(mean))
  res = -ldet1
  res += variance_trace + mean_contribution - mean.shape[0]
  return res * 0.5
