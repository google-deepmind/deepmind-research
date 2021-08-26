# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Adaptive gradient clipping transform for Optax."""
import jax
import jax.numpy as jnp
import optax


def compute_norm(x: jnp.ndarray, axis, keepdims: bool) -> jnp.ndarray:
  """Axis-wise euclidean norm."""
  return jnp.sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5 


def unitwise_norm(x: jnp.ndarray, eps: float) -> jnp.ndarray:
  """Compute norms of each output unit separately, also for linear layers."""
  if len(jnp.squeeze(x).shape) <= 1:  # Scalars and vectors
    axis = None
    keepdims = False
  elif len(x.shape) in [2, 3]:  # Linear layers of shape IO or multihead linear
    axis = 0
    keepdims = True
  elif len(x.shape) == 4:  # Conv kernels of shape HWIO
    axis = [0, 1, 2,]
    keepdims = True
  else:
    raise ValueError(f'Got a parameter with shape not in [0, 1, 2, 3, 4]! {x}')
  return jnp.maximum(compute_norm(x, axis, keepdims), eps)


def adaptive_grad_clip(clip: float, eps: float = 1e-3) -> optax.GradientTransformation:
  """Clip updates to be at most clipping * parameter_norm.

  References:
    [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
    Recognition Without Normalization.

  Args:
    clip: Maximum allowed ratio of update norm to parameter norm.
    eps: epsilon term to prevent clipping of zero-initialized params.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return optax.ClipByGlobalNormState()

  def _clip_fn(g_norm: jnp.ndarray, p_norm: jnp.ndarray, grad: jnp.ndarray) -> jnp.ndarray:
    return grad * jnp.minimum(p_norm / g_norm * clip, 1)
  
  def update_fn(updates, state, params):
    g_norm = jax.tree_map(lambda x: unitwise_norm(x, 1e-6), updates)
    p_norm = jax.tree_map(lambda x: unitwise_norm(x, eps), params)
    # If grad norm > clipping * param_norm, rescale
    updates = jax.tree_multimap(_clip_fn, g_norm, p_norm, updates)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)
