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
"""Utility functions."""

from typing import Optional, Text
from absl import logging
import jax
import jax.numpy as jnp


def topk_accuracy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    topk: int,
    ignore_label_above: Optional[int] = None,
) -> jnp.ndarray:
  """Top-num_codes accuracy."""
  assert len(labels.shape) == 1, 'topk expects 1d int labels.'
  assert len(logits.shape) == 2, 'topk expects 2d logits.'

  if ignore_label_above is not None:
    logits = logits[labels < ignore_label_above, :]
    labels = labels[labels < ignore_label_above]

  prds = jnp.argsort(logits, axis=1)[:, ::-1]
  prds = prds[:, :topk]
  total = jnp.any(prds == jnp.tile(labels[:, jnp.newaxis], [1, topk]), axis=1)

  return total


def softmax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    reduction: Optional[Text] = 'mean',
) -> jnp.ndarray:
  """Computes softmax cross entropy given logits and one-hot class labels.

  Args:
    logits: Logit output values.
    labels: Ground truth one-hot-encoded labels.
    reduction: Type of reduction to apply to loss.

  Returns:
    Loss value. If `reduction` is `none`, this has the same shape as `labels`;
    otherwise, it is scalar.

  Raises:
    ValueError: If the type of `reduction` is unsupported.
  """
  loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  if reduction == 'sum':
    return jnp.sum(loss)
  elif reduction == 'mean':
    return jnp.mean(loss)
  elif reduction == 'none' or reduction is None:
    return loss
  else:
    raise ValueError(f'Incorrect reduction mode {reduction}')


def l2_normalize(
    x: jnp.ndarray,
    axis: Optional[int] = None,
    epsilon: float = 1e-12,
) -> jnp.ndarray:
  """l2 normalize a tensor on an axis with numerical stability."""
  square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
  x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
  return x * x_inv_norm


def l2_weight_regularizer(params):
  """Helper to do lasso on weights.

  Args:
    params: the entire param set.

  Returns:
    Scalar of the l2 norm of the weights.
  """
  l2_norm = 0.
  for mod_name, mod_params in params.items():
    if 'norm' not in mod_name:
      for param_k, param_v in mod_params.items():
        if param_k != 'b' not in param_k:  # Filter out biases
          l2_norm += jnp.sum(jnp.square(param_v))
        else:
          logging.warning('Excluding %s/%s from optimizer weight decay!',
                          mod_name, param_k)
    else:
      logging.warning('Excluding %s from optimizer weight decay!', mod_name)

  return 0.5 * l2_norm


def regression_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Byol's regression loss. This is a simple cosine similarity."""
  normed_x, normed_y = l2_normalize(x, axis=-1), l2_normalize(y, axis=-1)
  return jnp.sum((normed_x - normed_y)**2, axis=-1)


def bcast_local_devices(value):
  """Broadcasts an object to all local devices."""
  devices = jax.local_devices()

  def _replicate(x):
    """Replicate an object on each device."""
    x = jnp.array(x)
    return jax.api.device_put_sharded(len(devices) * [x], devices)

  return jax.tree_util.tree_map(_replicate, value)


def get_first(xs):
  """Gets values from the first device."""
  return jax.tree_map(lambda x: x[0], xs)
