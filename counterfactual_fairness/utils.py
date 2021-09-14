# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Common utilities."""

from typing import Optional, Union

from jax import random
import jax.numpy as jnp
import pandas as pd
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def get_dataset(dataset: pd.DataFrame,
                batch_size: int,
                shuffle_size: int = 10000,
                num_epochs: Optional[int] = None) -> tf.data.Dataset:
  """Makes a tf.Dataset with correct preprocessing."""
  dataset_copy = dataset.copy()
  for column in dataset.columns:
    if dataset[column].dtype.name == 'category':
      dataset_copy.loc[:, column] = dataset[column].cat.codes
  ds = tf.data.Dataset.from_tensor_slices(dataset_copy.values)
  if shuffle_size > 0:
    ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True)
  ds = ds.repeat(num_epochs)
  return ds.batch(batch_size, drop_remainder=True)


def multinomial_mode(
    distribution_or_probs: Union[tfd.Distribution, jnp.DeviceArray]
    ) -> jnp.DeviceArray:
  """Calculates the (one-hot) mode of a multinomial distribution.

  Args:
    distribution_or_probs:
      `tfp.distributions.Distribution` | List[tensors].
      If the former, it is assumed that it has a `probs` property, and
      represents a distribution over categories. If the latter, these are
      taken to be the probabilities of categories directly.
      In either case, it is assumed that `probs` will be shape
      (batch_size, dim).

  Returns:
    `DeviceArray`, float32, (batch_size, dim).
    The mode of the distribution - this will be in one-hot form, but contain
    multiple non-zero entries in the event that more than one probability is
    joint-highest.
  """
  if isinstance(distribution_or_probs, tfd.Distribution):
    probs = distribution_or_probs.probs_parameter()
  else:
    probs = distribution_or_probs
  max_prob = jnp.max(probs, axis=1, keepdims=True)
  mode = jnp.int32(jnp.equal(probs, max_prob))
  return jnp.float32(mode / jnp.sum(mode, axis=1, keepdims=True))


def multinomial_class(
    distribution_or_probs: Union[tfd.Distribution, jnp.DeviceArray]
) -> jnp.DeviceArray:
  """Computes the mode class of a multinomial distribution.

  Args:
    distribution_or_probs:
      `tfp.distributions.Distribution` | DeviceArray.
      As for `multinomial_mode`.

  Returns:
    `DeviceArray`, float32, (batch_size,).
    For each element in the batch, the index of the class with highest
    probability.
  """
  if isinstance(distribution_or_probs, tfd.Distribution):
    return jnp.argmax(distribution_or_probs.logits_parameter(), axis=1)
  return jnp.argmax(distribution_or_probs, axis=1)


def multinomial_mode_ndarray(probs: jnp.DeviceArray) -> jnp.DeviceArray:
  """Calculates the (one-hot) mode from an ndarray of class probabilities.

  Equivalent to `multinomial_mode` above, but implemented for numpy ndarrays
  rather than Tensors.

  Args:
    probs: `DeviceArray`, (batch_size, dim). Probabilities for each class, for
      each element in a batch.

  Returns:
    `DeviceArray`, (batch_size, dim).
  """
  max_prob = jnp.amax(probs, axis=1, keepdims=True)
  mode = jnp.equal(probs, max_prob).astype(jnp.int32)
  return (mode / jnp.sum(mode, axis=1, keepdims=True)).astype(jnp.float32)


def multinomial_accuracy(distribution_or_probs: tfd.Distribution,
                         data: jnp.DeviceArray) -> jnp.DeviceArray:
  """Compute the accuracy, averaged over a batch of data.

  Args:
    distribution_or_probs:
      `tfp.distributions.Distribution` | List[tensors].
      As for functions above.
    data: `DeviceArray`. Reference data, of shape (batch_size, dim).

  Returns:
    `DeviceArray`, float32, ().
    Overall scalar accuracy.
  """
  return jnp.mean(
      jnp.sum(multinomial_mode(distribution_or_probs) * data, axis=1))


def softmax_ndarray(logits: jnp.DeviceArray) -> jnp.DeviceArray:
  """Softmax function, implemented for numpy ndarrays."""
  assert len(logits.shape) == 2
  # Normalise for better stability.
  s = jnp.max(logits, axis=1, keepdims=True)
  e_x = jnp.exp(logits - s)
  return e_x / jnp.sum(e_x, axis=1, keepdims=True)


def get_samples(distribution, num_samples, seed=None):
  """Given a batched distribution, compute samples and reshape along batch.

  That is, we have a distribution of shape (batch_size, ...), where each element
  of the tensor is independent. We then draw num_samples from each component, to
  give a tensor of shape:

      (num_samples, batch_size, ...)

  Args:
    distribution: `tfp.distributions.Distribution`. The distribution from which
      to sample.
    num_samples: `Integral` | `DeviceArray`, int32, (). The number of samples.
    seed: `Integral` | `None`. The seed that will be forwarded to the call to
      distribution.sample. Defaults to `None`.

  Returns:
    `DeviceArray`, float32, (batch_size * num_samples, ...).
    Samples for each element of the batch.
  """
  # Obtain the sample from the distribution, which will be of shape
  # [num_samples] + batch_shape + event_shape.
  sample = distribution.sample(num_samples, seed=seed)
  sample = sample.reshape((-1, sample.shape[-1]))

  # Combine the first two dimensions through a reshape, so the result will
  # be of shape (num_samples * batch_size,) + shape_tail.
  return sample


def mmd_loss(distribution: tfd.Distribution,
             is_a: jnp.DeviceArray,
             num_samples: int,
             rng: jnp.ndarray,
             num_random_features: int = 50,
             gamma: float = 1.):
  """Given two distributions, compute the Maximum Mean Discrepancy (MMD).

  More exactly, this uses the 'FastMMD' approximation, a.k.a. 'Random Fourier
  Features'. See the description, for example, in sections 2.3.1 and 2.4 of
  https://arxiv.org/pdf/1511.00830.pdf.

  Args:
    distribution: Distribution whose `sample()` method will return a
      DeviceArray of shape (batch_size, dim).
    is_a: A boolean array indicating which elements of the batch correspond
      to class A (the remaining indices correspond to class B).
    num_samples: The number of samples to draw from `distribution`.
    rng: Random seed provided by the user.
    num_random_features: The number of random fourier features
      used in the expansion.
    gamma: The value of gamma in the Gaussian MMD kernel.

  Returns:
    `DeviceArray`, shape ().
    The scalar MMD value for samples taken from the given distributions.
  """
  if distribution.event_shape == ():  # pylint: disable=g-explicit-bool-comparison
    dim_x = distribution.batch_shape[1]
  else:
    dim_x, = distribution.event_shape

  # Obtain samples from the distribution, which will be of shape
  # [num_samples] + batch_shape + event_shape.
  samples = distribution.sample(num_samples, seed=rng)

  w = random.normal(rng, shape=((dim_x, num_random_features)))
  b = random.uniform(rng, shape=(num_random_features,),
                     minval=0, maxval=2*jnp.pi)

  def features(x):
    """Compute the kitchen sink feature."""
    # We need to contract last axis of x with first of W - do this with
    # tensordot. The result has shape:
    #   (?, ?, num_random_features)
    return jnp.sqrt(2 / num_random_features) * jnp.cos(
        jnp.sqrt(2 / gamma) * jnp.tensordot(x, w, axes=1) + b)

  # Compute the expected values of the given features.
  # The first axis represents the samples from the distribution,
  # second axis represents the batch_size.
  # Each of these now has shape (num_random_features,)
  exp_features = features(samples)
  # Swap axes so that batch_size is the last dimension to be compatible
  # with is_a and is_b shape at the next step
  exp_features_reshaped = jnp.swapaxes(exp_features, 1, 2)
  # Current dimensions [num_samples, num_random_features, batch_size]
  exp_features_reshaped_a = jnp.where(is_a, exp_features_reshaped, 0)
  exp_features_reshaped_b = jnp.where(is_a, 0, exp_features_reshaped)
  exp_features_a = jnp.mean(exp_features_reshaped_a, axis=(0, 2))
  exp_features_b = jnp.mean(exp_features_reshaped_b, axis=(0, 2))

  assert exp_features_a.shape == (num_random_features,)
  difference = exp_features_a - exp_features_b

  # Compute the squared norm. Shape ().
  return jnp.tensordot(difference, difference, axes=1)


def mmd_loss_exact(distribution_a, distribution_b, num_samples, gamma=1.):
  """Exact estimate of MMD."""
  assert distribution_a.event_shape == distribution_b.event_shape
  assert distribution_a.batch_shape[1:] == distribution_b.batch_shape[1:]

  # shape (num_samples * batch_size_a, dim_x)
  samples_a = get_samples(distribution_a, num_samples)
  # shape (num_samples * batch_size_b, dim_x)
  samples_b = get_samples(distribution_b, num_samples)

  # Make matrices of shape
  #   (size_b, size_a, dim_x)
  # where:
  #   size_a = num_samples * batch_size_a
  #   size_b = num_samples * batch_size_b
  size_a = samples_a.shape[0]
  size_b = samples_b.shape[0]
  x_a = jnp.expand_dims(samples_a, axis=0)
  x_a = jnp.tile(x_a, (size_b, 1, 1))
  x_b = jnp.expand_dims(samples_b, axis=1)
  x_b = jnp.tile(x_b, (1, size_a, 1))

  def kernel_mean(x, y):
    """Gaussian kernel mean."""

    diff = x - y

    # Contract over dim_x.
    exponent = - jnp.einsum('ijk,ijk->ij', diff, diff) / gamma

    # This has shape (size_b, size_a).
    kernel_matrix = jnp.exp(exponent)

    # Shape ().
    return jnp.mean(kernel_matrix)

  # Equation 7 from arxiv 1511.00830
  return (
      kernel_mean(x_a, x_a)
      + kernel_mean(x_b, x_b)
      - 2 * kernel_mean(x_a, x_b))


def scalar_log_prob(distribution, val):
  """Compute the log_prob per batch entry.

  It is conceptually similar to:

    jnp.sum(distribution.log_prob(val), axis=1)

  However, classes like `tfp.distributions.Multinomial` have a log_prob which
  returns a tensor of shape (batch_size,), which will cause the above
  incantation to fail. In these cases we fall back to returning just:

    distribution.log_prob(val)

  Args:
    distribution: `tfp.distributions.Distribution` which implements log_prob.
    val: `DeviceArray`, (batch_size, dim).

  Returns:
    `DeviceArray`, (batch_size,).
    If the result of log_prob has a trailing dimension, we perform a reduce_sum
    over it.

  Raises:
    ValueError: If distribution.log_prob(val) has an unsupported shape.
  """
  log_prob_val = distribution.log_prob(val)
  if len(log_prob_val.shape) == 1:
    return log_prob_val
  elif len(log_prob_val.shape) > 2:
    raise ValueError('log_prob_val has unexpected shape {}.'.format(
        log_prob_val.shape))
  return jnp.sum(log_prob_val, axis=1)
