# Copyright 2020 DeepMind Technologies Limited.
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
"""Haiku modules for feature processing."""

import copy
from typing import Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import interpolation
import tensorflow_datasets as tfds

Array = chex.Array


def _moments(image):
  """Compute the first and second moments of a given image."""
  c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]
  total_image = np.sum(image)
  m0 = np.sum(c0 * image) / total_image
  m1 = np.sum(c1 * image) / total_image
  m00 = np.sum((c0 - m0)**2 * image) / total_image
  m11 = np.sum((c1 - m1)**2 * image) / total_image
  m01 = np.sum((c0 - m0) * (c1 - m1) * image) / total_image
  mu_vector = np.array([m0, m1])
  covariance_matrix = np.array([[m00, m01], [m01, m11]])
  return mu_vector, covariance_matrix


def _deskew(image):
  """Image deskew."""
  c, v = _moments(image)
  alpha = v[0, 1] / v[0, 0]
  affine = np.array([[1, 0], [alpha, 1]])
  ocenter = np.array(image.shape) / 2.0
  offset = c - np.dot(affine, ocenter)
  return interpolation.affine_transform(image, affine, offset=offset)


def _deskew_dataset(dataset):
  """Dataset deskew."""
  deskewed = copy.deepcopy(dataset)
  for k, before in dataset.items():
    images = before["image"]
    num_images = images.shape[0]
    after = np.stack([_deskew(i) for i in np.squeeze(images, axis=-1)], axis=0)
    deskewed[k]["image"] = np.reshape(after, (num_images, -1))
  return deskewed


def load_deskewed_mnist(*a, **k):
  """Returns deskewed MNIST numpy dataset."""
  mnist_data, info = tfds.load(*a, **k)
  mnist_data = tfds.as_numpy(mnist_data)
  deskewed_data = _deskew_dataset(mnist_data)
  return deskewed_data, info


class MeanStdEstimator(hk.Module):
  """Online mean and standard deviation estimator using Welford's algorithm."""

  def __call__(self, sample: jax.Array) -> Tuple[Array, Array]:
    if len(sample.shape) > 1:
      raise ValueError("sample must be a rank 0 or 1 DeviceArray.")

    count = hk.get_state("count", shape=(), dtype=jnp.int32, init=jnp.zeros)
    mean = hk.get_state(
        "mean", shape=sample.shape, dtype=jnp.float32, init=jnp.zeros)
    m2 = hk.get_state(
        "m2", shape=sample.shape, dtype=jnp.float32, init=jnp.zeros)

    count += 1
    delta = sample - mean
    mean += delta / count
    delta_2 = sample - mean
    m2 += delta * delta_2

    hk.set_state("count", count)
    hk.set_state("mean", mean)
    hk.set_state("m2", m2)

    stddev = jnp.sqrt(m2 / count)
    return mean, stddev
