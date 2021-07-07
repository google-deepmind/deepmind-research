# Copyright 2021 Deepmind Technologies Limited.
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

"""Datasets."""

from typing import Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2471, 0.2435, 0.2616)
_CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
_CIFAR100_STD = (0.2673, 0.2564, 0.2762)

_DATA_URL = 'https://storage.googleapis.com/dm-adversarial-robustness/'
_ALLOWED_FILES = ('cifar10_ddpm.npz',)
_WEBPAGE = ('https://github.com/deepmind/deepmind-research/tree/master/'
            'adversarial_robustness')


def cifar10_preprocess(mode: str = 'train'):
  """Preprocessing functions for CIFAR-10."""
  def _preprocess_fn_train(example):
    """Preprocessing of CIFAR-10 images for training."""
    image = example['image']
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = _random_jitter(image, pad=4, crop=32)
    image = tf.image.random_flip_left_right(image)
    label = tf.cast(example['label'], tf.int32)
    return {'image': image, 'label': label}

  def _preprocess_fn_test(example):
    """Preprocessing of CIFAR-10 images for testing."""
    image = example['image']
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    label = tf.cast(example['label'], tf.int32)
    return {'image': image, 'label': label}

  return _preprocess_fn_train if mode == 'train' else _preprocess_fn_test


def cifar10_normalize(image: chex.Array) -> chex.Array:
  means = jnp.array(_CIFAR10_MEAN, dtype=image.dtype)
  stds = jnp.array(_CIFAR10_STD, dtype=image.dtype)
  return (image - means) / stds


def mnist_normalize(image: chex.Array) -> chex.Array:
  image = jnp.pad(image, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant',
                  constant_values=0)
  return (image - .5) * 2.


def cifar100_normalize(image: chex.Array) -> chex.Array:
  means = jnp.array(_CIFAR100_MEAN, dtype=image.dtype)
  stds = jnp.array(_CIFAR100_STD, dtype=image.dtype)
  return (image - means) / stds


def load_cifar10(batch_sizes: Sequence[int],
                 subset: str = 'train',
                 is_training: bool = True,
                 drop_remainder: bool = True,
                 repeat: int = 1) -> tf.data.Dataset:
  """Loads CIFAR-10."""
  if subset == 'train':
    ds = tfds.load(name='cifar10', split=tfds.Split.TRAIN)
    # In Gowal et al. (https://arxiv.org/abs/2010.03593) and Rebuffi et al.
    # (https://arxiv.org/abs/2103.01946), we also keep a separate validation
    # subset for early stopping and would run: ds = ds.skip(1_024).
  elif subset == 'test':
    ds = tfds.load(name='cifar10', split=tfds.Split.TEST)
  else:
    raise ValueError('Unknown subset: "{}"'.format(subset))

  ds = ds.cache()
  if is_training:
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=50_000, seed=0)
    ds = _repeat_batch(batch_sizes, ds, repeat=repeat)
  ds = ds.map(cifar10_preprocess('train' if is_training else 'test'),
              num_parallel_calls=tf.data.AUTOTUNE)
  for batch_size in reversed(batch_sizes):
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  return ds.prefetch(tf.data.AUTOTUNE)


def load_extra(batch_sizes: Sequence[int],
               path_npz: str,
               is_training: bool = True,
               drop_remainder: bool = True) -> tf.data.Dataset:
  """Loads extra data from a given path."""
  if not tf.io.gfile.exists(path_npz):
    if path_npz in _ALLOWED_FILES:
      path_npz = tf.keras.utils.get_file(path_npz, _DATA_URL + path_npz)
    else:
      raise ValueError(f'Extra data not found ({path_npz}). See {_WEBPAGE} for '
                       'more details.')
  with tf.io.gfile.GFile(path_npz, 'rb') as fp:
    npzfile = np.load(fp)
    data = {'image': npzfile['image'], 'label': npzfile['label']}
    with tf.device('/device:cpu:0'):  # Prevent allocation to happen on GPU.
      ds = tf.data.Dataset.from_tensor_slices(data)
  ds = ds.cache()
  if is_training:
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=50_000, seed=jax.host_id())
  ds = ds.map(cifar10_preprocess('train' if is_training else 'test'),
              num_parallel_calls=tf.data.AUTOTUNE)
  for batch_size in reversed(batch_sizes):
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  return ds.prefetch(tf.data.AUTOTUNE)


def load_dummy_data(batch_sizes: Sequence[int],
                    is_training: bool = True,
                    **unused_kwargs) -> tf.data.Dataset:
  """Loads fictive data (use this function when testing)."""
  ds = tf.data.Dataset.from_tensor_slices({
      'image': np.zeros((1, 32, 32, 3), np.float32),
      'label': np.zeros((1,), np.int32),
  })
  ds = ds.repeat()
  if not is_training:
    total_batch_size = np.prod(batch_sizes)
    ds = ds.take(total_batch_size)
  ds = ds.map(cifar10_preprocess('train' if is_training else 'test'),
              num_parallel_calls=tf.data.AUTOTUNE)
  for batch_size in reversed(batch_sizes):
    ds = ds.batch(batch_size, drop_remainder=True)
  return ds.prefetch(tf.data.AUTOTUNE)


def _random_jitter(image: tf.Tensor, pad: int, crop: int) -> tf.Tensor:
  shape = image.shape.as_list()
  image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]])
  image = tf.image.random_crop(image, size=[crop, crop, shape[2]])
  return image


def _repeat_batch(batch_sizes: Sequence[int],
                  ds: tf.data.Dataset,
                  repeat: int = 1) -> tf.data.Dataset:
  """Tiles the inner most batch dimension."""
  if repeat <= 1:
    return ds
  if batch_sizes[-1] % repeat != 0:
    raise ValueError(f'The last element of `batch_sizes` ({batch_sizes}) must '
                     f'be divisible by `repeat` ({repeat}).')
  # Perform regular batching with reduced number of elements.
  for i, batch_size in enumerate(reversed(batch_sizes)):
    ds = ds.batch(batch_size // repeat if i == 0 else batch_size,
                  drop_remainder=True)
  # Repeat batch.
  fn = lambda x: tf.repeat(x, repeats=repeat, axis=len(batch_sizes) - 1)
  def repeat_inner_batch(example):
    return jax.tree_map(fn, example)
  ds = ds.map(repeat_inner_batch,
              num_parallel_calls=tf.data.AUTOTUNE)
  # Unbatch.
  for _ in batch_sizes:
    ds = ds.unbatch()
  return ds
