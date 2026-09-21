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

"""Dataset iterators Mnist, ColorMnist."""

import enum

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from avae import types


class Dataset(enum.Enum):
  color_mnist = enum.auto()


class MnistDataIterator(object):
  """Mnist data iterator class.

  Data is obtained as dataclass, types.LabelledData.
  """

  def __init__(self, subset: str, batch_size: int):
    """Class initializer.

    Args:
     subset: Subset of dataset.
     batch_size: Batch size of the returned dataset iterator.
    """
    datasets = tfds.load('mnist')
    train_dataset = datasets[subset]
    def _map_fun(x):
      return {'data': tf.cast(x['image'], tf.float32) / 255.0,
              'label': x['label']}
    train_dataset = train_dataset.map(_map_fun)
    train_dataset = train_dataset.batch(batch_size,
                                        drop_remainder=True).repeat()
    self._iterator = iter(tfds.as_numpy(train_dataset))
    self._batch_size = batch_size

  def __iter__(self):
    return self

  def __next__(self) -> types.LabelledData:
    return types.LabelledData(**next(self._iterator))


class ColorMnistDataIterator(MnistDataIterator):
  """Color Mnist data iterator.

  Each ColorMnist image is of shape (28, 28, 3). ColorMnist digit can have 7
  different colors by permution of RGB channels (turning on and off RGB
  channels, except for all channels off permutation).

  Data is obtained as dataclass, util_dataclasses.LabelledData.
  """

  def __next__(self) -> types.LabelledData:
    mnist_batch = next(self._iterator)
    mnist_image = mnist_batch['data']
    # Colors are supported by turning off and on RGB channels.
    # Thus possible colors are
    # [black (excluded), red, green, yellow, blue, magenta, cyan, white]
    # color_id takes value from [1,8)
    color_id = np.random.randint(7, size=self._batch_size) + 1
    red_channel_bool = np.mod(color_id, 2)
    red_channel_bool = jnp.reshape(red_channel_bool, [-1, 1, 1, 1])
    blue_channel_bool = np.floor_divide(color_id, 4)
    blue_channel_bool = jnp.reshape(blue_channel_bool, [-1, 1, 1, 1])
    green_channel_bool = np.mod(np.floor_divide(color_id, 2), 2)
    green_channel_bool = jnp.reshape(green_channel_bool, [-1, 1, 1, 1])

    color_mnist_image = jnp.stack([
        jnp.multiply(red_channel_bool, mnist_image),
        jnp.multiply(blue_channel_bool, mnist_image),
        jnp.multiply(green_channel_bool, mnist_image)], axis=3)
    color_mnist_image = jnp.reshape(color_mnist_image, [-1, 28, 28, 3])
    # Color id takes value [1, 8)
    # Although to make classification code easier `color` label attached to data
    # takes value in [0, 7) (by subtracting 1 from color id)
    return types.LabelledData(
        data=color_mnist_image, label=mnist_batch['label'])
