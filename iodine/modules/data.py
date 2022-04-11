# Copyright 2019 Deepmind Technologies Limited.
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
"""Data loading functionality for IODINE."""
# pylint: disable=g-multiple-import, missing-docstring, unused-import
import os.path

from iodine.modules.utils import flatten_all_but_last, ensure_3d
from multi_object_datasets import (
    clevr_with_masks,
    multi_dsprites,
    tetrominoes,
    objects_room,
)
from shapeguard import ShapeGuard
import sonnet as snt
import tensorflow.compat.v1 as tf


class IODINEDataset(snt.AbstractModule):
  num_true_objects = 1
  num_channels = 3

  factors = {}

  def __init__(
      self,
      path,
      batch_size,
      image_dim,
      crop_region=None,
      shuffle_buffer=1000,
      max_num_objects=None,
      min_num_objects=None,
      grayscale=False,
      name="dataset",
      **kwargs,
  ):
    super().__init__(name=name)
    self.path = os.path.abspath(os.path.expanduser(path))
    self.batch_size = batch_size
    self.crop_region = crop_region
    self.image_dim = image_dim
    self.shuffle_buffer = shuffle_buffer
    self.max_num_objects = max_num_objects
    self.min_num_objects = min_num_objects
    self.grayscale = grayscale
    self.dataset = None

  def _build(self, subset="train"):
    dataset = self.dataset

    # filter by number of objects
    if self.max_num_objects is not None or self.min_num_objects is not None:
      dataset = self.dataset.filter(self.filter_by_num_objects)

    if subset == "train":
      # normal mode returns a shuffled dataset iterator
      if self.shuffle_buffer is not None:
        dataset = dataset.shuffle(self.shuffle_buffer)
    elif subset == "summary":
      # for generating summaries and overview images
      # returns a single fixed batch
      dataset = dataset.take(self.batch_size)

    # repeat and batch
    dataset = dataset.repeat().batch(self.batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()

    # preprocess the data to ensure correct format, scale images etc.
    data = self.preprocess(data)
    return data

  def filter_by_num_objects(self, d):
    if "visibility" not in d:
      return tf.constant(True)
    min_num_objects = self.max_num_objects or 0
    max_num_objects = self.max_num_objects or 6

    min_predicate = tf.greater_equal(
        tf.reduce_sum(d["visibility"]),
        tf.constant(min_num_objects - 1e-5, dtype=tf.float32),
    )
    max_predicate = tf.less_equal(
        tf.reduce_sum(d["visibility"]),
        tf.constant(max_num_objects + 1e-5, dtype=tf.float32),
    )
    return tf.logical_and(min_predicate, max_predicate)

  def preprocess(self, data):
    sg = ShapeGuard(dims={
        "B": self.batch_size,
        "H": self.image_dim[0],
        "W": self.image_dim[1]
    })
    image = sg.guard(data["image"], "B, h, w, C")
    mask = sg.guard(data["mask"], "B, L, h, w, 1")

    # to float
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0

    # crop
    if self.crop_region is not None:
      height_slice = slice(self.crop_region[0][0], self.crop_region[0][1])
      width_slice = slice(self.crop_region[1][0], self.crop_region[1][1])
      image = image[:, height_slice, width_slice, :]

      mask = mask[:, :, height_slice, width_slice, :]

    flat_mask, unflatten = flatten_all_but_last(mask, n_dims=3)

    # rescale
    size = tf.constant(
        self.image_dim, dtype=tf.int32, shape=[2], verify_shape=True)
    image = tf.image.resize_images(
        image, size, method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.image.resize_images(
        flat_mask, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if self.grayscale:
      image = tf.reduce_mean(image, axis=-1, keepdims=True)

    output = {
        "image": sg.guard(image[:, None], "B, T, H, W, C"),
        "mask": sg.guard(unflatten(mask)[:, None], "B, T, L, H, W, 1"),
        "factors": self.preprocess_factors(data, sg),
    }

    if "visibility" in data:
      output["visibility"] = sg.guard(data["visibility"], "B, L")
    else:
      output["visibility"] = tf.ones(sg["B, L"], dtype=tf.float32)

    return output

  def preprocess_factors(self, data, sg):
    return {
        name: sg.guard(ensure_3d(data[name]), "B, L, *")
        for name in self.factors
    }

  def get_placeholders(self, batch_size=None):
    batch_size = batch_size or self.batch_size
    sg = ShapeGuard(
        dims={
            "B": batch_size,
            "H": self.image_dim[0],
            "W": self.image_dim[1],
            "L": self.num_true_objects,
            "C": 3,
            "T": 1,
        })
    return {
        "image": tf.placeholder(dtype=tf.float32, shape=sg["B, T, H, W, C"]),
        "mask": tf.placeholder(dtype=tf.float32, shape=sg["B, T, L, H, W, 1"]),
        "visibility": tf.placeholder(dtype=tf.float32, shape=sg["B, L"]),
        "factors": {
            name:
            tf.placeholder(dtype=dtype, shape=sg["B, L, {}".format(size)])
            for name, (dtype, size) in self.factors
        },
    }


class CLEVR(IODINEDataset):
  num_true_objects = 11
  num_channels = 3
  factors = {
      "color": (tf.uint8, 1),
      "shape": (tf.uint8, 1),
      "size": (tf.uint8, 1),
      "position": (tf.float32, 3),
      "rotation": (tf.float32, 1),
  }

  def __init__(
      self,
      path,
      crop_region=((29, 221), (64, 256)),
      image_dim=(128, 128),
      name="clevr",
      **kwargs,
  ):
    super().__init__(
        path=path,
        crop_region=crop_region,
        image_dim=image_dim,
        name=name,
        **kwargs)
    self.dataset = clevr_with_masks.dataset(self.path)

  def preprocess_factors(self, data, sg):

    return {
        "color": sg.guard(ensure_3d(data["color"]), "B, L, 1"),
        "shape": sg.guard(ensure_3d(data["shape"]), "B, L, 1"),
        "size": sg.guard(ensure_3d(data["color"]), "B, L, 1"),
        "position": sg.guard(ensure_3d(data["pixel_coords"]), "B, L, 3"),
        "rotation": sg.guard(ensure_3d(data["rotation"]), "B, L, 1"),
    }


class MultiDSprites(IODINEDataset):
  num_true_objects = 6
  num_channels = 3
  factors = {
      "color": (tf.float32, 3),
      "shape": (tf.uint8, 1),
      "scale": (tf.float32, 1),
      "x": (tf.float32, 1),
      "y": (tf.float32, 1),
      "orientation": (tf.float32, 1),
  }

  def __init__(
      self,
      path,
      # variant from ['binarized', 'colored_on_grayscale', 'colored_on_colored']
      dataset_variant="colored_on_grayscale",
      image_dim=(64, 64),
      name="multi_dsprites",
      **kwargs,
  ):
    super().__init__(path=path, name=name, image_dim=image_dim, **kwargs)
    self.dataset_variant = dataset_variant
    self.dataset = multi_dsprites.dataset(self.path, self.dataset_variant)


class Tetrominoes(IODINEDataset):
  num_true_objects = 6
  num_channels = 3
  factors = {
      "color": (tf.uint8, 3),
      "shape": (tf.uint8, 1),
      "position": (tf.float32, 2),
  }

  def __init__(self, path, image_dim=(35, 35), name="tetrominoes", **kwargs):
    super().__init__(path=path, name=name, image_dim=image_dim, **kwargs)
    self.dataset = tetrominoes.dataset(self.path)

  def preprocess_factors(self, data, sg):
    pos_x = ensure_3d(data["x"])
    pos_y = ensure_3d(data["y"])
    position = tf.concat([pos_x, pos_y], axis=2)

    return {
        "color": sg.guard(ensure_3d(data["color"]), "B, L, 3"),
        "shape": sg.guard(ensure_3d(data["shape"]), "B, L, 1"),
        "position": sg.guard(ensure_3d(position), "B, L, 2"),
    }
