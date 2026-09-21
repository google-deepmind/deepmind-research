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

"""Interface for loading sketchy data into tensorflow."""

import tensorflow.compat.v2 as tf


def load_frames(filenames, num_parallel_reads=1, num_map_threads=None):
  if not num_map_threads:
    num_map_threads = num_parallel_reads
  dataset = tf.data.TFRecordDataset(
      filenames, num_parallel_reads=num_parallel_reads)
  return dataset.map(_parse_example, num_parallel_calls=num_map_threads)


_FEATURES = {
    # Actions
    'actions':
        tf.io.FixedLenFeature(shape=7, dtype=tf.float32),

    # Observations
    'gripper/joints/velocity':
        tf.io.FixedLenFeature(shape=1, dtype=tf.float32),
    'gripper/joints/torque':
        tf.io.FixedLenFeature(shape=1, dtype=tf.float32),
    'gripper/grasp':
        tf.io.FixedLenFeature(shape=1, dtype=tf.int64),
    'gripper/joints/angle':
        tf.io.FixedLenFeature(shape=1, dtype=tf.float32),
    'sawyer/joints/velocity':
        tf.io.FixedLenFeature(shape=7, dtype=tf.float32),
    'sawyer/pinch/pose':
        tf.io.FixedLenFeature(shape=7, dtype=tf.float32),
    'sawyer/tcp/pose':
        tf.io.FixedLenFeature(shape=7, dtype=tf.float32),
    'sawyer/tcp/effort':
        tf.io.FixedLenFeature(shape=6, dtype=tf.float32),
    'sawyer/joints/torque':
        tf.io.FixedLenFeature(shape=7, dtype=tf.float32),
    'sawyer/tcp/velocity':
        tf.io.FixedLenFeature(shape=6, dtype=tf.float32),
    'sawyer/joints/angle':
        tf.io.FixedLenFeature(shape=7, dtype=tf.float32),
    'wrist/torque':
        tf.io.FixedLenFeature(shape=3, dtype=tf.float32),
    'wrist/force':
        tf.io.FixedLenFeature(shape=3, dtype=tf.float32),
    'pixels/basket_front_left':
        tf.io.FixedLenFeature(shape=1, dtype=tf.string),
    'pixels/basket_back_left':
        tf.io.FixedLenFeature(shape=1, dtype=tf.string),
    'pixels/basket_front_right':
        tf.io.FixedLenFeature(shape=1, dtype=tf.string),
    'pixels/royale_camera_driver_depth':
        tf.io.FixedLenFeature(shape=(171, 224, 1), dtype=tf.float32),
    'pixels/royale_camera_driver_gray':
        tf.io.FixedLenFeature(shape=1, dtype=tf.string),
    'pixels/usbcam0':
        tf.io.FixedLenFeature(shape=1, dtype=tf.string),
    'pixels/usbcam1':
        tf.io.FixedLenFeature(shape=1, dtype=tf.string),
}


def _parse_example(example):
  return _decode_images(tf.io.parse_single_example(example, _FEATURES))


def _decode_images(record):
  for name, value in list(record.items()):
    if value.dtype == tf.string:
      record[name] = tf.io.decode_jpeg(value[0])
  return record
