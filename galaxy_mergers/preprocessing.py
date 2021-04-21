# Copyright 2021 DeepMind Technologies Limited.
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

"""Pre-processing functions for input data."""

import functools
from absl import logging
import tensorflow.compat.v2 as tf
from galaxy_mergers import losses


CROP_TYPE_NONE = 'crop_none'
CROP_TYPE_FIXED = 'crop_fixed'
CROP_TYPE_RANDOM = 'crop_random'

DATASET_FREQUENCY_MEAN = 4.0
DATASET_FREQUENCY_RANGE = 8.0

PHYSICAL_FEATURES_MIN_MAX = {
    'redshift': (0.572788, 2.112304),
    'mass': (9.823963, 10.951282)
}

ALL_FREQUENCIES = [105, 125, 160, 435, 606, 775, 850]

VALID_ADDITIONAL_FEATURES = ['redshift', 'sequence_average_redshift', 'mass']


def _make_padding_sizes(pad_size, random_centering):
  if random_centering:
    pad_size_left = tf.random.uniform(
        shape=[], minval=0, maxval=pad_size+1, dtype=tf.int32)
  else:
    pad_size_left = pad_size // 2
  pad_size_right = pad_size - pad_size_left
  return pad_size_left, pad_size_right


def resize_and_pad(image, target_size, random_centering):
  """Resize image to target_size (<= image.size) and pad to original size."""
  original_shape = image.shape
  size = tf.reshape(target_size, [1])
  size = tf.concat([size, size], axis=0)
  image = tf.image.resize(image, size=size)
  pad_size = original_shape[1] - target_size
  pad_size_left, pad_size_right = _make_padding_sizes(
      pad_size, random_centering)
  padding = [[pad_size_left, pad_size_right],
             [pad_size_left, pad_size_right], [0, 0]]
  if len(original_shape) == 4:
    padding = [[0, 0]] + padding
  image = tf.pad(image, padding)
  image.set_shape(original_shape)
  return image


def resize_and_extract(image, target_size, random_centering):
  """Upscale image to target_size (>image.size), extract original size crop."""
  original_shape = image.shape
  size = tf.reshape(target_size, [1])
  size = tf.concat([size, size], axis=0)
  image = tf.image.resize(image, size=size)
  pad_size = target_size - original_shape[1]
  pad_size_left, pad_size_right = _make_padding_sizes(
      pad_size, random_centering)
  if len(original_shape) == 3:
    image = tf.expand_dims(image, 0)
  image = tf.cond(pad_size_right > 0,
                  lambda: image[:, pad_size_left:-pad_size_right, :, :],
                  lambda: image[:, pad_size_left:, :, :])
  image = tf.cond(pad_size_right > 0,
                  lambda: image[:, :, pad_size_left:-pad_size_right, :],
                  lambda: image[:, :, pad_size_left:, :])
  if len(original_shape) == 3:
    image = tf.squeeze(image, 0)
  image.set_shape(original_shape)
  return image


def resize_and_center(image, target_size, random_centering):
  return tf.cond(
      tf.math.less_equal(target_size, image.shape[1]),
      lambda: resize_and_pad(image, target_size, random_centering),
      lambda: resize_and_extract(image, target_size, random_centering))


def random_rotation_and_flip(image):
  angle = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
  return tf.image.random_flip_left_right(tf.image.rot90(image, angle))


def get_all_rotations_and_flips(images):
  assert isinstance(images, list)
  new_images = []
  for image in images:
    for rotation in range(4):
      new_images.append(tf.image.rot90(image, rotation))
      flipped_image = tf.image.flip_left_right(image)
      new_images.append(tf.image.rot90(flipped_image, rotation))
  return new_images


def random_rescaling(image, random_centering):
  assert image.shape.as_list()[0] == image.shape.as_list()[1]
  original_size = image.shape.as_list()[1]
  min_size = 2 * (original_size // 4)
  max_size = original_size * 2
  target_size = tf.random.uniform(
      shape=[], minval=min_size, maxval=max_size // 2,
      dtype=tf.int32) * 2
  return resize_and_center(image, target_size, random_centering)


def get_all_rescalings(images, image_width, random_centering):
  """Get a uniform sample of rescalings of all images in input."""
  assert isinstance(images, list)
  min_size = 2 * (image_width // 4)
  max_size = image_width * 2
  delta_size = (max_size + 2 - min_size) // 5
  sizes = range(min_size, max_size + 2, delta_size)
  new_images = []
  for image in images:
    for size in sizes:
      new_images.append(resize_and_center(image, size, random_centering))
  return new_images


def move_repeats_to_batch(image, n_repeats):
  width, height, n_channels = image.shape.as_list()[1:]
  image = tf.reshape(image, [-1, width, height, n_channels, n_repeats])
  image = tf.transpose(image, [0, 4, 1, 2, 3])  # [B, repeats, x, y, c]
  return tf.reshape(image, [-1, width, height, n_channels])


def get_classification_label(dataset_row, class_boundaries):
  merge_time = dataset_row['grounded_normalized_time']
  label = tf.dtypes.cast(0, tf.int64)
  for category, intervals in class_boundaries.items():
    for interval in intervals:
      if merge_time > interval[0] and merge_time < interval[1]:
        label = tf.dtypes.cast(int(category), tf.int64)
  return label


def get_regression_label(dataset_row, task_type):
  """Returns time-until-merger regression target given desired modeling task."""
  if task_type == losses.TASK_NORMALIZED_REGRESSION:
    return tf.dtypes.cast(dataset_row['normalized_time'], tf.float32)
  elif task_type == losses.TASK_GROUNDED_UNNORMALIZED_REGRESSION:
    return tf.dtypes.cast(dataset_row['grounded_normalized_time'], tf.float32)
  elif task_type == losses.TASK_UNNORMALIZED_REGRESSION:
    return tf.dtypes.cast(dataset_row['unnormalized_time'], tf.float32)
  elif task_type == losses.TASK_CLASSIFICATION:
    return tf.dtypes.cast(dataset_row['grounded_normalized_time'], tf.float32)
  else:
    raise ValueError


def get_normalized_time_target(dataset_row):
  return tf.dtypes.cast(dataset_row['normalized_time'], tf.float32)


def apply_time_filter(dataset_row, time_interval):
  """Returns True if data is within the given time intervals."""
  merge_time = dataset_row['grounded_normalized_time']
  lower_time, upper_time = time_interval
  return merge_time > lower_time and merge_time < upper_time


def normalize_physical_feature(name, dataset_row):
  min_feat, max_feat = PHYSICAL_FEATURES_MIN_MAX[name]
  value = getattr(dataset_row, name)
  return 2 * (value - min_feat) / (max_feat - min_feat) - 1


def prepare_dataset(ds, target_size, crop_type, n_repeats, augmentations,
                    task_type, additional_features, class_boundaries,
                    time_intervals=None, frequencies_to_use='all',
                    additional_lambdas=None):
  """Prepare a zipped dataset of image, classification/regression labels."""
  def _prepare_image(dataset_row):
    """Transpose, crop and cast an image."""
    image = tf.dtypes.cast(dataset_row['image'], tf.float32)
    image = tf.reshape(image, tf.cast(dataset_row['image_shape'], tf.int32))
    image = tf.transpose(image, perm=[1, 2, 0])  # Convert to NHWC

    freqs = ALL_FREQUENCIES if frequencies_to_use == 'all' else frequencies_to_use
    idxs_to_keep = [ALL_FREQUENCIES.index(f) for f in freqs]
    image = tf.gather(params=image, indices=idxs_to_keep, axis=-1)

    # Based on offline computation on the empirical frequency range:
    # Converts [0, 8.] ~~> [-1, 1]
    image = (image - DATASET_FREQUENCY_MEAN)/(DATASET_FREQUENCY_RANGE/2.0)

    def crop(image):
      if crop_type == CROP_TYPE_FIXED:
        crop_loc = tf.cast(dataset_row['proposed_crop'][0], tf.int32)
        crop_size = tf.cast(dataset_row['proposed_crop'][1], tf.int32)
        image = image[
            crop_loc[0]:crop_loc[0] + crop_size[0],
            crop_loc[1]:crop_loc[1] + crop_size[1], :]
        image = tf.image.resize(image, target_size[0:2])
        image.set_shape([target_size[0], target_size[1], target_size[2]])

      elif crop_type == CROP_TYPE_RANDOM:
        image = tf.image.random_crop(image, target_size)
        image.set_shape([target_size[0], target_size[1], target_size[2]])

      elif crop_type != CROP_TYPE_NONE:
        raise NotImplementedError

      return image

    repeated_images = []
    for _ in range(n_repeats):
      repeated_images.append(crop(image))
    image = tf.concat(repeated_images, axis=-1)

    if augmentations['rotation_and_flip']:
      image = random_rotation_and_flip(image)
    if augmentations['rescaling']:
      image = random_rescaling(image, augmentations['translation'])

    return image

  def get_regression_label_wrapper(dataset_row):
    return get_regression_label(dataset_row, task_type=task_type)

  def get_classification_label_wrapper(dataset_row):
    return get_classification_label(dataset_row,
                                    class_boundaries=class_boundaries)

  if time_intervals:
    for time_interval in time_intervals:
      filter_fn = functools.partial(apply_time_filter,
                                    time_interval=time_interval)
      ds = ds.filter(filter_fn)

  datasets = [ds.map(_prepare_image)]

  if additional_features:
    additional_features = additional_features.split(',')
    assert all([f in VALID_ADDITIONAL_FEATURES for f in additional_features])
    logging.info('Running with additional features: %s.',
                 ', '.join(additional_features))

    def _prepare_additional_features(dataset_row):
      features = []
      for f in additional_features:
        features.append(normalize_physical_feature(f, dataset_row))
      features = tf.convert_to_tensor(features, dtype=tf.float32)
      features.set_shape([len(additional_features)])
      return features

    datasets += [ds.map(_prepare_additional_features)]

  datasets += [
      ds.map(get_classification_label_wrapper),
      ds.map(get_regression_label_wrapper),
      ds.map(get_normalized_time_target)]

  if additional_lambdas:
    for process_fn in additional_lambdas:
      datasets += [ds.map(process_fn)]

  return tf.data.Dataset.zip(tuple(datasets))

