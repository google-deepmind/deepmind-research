# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet dataset with pre-processing and augmentation.

Deng, et al CVPR 2009 - ImageNet: A large-scale hierarchical image database.
https://image-net.org/
"""

import enum
from typing import Any, Generator, Mapping, Optional, Sequence, Text, Tuple

import jax
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from perceiver.train import autoaugment


Batch = Mapping[Text, np.ndarray]
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)
AUTOTUNE = tf.data.experimental.AUTOTUNE

INPUT_DIM = 224  # The number of pixels in the image resize.


class Split(enum.Enum):
  """ImageNet dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @classmethod
  def from_string(cls, name: Text) -> 'Split':
    return {'TRAIN': Split.TRAIN, 'TRAIN_AND_VALID': Split.TRAIN_AND_VALID,
            'VALID': Split.VALID, 'VALIDATION': Split.VALID,
            'TEST': Split.TEST}[name.upper()]

  @property
  def num_examples(self):
    return {Split.TRAIN_AND_VALID: 1281167, Split.TRAIN: 1271167,
            Split.VALID: 10000, Split.TEST: 50000}[self]


def load(
    split: Split,
    *,
    is_training: bool,
    # batch_dims should be:
    # [device_count, per_device_batch_size] or [total_batch_size]
    batch_dims: Sequence[int],
    augmentation_settings: Mapping[str, Any],
    # The shape to which images are resized.
    im_dim: int = INPUT_DIM,
    threadpool_size: int = 48,
    max_intra_op_parallelism: int = 1,
) -> Generator[Batch, None, None]:
  """Loads the given split of the dataset."""
  start, end = _shard(split, jax.host_id(), jax.host_count())

  im_size = (im_dim, im_dim)

  total_batch_size = np.prod(batch_dims)

  tfds_split = tfds.core.ReadInstruction(_to_tfds_split(split),
                                         from_=start, to=end, unit='abs')

  ds = tfds.load('imagenet2012:5.*.*', split=tfds_split,
                 decoders={'image': tfds.decode.SkipDecoding()})

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = threadpool_size
  options.experimental_threading.max_intra_op_parallelism = (
      max_intra_op_parallelism)
  options.experimental_optimization.map_parallelization = True
  if is_training:
    options.experimental_deterministic = False
  ds = ds.with_options(options)

  if is_training:
    if jax.host_count() > 1:
      # Only cache if we are reading a subset of the dataset.
      ds = ds.cache()
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)

  else:
    if split.num_examples % total_batch_size != 0:
      raise ValueError(f'Test/valid must be divisible by {total_batch_size}')

  def crop_augment_preprocess(example):
    image, _ = _preprocess_image(
        example['image'], is_training, im_size, augmentation_settings)

    label = tf.cast(example['label'], tf.int32)

    out = {'images': image, 'labels': label}

    if is_training:
      if augmentation_settings['cutmix']:
        out['mask'] = cutmix_padding(*im_size)
        out['cutmix_ratio'] = tf.reduce_mean(out['mask'])
      if augmentation_settings['mixup_alpha'] is not None:
        beta = tfp.distributions.Beta(
            augmentation_settings['mixup_alpha'],
            augmentation_settings['mixup_alpha'])
        out['mixup_ratio'] = beta.sample()
    return out

  ds = ds.map(crop_augment_preprocess, num_parallel_calls=AUTOTUNE)

  # Mixup/cutmix by temporarily batching (using the per-device batch size):
  use_cutmix = augmentation_settings['cutmix']
  use_mixup = augmentation_settings['mixup_alpha'] is not None
  if is_training and (use_cutmix or use_mixup):
    inner_batch_size = batch_dims[-1]
    # Apply mixup, cutmix, or mixup + cutmix on batched data.
    # We use data from 2 batches to produce 1 mixed batch.
    ds = ds.batch(inner_batch_size * 2)
    if not use_cutmix and use_mixup:
      ds = ds.map(my_mixup, num_parallel_calls=AUTOTUNE)
    elif use_cutmix and not use_mixup:
      ds = ds.map(my_cutmix, num_parallel_calls=AUTOTUNE)
    elif use_cutmix and use_mixup:
      ds = ds.map(my_mixup_cutmix, num_parallel_calls=AUTOTUNE)

    # Unbatch for further processing.
    ds = ds.unbatch()

  for batch_size in reversed(batch_dims):
    ds = ds.batch(batch_size)

  ds = ds.prefetch(AUTOTUNE)

  yield from tfds.as_numpy(ds)


# cutmix_padding, my_cutmix, my_mixup, and my_mixup_cutmix taken from:
# https://github.com/deepmind/deepmind-research/blob/master/nfnets/dataset.py
def cutmix_padding(h, w):
  """Returns image mask for CutMix.

  Taken from (https://github.com/google/edward2/blob/master/experimental
  /marginalization_mixup/data_utils.py#L367)
  Args:
    h: image height.
    w: image width.
  """
  r_x = tf.random.uniform([], 0, w, tf.int32)
  r_y = tf.random.uniform([], 0, h, tf.int32)

  # Beta dist in paper, but they used Beta(1,1) which is just uniform.
  image1_proportion = tf.random.uniform([])
  patch_length_ratio = tf.math.sqrt(1 - image1_proportion)
  r_w = tf.cast(patch_length_ratio * tf.cast(w, tf.float32), tf.int32)
  r_h = tf.cast(patch_length_ratio * tf.cast(h, tf.float32), tf.int32)
  bbx1 = tf.clip_by_value(tf.cast(r_x - r_w // 2, tf.int32), 0, w)
  bby1 = tf.clip_by_value(tf.cast(r_y - r_h // 2, tf.int32), 0, h)
  bbx2 = tf.clip_by_value(tf.cast(r_x + r_w // 2, tf.int32), 0, w)
  bby2 = tf.clip_by_value(tf.cast(r_y + r_h // 2, tf.int32), 0, h)

  # Create the binary mask.
  pad_left = bbx1
  pad_top = bby1
  pad_right = tf.maximum(w - bbx2, 0)
  pad_bottom = tf.maximum(h - bby2, 0)
  r_h = bby2 - bby1
  r_w = bbx2 - bbx1

  mask = tf.pad(
      tf.ones((r_h, r_w)),
      paddings=[[pad_top, pad_bottom], [pad_left, pad_right]],
      mode='CONSTANT',
      constant_values=0)
  mask.set_shape((h, w))
  return mask[..., None]  # Add channel dim.


def my_cutmix(batch):
  """Apply CutMix: https://arxiv.org/abs/1905.04899."""
  batch = dict(**batch)
  bs = tf.shape(batch['images'])[0] // 2
  mask = batch['mask'][:bs]
  images = (mask * batch['images'][:bs] + (1.0 - mask) * batch['images'][bs:])
  mix_labels = batch['labels'][bs:]
  labels = batch['labels'][:bs]
  ratio = batch['cutmix_ratio'][:bs]
  return {'images': images, 'labels': labels,
          'mix_labels': mix_labels, 'ratio': ratio}


def my_mixup(batch):
  """Apply mixup: https://arxiv.org/abs/1710.09412."""
  batch = dict(**batch)
  bs = tf.shape(batch['images'])[0] // 2
  ratio = batch['mixup_ratio'][:bs, None, None, None]
  images = (ratio * batch['images'][:bs] + (1.0 - ratio) * batch['images'][bs:])
  mix_labels = batch['labels'][bs:]
  labels = batch['labels'][:bs]
  ratio = ratio[..., 0, 0, 0]  # Unsqueeze
  return {'images': images, 'labels': labels,
          'mix_labels': mix_labels, 'ratio': ratio}


def my_mixup_cutmix(batch):
  """Apply mixup to half the batch, and cutmix to the other."""
  batch = dict(**batch)
  bs = tf.shape(batch['images'])[0] // 4
  mixup_ratio = batch['mixup_ratio'][:bs, None, None, None]
  mixup_images = (mixup_ratio * batch['images'][:bs]
                  + (1.0 - mixup_ratio) * batch['images'][bs:2*bs])
  mixup_labels = batch['labels'][:bs]
  mixup_mix_labels = batch['labels'][bs:2*bs]

  cutmix_mask = batch['mask'][2*bs:3*bs]
  cutmix_images = (cutmix_mask * batch['images'][2*bs:3*bs]
                   + (1.0 - cutmix_mask) * batch['images'][-bs:])
  cutmix_labels = batch['labels'][2*bs:3*bs]
  cutmix_mix_labels = batch['labels'][-bs:]
  cutmix_ratio = batch['cutmix_ratio'][2*bs : 3*bs]

  return {'images': tf.concat([mixup_images, cutmix_images], axis=0),
          'labels': tf.concat([mixup_labels, cutmix_labels], axis=0),
          'mix_labels': tf.concat([mixup_mix_labels, cutmix_mix_labels], 0),
          'ratio': tf.concat([mixup_ratio[..., 0, 0, 0], cutmix_ratio], axis=0)}


def _to_tfds_split(split: Split) -> tfds.Split:
  """Returns the TFDS split appropriately sharded."""
  # NOTE: Imagenet did not release labels for the test split used in the
  # competition, so it has been typical at DeepMind to consider the VALID
  # split the TEST split and to reserve 10k images from TRAIN for VALID.
  if split in (
      Split.TRAIN, Split.TRAIN_AND_VALID, Split.VALID):
    return tfds.Split.TRAIN
  else:
    assert split == Split.TEST
    return tfds.Split.VALIDATION


def _shard(
    split: Split, shard_index: int, num_shards: int) -> Tuple[int, int]:
  """Returns [start, end) for the given shard index."""
  assert shard_index < num_shards
  arange = np.arange(split.num_examples)
  shard_range = np.array_split(arange, num_shards)[shard_index]
  start, end = shard_range[0], (shard_range[-1] + 1)
  if split == Split.TRAIN:
    # Note that our TRAIN=TFDS_TRAIN[10000:] and VALID=TFDS_TRAIN[:10000].
    offset = Split.VALID.num_examples
    start += offset
    end += offset
  return start, end


def _preprocess_image(
    image_bytes: tf.Tensor,
    is_training: bool,
    image_size: Sequence[int],
    augmentation_settings: Mapping[str, Any],
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Returns processed and resized images."""

  # Get the image crop.
  if is_training:
    image, im_shape = _decode_and_random_crop(image_bytes)
    image = tf.image.random_flip_left_right(image)
  else:
    image, im_shape = _decode_and_center_crop(image_bytes)
  assert image.dtype == tf.uint8

  # Optionally apply RandAugment: https://arxiv.org/abs/1909.13719
  if is_training:
    if augmentation_settings['randaugment'] is not None:
      # Input and output images are dtype uint8.
      image = autoaugment.distort_image_with_randaugment(
          image,
          num_layers=augmentation_settings['randaugment']['num_layers'],
          magnitude=augmentation_settings['randaugment']['magnitude'])

  # Resize and normalize the image crop.
  # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
  # clamping overshoots. This means values returned will be outside the range
  # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
  image = tf.image.resize(
      image, image_size, tf.image.ResizeMethod.BICUBIC)
  image = _normalize_image(image)

  return image, im_shape


def _normalize_image(image: tf.Tensor) -> tf.Tensor:
  """Normalize the image to zero mean and unit variance."""
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image


def _distorted_bounding_box_crop(
    image_bytes: tf.Tensor,
    *,
    jpeg_shape: tf.Tensor,
    bbox: tf.Tensor,
    min_object_covered: float,
    aspect_ratio_range: Tuple[float, float],
    area_range: Tuple[float, float],
    max_attempts: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Generates cropped_image using one of the bboxes randomly distorted."""
  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      jpeg_shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = [offset_y, offset_x, target_height, target_width]

  if image_bytes.dtype == tf.dtypes.string:
    image = tf.image.decode_and_crop_jpeg(image_bytes,
                                          tf.stack(crop_window),
                                          channels=3)
  else:
    image = tf.image.crop_to_bounding_box(image_bytes, *crop_window)

  im_shape = tf.stack([target_height, target_width])
  return image, im_shape


def _decode_whole_image(image_bytes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  image = tf.io.decode_jpeg(image_bytes, channels=3)
  im_shape = tf.io.extract_jpeg_shape(image_bytes, output_type=tf.int32)
  return image, im_shape


def _decode_and_random_crop(
    image_bytes: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Make a random crop of INPUT_DIM."""

  if image_bytes.dtype == tf.dtypes.string:
    jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
  else:
    jpeg_shape = tf.shape(image_bytes)

  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image, im_shape = _distorted_bounding_box_crop(
      image_bytes,
      jpeg_shape=jpeg_shape,
      bbox=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3 / 4, 4 / 3),
      area_range=(0.08, 1.0),
      max_attempts=10)

  if tf.reduce_all(tf.equal(jpeg_shape, tf.shape(image))):
    # If the random crop failed fall back to center crop.
    image, im_shape = _decode_and_center_crop(image_bytes, jpeg_shape)
  return image, im_shape


def _center_crop(image, crop_dim):
  """Center crops an image to a target dimension."""
  image_height = image.shape[0]
  image_width = image.shape[1]
  offset_height = ((image_height - crop_dim) + 1) // 2
  offset_width = ((image_width - crop_dim) + 1) // 2
  return tf.image.crop_to_bounding_box(
      image, offset_height, offset_width, crop_dim, crop_dim)


def _decode_and_center_crop(
    image_bytes: tf.Tensor,
    jpeg_shape: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Crops to center of image with padding then scales."""
  if jpeg_shape is None:
    if image_bytes.dtype == tf.dtypes.string:
      jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
    else:
      jpeg_shape = tf.shape(image_bytes)

  image_height = jpeg_shape[0]
  image_width = jpeg_shape[1]

  padded_center_crop_size = tf.cast(
      ((INPUT_DIM / (INPUT_DIM + 32)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = [offset_height, offset_width,
                 padded_center_crop_size, padded_center_crop_size]

  if image_bytes.dtype == tf.dtypes.string:
    image = tf.image.decode_and_crop_jpeg(image_bytes,
                                          tf.stack(crop_window),
                                          channels=3)
  else:
    image = tf.image.crop_to_bounding_box(image_bytes, *crop_window)

  im_shape = tf.stack([padded_center_crop_size, padded_center_crop_size])
  return image, im_shape
