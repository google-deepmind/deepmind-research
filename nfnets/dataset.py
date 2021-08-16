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
"""ImageNet dataset with typical pre-processing and advanced augs."""
# pylint: disable=logging-format-interpolation

import enum
import itertools as it
import logging
import re
from typing import Generator, Mapping, Optional, Sequence, Text, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from nfnets import autoaugment


Batch = Mapping[Text, np.ndarray]
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Split(enum.Enum):
  """Imagenet dataset split."""
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
    batch_dims: Sequence[int],
    name: str = 'imagenet',
    dtype: jnp.dtype = jnp.float32,
    transpose: bool = False,
    fake_data: bool = False,
    image_size: Tuple[int, int] = (224, 224),
    augment_name: Optional[str] = None,
    eval_preproc: str = 'crop_resize',
    augment_before_mix: bool = True,
) -> Generator[Batch, None, None]:
  """Loads the given split of the dataset.

  Args:
    split: Dataset split to use.
    is_training: If true, use training preproc and augmentation.
    batch_dims: List indicating how to batch the dataset (typically expected to
      be of shape (num_devices, bs_per_device)
    name: Which dataset to use, (must be 'imagenet')
    dtype: One of float32 or bfloat16 (bf16 may not be supported fully)
    transpose: If true, employs double transpose trick.
    fake_data: Return batches of fake data for debugging purposes.
    image_size: Final image size returned by dataset pipeline. Note that the
      exact procedure to arrive at this size will depend on the chosen preproc.
    augment_name: Optional additional aug strategy (applied atop the default
      of distorted bboxes and random L/R flips). Specified with a string
      such as 'cutmix_mixup_0.4_randaugment_415'. See README for deets.
    eval_preproc: Eval preproc method, either 'crop_resize' (crop on the long
      edge then resize) or `resize_crop_{pct}`, which will resize the image to
      `image_size / pct` on each side then take a center crop.
    augment_before_mix: Apply augs like RA/AA before or after cutmix/mixup.

  Yields:
    A TFDS numpy iterator.
  """
  start, end = _shard(split, jax.host_id(), jax.host_count())

  if fake_data:
    print('Using fake data!')
    images = np.zeros(tuple(batch_dims) + image_size + (3,), dtype=dtype)
    labels = np.zeros(tuple(batch_dims), dtype=np.int32)
    if transpose:
      axes = tuple(range(images.ndim))
      axes = axes[:-4] + axes[-3:] + (axes[-4],)  # NHWC -> HWCN
      images = np.transpose(images, axes)
    yield from it.repeat({'images': images, 'labels': labels}, end - start)
    return

  total_batch_size = np.prod(batch_dims)

  if name.lower() == 'imagenet':
    tfds_split = tfds.core.ReadInstruction(_to_tfds_split(split),
                                           from_=start, to=end, unit='abs')

    ds = tfds.load('imagenet2012:5.*.*', split=tfds_split,
                   decoders={'image': tfds.decode.SkipDecoding()})
  else:
    raise ValueError('Only imagenet is presently supported for this dataset.')

  options = ds.options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  options.experimental_optimization.map_parallelization = True
  options.experimental_optimization.parallel_batch = True
  options.experimental_optimization.hoist_random_uniform = True

  if is_training:
    options.experimental_deterministic = False

  if is_training:
    if jax.host_count() > 1:
      # Only cache if we are reading a subset of the dataset.
      ds = ds.cache()
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=None)

  else:
    if split.num_examples % total_batch_size != 0:
      raise ValueError(f'Test/valid must be divisible by {total_batch_size}')

  def augment_normalize(batch):
    """Optionally augment, then normalize an image."""
    batch = dict(**batch)
    image = _augment_image(batch['images'], is_training, augment_name)
    batch['images'] = _normalize_image(image)
    return batch

  def preprocess(example):
    image = _preprocess_image(example['image'], is_training, image_size,
                              eval_preproc)
    label = tf.cast(example['label'], tf.int32)
    out = {'images': image, 'labels': label}
    if augment_name is not None and 'cutmix' in augment_name:
      out['mask'] = cutmix_padding(*image_size)
      out['cutmix_ratio'] = tf.reduce_mean(out['mask'])
    if augment_name is not None and 'mixup' in augment_name:
      mixup_alpha = 0.2  # default to alpha=0.2
      # If float provided, get it
      if 'mixup_' in augment_name:
        alpha = augment_name.split('mixup_')[1].split('_')
        if any(alpha) and re.match(r'^-?\d+(?:\.\d+)?$', alpha[0]) is not None:
          mixup_alpha = float(alpha[0])
      beta = tfp.distributions.Beta(mixup_alpha, mixup_alpha)
      out['mixup_ratio'] = beta.sample()
    # Apply augs before mixing?
    if augment_before_mix or augment_name is None:
      out = augment_normalize(out)
    return out

  ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
  ds = ds.prefetch(AUTOTUNE)

  def transpose_fn(batch):
    # Applies the double-transpose trick for TPU.
    batch = dict(**batch)
    batch['images'] = tf.transpose(batch['images'], (1, 2, 3, 0))
    return batch

  def cast_fn(batch):
    batch = dict(**batch)
    batch['images'] = tf.cast(batch['images'], _to_tf_dtype(dtype))
    return batch

  for i, batch_size in enumerate(reversed(batch_dims)):
    if i == 0:
      # Deal with vectorized MixUp + CutMix ops
      if augment_name is not None:
        if 'mixup' in augment_name or 'cutmix' in augment_name:
          ds = ds.batch(batch_size * 2)
        else:
          ds = ds.map(augment_normalize, num_parallel_calls=AUTOTUNE)
          ds = ds.batch(batch_size)
        # Apply mixup, cutmix, or mixup + cutmix
        if 'mixup' in augment_name and 'cutmix' not in augment_name:
          logging.info('Applying MixUp!')
          ds = ds.map(my_mixup, num_parallel_calls=AUTOTUNE)
        elif 'cutmix' in augment_name and 'mixup' not in augment_name:
          logging.info('Applying CutMix!')
          ds = ds.map(my_cutmix, num_parallel_calls=AUTOTUNE)
        elif 'mixup' in augment_name and 'cutmix' in augment_name:
          logging.info('Applying MixUp and CutMix!')
          ds = ds.map(my_mixup_cutmix, num_parallel_calls=AUTOTUNE)
        # If applying augs after mixing, unbatch, map, and rebatch
        if (not augment_before_mix and
            ('mixup' in augment_name or 'cutmix' in augment_name)):
          ds = ds.unbatch().map(augment_normalize, num_parallel_calls=AUTOTUNE)
          ds = ds.batch(batch_size)
      else:
        ds = ds.batch(batch_size)
      # Transpose and cast as needbe
      if transpose:
        ds = ds.map(transpose_fn)  # NHWC -> HWCN
      # NOTE: You may be tempted to move the casting earlier on in the pipeline,
      # but for bf16 some operations will end up silently placed on the TPU and
      # this causes stalls while TF and JAX battle for the accelerator.
      ds = ds.map(cast_fn)
    else:
      ds = ds.batch(batch_size)

  ds = ds.prefetch(AUTOTUNE)
  ds = tfds.as_numpy(ds)

  if dtype == jnp.bfloat16:
    # JAX and TF disagree on the NumPy bfloat16 type so we need to reinterpret
    # tf_bfloat16->jnp.bfloat16.
    for batch in ds:
      batch['images'] = batch['images'].view(jnp.bfloat16)
      yield batch
  else:
    yield from ds


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
  """Cutmix."""
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
  """Mixup."""
  batch = dict(**batch)
  bs = tf.shape(batch['images'])[0] // 2
  ratio = batch['mixup_ratio'][:bs, None, None, None]
  images = (ratio * batch['images'][:bs] + (1.0 - ratio) * batch['images'][bs:])
  mix_labels = batch['labels'][bs:]
  labels = batch['labels'][:bs]
  ratio = ratio[..., 0, 0, 0]  # Unsqueeze
  return {'images': images, 'labels': labels,
          'mix_labels': mix_labels, 'ratio': ratio}


def mixup_or_cutmix(batch):
  """Randomly applies one of cutmix or mixup to a batch."""
  logging.info('Randomly applying cutmix or mixup with 50% chance!')
  return tf.cond(
      tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool),
      lambda: my_mixup(batch),
      lambda: my_cutmix(batch))


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


def _to_tf_dtype(jax_dtype: jnp.dtype) -> tf.DType:
  if jax_dtype == jnp.bfloat16:
    return tf.bfloat16
  else:
    return tf.dtypes.as_dtype(jax_dtype)


def _to_tfds_split(split: Split) -> tfds.Split:
  """Returns the TFDS split appropriately sharded."""
  if split in (Split.TRAIN, Split.TRAIN_AND_VALID, Split.VALID):
    return tfds.Split.TRAIN
  else:
    assert split == Split.TEST
    return tfds.Split.VALIDATION


def _shard(split: Split, shard_index: int, num_shards: int) -> Tuple[int, int]:
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
    eval_preproc: str = 'crop_resize'
) -> tf.Tensor:
  """Returns processed and resized images."""
  # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
  # clamping overshoots. This means values returned will be outside the range
  # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
  if is_training:
    image = _decode_and_random_crop(image_bytes, image_size)
    image = tf.image.random_flip_left_right(image)
    assert image.dtype == tf.uint8
    image = tf.image.resize(image, image_size, tf.image.ResizeMethod.BICUBIC)
  else:
    if eval_preproc == 'crop_resize':
      image = _decode_and_center_crop(image_bytes, image_size=image_size)
      assert image.dtype == tf.uint8
      image = tf.image.resize(image, image_size, tf.image.ResizeMethod.BICUBIC)
    elif 'resize_crop' in eval_preproc:
      # Pass in crop percent
      crop_pct = float(eval_preproc.split('_')[-1])
      image = _decode_and_resize_then_crop(image_bytes, image_size=image_size,
                                           crop_pct=crop_pct)
    else:
      raise ValueError(f'Unknown Eval Preproc {eval_preproc} provided!')
  return image


def _augment_image(
    image: tf.Tensor,
    is_training: bool,
    augment_name: Optional[str] = None,
) -> tf.Tensor:
  """Applies AA/RA to an image."""
  if is_training and augment_name:
    if 'autoaugment' in augment_name or 'randaugment' in augment_name:
      input_image_type = image.dtype
      image = tf.clip_by_value(image, 0.0, 255.0)
      # Autoaugment requires a uint8 image; we cast here and then cast back
      image = tf.cast(image, dtype=tf.uint8)
      if 'autoaugment' in augment_name:
        logging.info(f'Applying AutoAugment policy {augment_name}')
        image = autoaugment.distort_image_with_autoaugment(image, 'v0')
      elif 'randaugment' in augment_name:
        magnitude = int(augment_name.split('_')[-1])  # pytype: disable=attribute-error
        # Allow passing in num_layers as a magnitude > 100
        if magnitude > 100:
          num_layers = magnitude // 100
          magnitude = magnitude - int(num_layers * 100)
        else:
          num_layers = 2
        logging.info(f'Applying RA {num_layers} x {magnitude}')
        image = autoaugment.distort_image_with_randaugment(
            image, num_layers=num_layers, magnitude=magnitude)
      image = tf.cast(image, dtype=input_image_type)
  return image


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
) -> tf.Tensor:
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
  image = crop(image_bytes, crop_window)
  return image


def _decode_and_random_crop(image_bytes: tf.Tensor,
                            image_size: Sequence[int] = (224, 224),
                            jpeg_shape: Optional[tf.Tensor] = None
                            ) -> tf.Tensor:
  """Make a random crop of chosen size."""
  if jpeg_shape is None:
    jpeg_shape = get_shape(image_bytes)
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = _distorted_bounding_box_crop(
      image_bytes,
      jpeg_shape=jpeg_shape,
      bbox=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3 / 4, 4 / 3),
      area_range=(0.08, 1.0),
      max_attempts=10)
  if tf.reduce_all(tf.equal(jpeg_shape, tf.shape(image))):
    # If the random crop failed fall back to center crop.
    image = _decode_and_center_crop(image_bytes, jpeg_shape, image_size)
  return image


def _decode_and_center_crop(
    image_bytes: tf.Tensor,
    jpeg_shape: Optional[tf.Tensor] = None,
    image_size: Sequence[int] = (224, 224),
) -> tf.Tensor:
  """Crops to center of image with padding then scales."""
  if jpeg_shape is None:
    jpeg_shape = get_shape(image_bytes)
  image_height = jpeg_shape[0]
  image_width = jpeg_shape[1]
  # Pad the image with at least 32px on the short edge and take a
  # crop that maintains aspect ratio.
  scale = tf.minimum(tf.cast(image_height, tf.float32) / (image_size[0] + 32),
                     tf.cast(image_width, tf.float32) / (image_size[1] + 32))
  padded_center_crop_height = tf.cast(scale * image_size[0], tf.int32)
  padded_center_crop_width = tf.cast(scale * image_size[1], tf.int32)
  offset_height = ((image_height - padded_center_crop_height) + 1) // 2
  offset_width = ((image_width - padded_center_crop_width) + 1) // 2
  crop_window = [offset_height, offset_width,
                 padded_center_crop_height, padded_center_crop_width]
  image = crop(image_bytes, crop_window)
  return image


def get_shape(image_bytes):
  """Gets the image shape for jpeg bytes or a uint8 decoded image."""
  if image_bytes.dtype == tf.dtypes.string:
    image_shape = tf.image.extract_jpeg_shape(image_bytes)
  else:
    image_shape = tf.shape(image_bytes)
  return image_shape


def crop(image_bytes, crop_window):
  """Helper function to crop a jpeg or a decoded image."""
  if image_bytes.dtype == tf.dtypes.string:
    image = tf.image.decode_and_crop_jpeg(image_bytes,
                                          tf.stack(crop_window),
                                          channels=3)
  else:
    image = tf.image.crop_to_bounding_box(image_bytes, *crop_window)
  return image


def _decode_and_resize_then_crop(
    image_bytes: tf.Tensor,
    image_size: Sequence[int] = (224, 224),
    crop_pct: float = 1.0,
) -> tf.Tensor:
  """Rescales an image to image_size / crop_pct, then center crops."""
  image = tf.image.decode_jpeg(image_bytes, channels=3)
  # Scale image to "scaled size" before taking a center crop
  if crop_pct > 1.0:  # If crop_pct is >1, treat it as num pad pixels (like VGG)
    scale_size = tuple([int(x + crop_pct) for x in image_size])
  else:
    scale_size = tuple([int(float(x) / crop_pct) for x in image_size])
  image = tf.image.resize(image, scale_size, tf.image.ResizeMethod.BICUBIC)
  crop_height = tf.cast(image_size[0], tf.int32)
  crop_width = tf.cast(image_size[1], tf.int32)
  offset_height = ((scale_size[0] - crop_height) + 1) // 2
  offset_width = ((scale_size[1] - crop_width) + 1) // 2
  crop_window = [offset_height, offset_width, crop_height, crop_width]
  image = crop(image, crop_window)
  return image
