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

"""Helpers to pre-process Antennae galaxy images."""

import collections
import os

from astropy.io import fits
import numpy as np
from scipy import ndimage
import tensorflow.compat.v2 as tf


def norm_antennae_images(images, scale=1000):
  return tf.math.asinh(images/scale)


def renorm_antennae(images):
  median = np.percentile(images.numpy().flatten(), 50)
  img_range = np.ptp(images.numpy().flatten())
  return (images - median) / (img_range / 2)


def get_antennae_images(antennae_fits_dir):
  """Load the raw Antennae galaxy images."""
  all_fits_files = [
      os.path.join(antennae_fits_dir, f)
      for f in os.listdir(antennae_fits_dir)
  ]
  freq_mapping = {'red': 160, 'blue': 850}

  paired_fits_files = collections.defaultdict(list)
  for f in all_fits_files:
    redshift = float(f[-8:-5])
    paired_fits_files[redshift].append(f)

  for redshift, files in paired_fits_files.items():
    paired_fits_files[redshift] = sorted(
        files, key=lambda f: freq_mapping[f.split('/')[-1].split('_')[0]])

  print('Reading files:', paired_fits_files)
  print('Redshifts:', sorted(paired_fits_files.keys()))

  galaxy_views = collections.defaultdict(list)
  for redshift in paired_fits_files:
    for view_path in paired_fits_files[redshift]:
      with open(view_path, 'rb') as f:
        fits_data = fits.open(f)
        galaxy_views[redshift].append(np.array(fits_data[0].data))

  batched_images = []
  for redshift in paired_fits_files:
    img = tf.constant(np.array(galaxy_views[redshift]))
    img = tf.transpose(img, (1, 2, 0))
    img = tf.image.resize(img, size=(60, 60))
    batched_images.append(img)

  return tf.stack(batched_images)


def preprocess_antennae_images(antennae_images):
  """Pre-process the Antennae galaxy images into a reasonable range."""
  rotated_antennae_images = [
      ndimage.rotate(img, 10, reshape=True, cval=-1)[10:-10, 10:-10]
      for img in antennae_images
  ]
  rotated_antennae_images = [
      np.clip(img, 0, 1e9) for img in rotated_antennae_images
  ]
  rotated_antennae_images = tf.stack(rotated_antennae_images)
  normed_antennae_images = norm_antennae_images(rotated_antennae_images)
  normed_antennae_images = tf.clip_by_value(normed_antennae_images, 1, 4.5)
  renormed_antennae_images = renorm_antennae(normed_antennae_images)
  return renormed_antennae_images
