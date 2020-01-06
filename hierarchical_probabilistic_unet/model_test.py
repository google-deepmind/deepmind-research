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

"""Tests for the Hierarchical Probabilistic U-Net open-source version."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import HierarchicalProbUNet
import tensorflow.compat.v1 as tf


_NUM_CLASSES = 2
_BATCH_SIZE = 2
_SPATIAL_SHAPE = [32, 32]
_CHANNELS_PER_BLOCK = [5, 7, 9, 11, 13]
_IMAGE_SHAPE = [_BATCH_SIZE] + _SPATIAL_SHAPE + [1]
_BOTTLENECK_SIZE = _SPATIAL_SHAPE[0] // 2 ** (len(_CHANNELS_PER_BLOCK) - 1)
_SEGMENTATION_SHAPE = [_BATCH_SIZE] + _SPATIAL_SHAPE + [_NUM_CLASSES]
_LATENT_DIMS = [3, 2, 1]
_INITIALIZERS = {'w': tf.orthogonal_initializer(gain=1.0, seed=None),
                 'b': tf.truncated_normal_initializer(stddev=0.001)}


def _get_placeholders():
  """Returns placeholders for the image and segmentation."""
  img = tf.placeholder(dtype=tf.float32, shape=_IMAGE_SHAPE)
  seg = tf.placeholder(dtype=tf.float32, shape=_SEGMENTATION_SHAPE)
  return img, seg


class HierarchicalProbUNetTest(tf.test.TestCase):

  def test_shape_of_sample(self):
    hpu_net = HierarchicalProbUNet(latent_dims=_LATENT_DIMS,
                                   channels_per_block=_CHANNELS_PER_BLOCK,
                                   num_classes=_NUM_CLASSES,
                                   initializers=_INITIALIZERS)
    img, _ = _get_placeholders()
    sample = hpu_net.sample(img)
    self.assertEqual(sample.shape.as_list(), _SEGMENTATION_SHAPE)

  def test_shape_of_reconstruction(self):
    hpu_net = HierarchicalProbUNet(latent_dims=_LATENT_DIMS,
                                   channels_per_block=_CHANNELS_PER_BLOCK,
                                   num_classes=_NUM_CLASSES,
                                   initializers=_INITIALIZERS)
    img, seg = _get_placeholders()
    reconstruction = hpu_net.reconstruct(img, seg)
    self.assertEqual(reconstruction.shape.as_list(), _SEGMENTATION_SHAPE)

  def test_shapes_in_prior(self):
    hpu_net = HierarchicalProbUNet(latent_dims=_LATENT_DIMS,
                                   channels_per_block=_CHANNELS_PER_BLOCK,
                                   num_classes=_NUM_CLASSES,
                                   initializers=_INITIALIZERS)
    img, _ = _get_placeholders()
    prior_out = hpu_net._prior(img)
    distributions = prior_out['distributions']
    latents = prior_out['used_latents']
    encoder_features = prior_out['encoder_features']
    decoder_features = prior_out['decoder_features']

    # Test number of latent disctributions.
    self.assertEqual(len(distributions), len(_LATENT_DIMS))

    # Test shapes of latent scales.
    for level in range(len(_LATENT_DIMS)):
      latent_spatial_shape = _BOTTLENECK_SIZE * 2 ** level
      latent_shape = [_BATCH_SIZE, latent_spatial_shape, latent_spatial_shape,
                      _LATENT_DIMS[level]]
      self.assertEqual(latents[level].shape.as_list(), latent_shape)

    # Test encoder shapes.
    for level in range(len(_CHANNELS_PER_BLOCK)):
      spatial_shape = _SPATIAL_SHAPE[0] // 2 ** level
      feature_shape = [_BATCH_SIZE, spatial_shape, spatial_shape,
                       _CHANNELS_PER_BLOCK[level]]
      self.assertEqual(encoder_features[level].shape.as_list(), feature_shape)

    # Test decoder shape.
    start_level = len(_LATENT_DIMS)
    latent_spatial_shape = _BOTTLENECK_SIZE * 2 ** start_level
    latent_shape = [_BATCH_SIZE, latent_spatial_shape, latent_spatial_shape,
                    _CHANNELS_PER_BLOCK[::-1][start_level]]
    self.assertEqual(decoder_features.shape.as_list(), latent_shape)

  def test_shape_of_kl(self):
    hpu_net = HierarchicalProbUNet(latent_dims=_LATENT_DIMS,
                                   channels_per_block=_CHANNELS_PER_BLOCK,
                                   num_classes=_NUM_CLASSES,
                                   initializers=_INITIALIZERS)
    img, seg = _get_placeholders()
    kl_dict = hpu_net.kl(img, seg)
    self.assertEqual(len(kl_dict), len(_LATENT_DIMS))


if __name__ == '__main__':
  tf.test.main()


