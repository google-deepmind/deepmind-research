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

"""Test for the Transporter module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from transporter import transporter


IMAGE_H = 16
IMAGE_W = 16
IMAGE_C = 3
BATCH_SIZE = 4
IMAGE_BATCH_SHAPE = (BATCH_SIZE, IMAGE_H, IMAGE_W, IMAGE_C)

FILTERS = (16, 16, 32, 32, 64, 64)
STRIDES = (1, 1, 2, 1, 2, 1)
KERNEL_SIZES = (7, 3, 3, 3, 3, 3)


class TransporterTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {'norm_type': 'batch'},
      {'norm_type': 'layer'},
      {'norm_type': 'instance'})
  def test_output_shape(self, norm_type):
    encoder_ctor = transporter.Encoder
    encoder_kwargs = {
        'filters': FILTERS,
        'strides': STRIDES,
        'kernel_sizes': KERNEL_SIZES,
        'norm_type': norm_type,
    }
    decoder_filters = 4
    num_keypoints = 5
    gauss_std = 0.1

    encoder = encoder_ctor(name='encoder', **encoder_kwargs)
    keypoint_encoder = encoder_ctor(name='keypoint_encoder', **encoder_kwargs)
    keypointer = transporter.KeyPointer(keypoint_encoder=keypoint_encoder,
                                        num_keypoints=num_keypoints,
                                        gauss_std=gauss_std)

    decoder = transporter.Decoder(initial_filters=decoder_filters,
                                  output_size=[IMAGE_H, IMAGE_W],
                                  output_channels=IMAGE_C,
                                  norm_type=norm_type)
    model = transporter.Transporter(encoder=encoder,
                                    decoder=decoder,
                                    keypointer=keypointer)

    image_a = tf.random.normal(IMAGE_BATCH_SHAPE)
    image_b = tf.random.normal(IMAGE_BATCH_SHAPE)

    transporter_results = model(image_a, image_b, is_training=True)
    reconstructed_image_b = transporter_results['reconstructed_image_b']

    self.assertEqual(reconstructed_image_b.shape, IMAGE_BATCH_SHAPE)

  def testIncorrectEncoderShapes(self):
    """Test that a possible misconfiguration throws an error as expected.

    If the two encoders used produce different spatial sizes for their
    feature maps, this should cause an error when multiplying tensors together.
    """
    decoder_filters = 4
    num_keypoints = 5
    gauss_std = 0.1

    encoder = transporter.Encoder(
        filters=FILTERS,
        strides=STRIDES,
        kernel_sizes=KERNEL_SIZES)
    # Use less conv layers in this, in particular one less stride 2 layer, so
    # we will get a different spatial output resolution.
    keypoint_encoder = transporter.Encoder(
        filters=FILTERS[:-2],
        strides=STRIDES[:-2],
        kernel_sizes=KERNEL_SIZES[:-2])

    keypointer = transporter.KeyPointer(
        keypoint_encoder=keypoint_encoder,
        num_keypoints=num_keypoints,
        gauss_std=gauss_std)

    decoder = transporter.Decoder(
        initial_filters=decoder_filters,
        output_size=[IMAGE_H, IMAGE_W],
        output_channels=IMAGE_C)
    model = transporter.Transporter(
        encoder=encoder,
        decoder=decoder,
        keypointer=keypointer)

    with self.assertRaisesRegexp(ValueError, 'Dimensions must be equal'):
      model(tf.random.normal(IMAGE_BATCH_SHAPE),
            tf.random.normal(IMAGE_BATCH_SHAPE),
            is_training=True)


class EncoderTest(tf.test.TestCase):

  def test_output_shape(self):
    image_batch = tf.random.normal(shape=IMAGE_BATCH_SHAPE)

    filters = (4, 4, 8, 8, 16, 16)
    encoder = transporter.Encoder(filters=filters,
                                  strides=STRIDES,
                                  kernel_sizes=KERNEL_SIZES)

    features = encoder(image_batch, is_training=True)

    self.assertEqual(features.shape, (BATCH_SIZE,
                                      IMAGE_H // 4,
                                      IMAGE_W // 4,
                                      filters[-1]))


class KeyPointerTest(tf.test.TestCase):

  def test_output_shape(self):
    image_batch = tf.random.normal(shape=IMAGE_BATCH_SHAPE)
    num_keypoints = 6
    gauss_std = 0.1

    keypoint_encoder = transporter.Encoder(filters=FILTERS,
                                           strides=STRIDES,
                                           kernel_sizes=KERNEL_SIZES)
    keypointer = transporter.KeyPointer(keypoint_encoder=keypoint_encoder,
                                        num_keypoints=num_keypoints,
                                        gauss_std=gauss_std)

    keypointer_results = keypointer(image_batch, is_training=True)

    self.assertEqual(keypointer_results['centers'].shape,
                     (BATCH_SIZE, num_keypoints, 2))
    self.assertEqual(keypointer_results['heatmaps'].shape,
                     (BATCH_SIZE, IMAGE_H // 4, IMAGE_W // 4, num_keypoints))


class DecoderTest(tf.test.TestCase):

  def test_output_shape(self):
    feature_batch = tf.random.normal(shape=(BATCH_SIZE,
                                            IMAGE_H // 4,
                                            IMAGE_W // 4,
                                            64))

    decoder = transporter.Decoder(initial_filters=64,
                                  output_size=[IMAGE_H, IMAGE_W],
                                  output_channels=IMAGE_C)

    reconstructed_image_batch = decoder(feature_batch, is_training=True)

    self.assertEqual(reconstructed_image_batch.shape, IMAGE_BATCH_SHAPE)

  def test_encoder_decoder_output_shape(self):
    image_batch = tf.random.normal(shape=IMAGE_BATCH_SHAPE)

    encoder = transporter.Encoder(filters=FILTERS,
                                  strides=STRIDES,
                                  kernel_sizes=KERNEL_SIZES)
    decoder = transporter.Decoder(initial_filters=4,
                                  output_size=[IMAGE_H, IMAGE_W],
                                  output_channels=IMAGE_C)

    features = encoder(image_batch, is_training=True)
    reconstructed_images = decoder(features, is_training=True)

    self.assertEqual(reconstructed_images.shape, IMAGE_BATCH_SHAPE)


if __name__ == '__main__':
  tf.test.main()

