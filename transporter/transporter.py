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

"""Transporter architecture in Sonnet/TF 1: https://arxiv.org/abs/1906.11883."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import sonnet as snt
import tensorflow.compat.v1 as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers

nest = contrib_framework.nest

# Paper submission used BatchNorm, but we have since found that Layer & Instance
# norm can be quite a lot more stable.
_NORMALIZATION_CTORS = {
    "layer": snt.LayerNorm,
    "instance": functools.partial(snt.LayerNorm, axis=[1, 2]),
    "batch": snt.BatchNormV2,
}


def _connect_module_with_kwarg_if_supported(module,
                                            input_tensor,
                                            kwarg_name,
                                            kwarg_value):
  """Connects a module to some input, plus a kwarg= if supported by module."""
  if snt.supports_kwargs(module, kwarg_name) == "supported":
    kwargs = {kwarg_name: kwarg_value}
  else:
    kwargs = {}
  return module(input_tensor, **kwargs)


class Transporter(snt.AbstractModule):
  """Sonnet module implementing the Transporter architecture."""

  def __init__(
      self,
      encoder,
      keypointer,
      decoder,
      name="transporter"):
    """Initialize the Transporter module.

    Args:
      encoder: `snt.AbstractModule` mapping images to features (see `Encoder`)
      keypointer: `snt.AbstractModule` mapping images to keypoint masks (see
          `KeyPointer`)
      decoder: `snt.AbstractModule` decoding features to images (see `Decoder`)
      name: `str` module name
    """
    super(Transporter, self).__init__(name=name)

    self._encoder = encoder
    self._decoder = decoder
    self._keypointer = keypointer

  def _build(self, image_a, image_b, is_training):
    """Reconstructs image_b using feature transport from image_a.

    This approaches matches the NeurIPS submission.

    Args:
      image_a: Tensor of shape [B, H, W, C] containing a batch of images.
      image_b: Tensor of shape [B, H, W, C] containing a batch of images.
      is_training: `bool` indication whether the model is in training mode.

    Returns:
      A dict containing keys:
        'reconstructed_image_b': Reconstruction of image_b, with the same shape.
        'features_a': Tensor of shape [B, F_h, F_w, N] of the extracted features
            for `image_a`.
        'features_b': Tensor of shape [B, F_h, F_w, N] of the extracted features
            for `image_b`.
        'keypoints_a': The result of the keypointer module on image_a, with stop
          gradients applied.
        'keypoints_b': The result of the keypointer module on image_b.
    """
    # Process both images. All gradients related to image_a are stopped.
    image_a_features = tf.stop_gradient(
        self._encoder(image_a, is_training=is_training))
    image_a_keypoints = nest.map_structure(
        tf.stop_gradient, self._keypointer(image_a, is_training=is_training))

    image_b_features = self._encoder(image_b, is_training=is_training)
    image_b_keypoints = self._keypointer(image_b, is_training=is_training)

    # Transport features
    num_keypoints = image_a_keypoints["heatmaps"].shape[-1]
    transported_features = image_a_features
    for k in range(num_keypoints):
      mask_a = image_a_keypoints["heatmaps"][..., k, None]
      mask_b = image_b_keypoints["heatmaps"][..., k, None]

      # suppress features from image a, around both keypoint locations.
      transported_features = (
          (1 - mask_a) * (1 - mask_b) * transported_features)

      # copy features from image b around keypoints for image b.
      transported_features += (mask_b * image_b_features)

    reconstructed_image_b = self._decoder(
        transported_features, is_training=is_training)

    return {
        "reconstructed_image_b": reconstructed_image_b,
        "features_a": image_a_features,
        "features_b": image_b_features,
        "keypoints_a": image_a_keypoints,
        "keypoints_b": image_b_keypoints,
    }


def reconstruction_loss(image, predicted_image, loss_type="l2"):
  """Returns the reconstruction loss between the image and the predicted_image.

  Args:
    image: target image tensor of shape [B, H, W, C]
    predicted_image: reconstructed image as returned by the model
    loss_type: `str` reconstruction loss, either `l2` (default) or `l1`.

  Returns:
    The reconstruction loss
  """

  if loss_type == "l2":
    return tf.reduce_mean(tf.square(image - predicted_image))
  elif loss_type == "l1":
    return tf.reduce_mean(tf.abs(image - predicted_image))
  else:
    raise ValueError("Unknown loss type: {}".format(loss_type))


class Encoder(snt.AbstractModule):
  """Encoder module mapping an image to features.

  The encoder is a standard convolutional network with ReLu activations.
  """

  def __init__(
      self,
      filters=(16, 16, 32, 32),
      kernel_sizes=(7, 3, 3, 3),
      strides=(1, 1, 2, 1),
      norm_type="batch",
      name="encoder"):
    """Initialize the Encoder.

    Args:
      filters: tuple of `int`. The ith layer of the encoder will
        consist of `filters[i]` filters.
      kernel_sizes: tuple of `int` kernel sizes for each layer
      strides: tuple of `int` strides for each layer
      norm_type: string, one of 'instance', 'layer', 'batch'.
      name: `str` name of the module.
    """
    super(Encoder, self).__init__(name=name)
    if len({len(filters), len(kernel_sizes), len(strides)}) != 1:
      raise ValueError(
          "length of filters/kernel_sizes/strides lists must be the same")
    self._filters = filters
    self._kernels = kernel_sizes
    self._strides = strides
    self._norm_ctor = _NORMALIZATION_CTORS[norm_type]

  def _build(self, image, is_training):
    """Connect the Encoder.

    Args:
      image: A batch of images of shape [B, H, W, C]
      is_training: `bool` indicating if the model is in training mode.

    Returns:
      A tensor of features of shape [B, F_h, F_w, N] where F_h and F_w are the
       height and width of the feature map and N = 4 * `self._filters`
    """
    regularizers = {"w": contrib_layers.l2_regularizer(1.0)}

    features = image
    for l in range(len(self._filters)):
      with tf.variable_scope("conv_{}".format(l + 1)):
        conv = snt.Conv2D(
            self._filters[l],
            self._kernels[l],
            self._strides[l],
            padding=snt.SAME,
            regularizers=regularizers,
            name="conv_{}".format(l+1))
        norm_module = self._norm_ctor(name="normalization")

      features = conv(features)
      features = _connect_module_with_kwarg_if_supported(
          norm_module, features, "is_training", is_training)
      features = tf.nn.relu(features)

    return features


class KeyPointer(snt.AbstractModule):
  """Module for extracting keypoints from an image."""

  def __init__(self,
               num_keypoints,
               gauss_std,
               keypoint_encoder,
               custom_getter=None,
               name="key_pointer"):
    """Iniitialize the keypointer.

    Args:
      num_keypoints: `int` number of keypoints to extract
      gauss_std: `float` size of the keypoints, relative to the image dimensions
        normalized to the range [-1, 1]
      keypoint_encoder: sonnet Module which produces a feature map. Must accept
        an is_training kwarg. When used in the Transporter, the output spatial
        resolution of this encoder should match the output spatial resolution
        of the other encoder, although these two encoders should not share
        weights.
      custom_getter: optional custom getter for variables in this module.
      name: `str` name of the module
    """
    super(KeyPointer, self).__init__(name=name, custom_getter=custom_getter)
    self._num_keypoints = num_keypoints
    self._gauss_std = gauss_std
    self._keypoint_encoder = keypoint_encoder

  def _build(self, image, is_training):
    """Compute the gaussian keypoints for the image.

    Args:
      image: Image tensor of shape [B, H, W, C]
      is_training: `bool` whether the model is in training or evaluation mode

    Returns:
      a dict with keys:
        'centers': A tensor of shape [B, K, 2] of the center locations for each
            of the K keypoints.
        'heatmaps': A tensor of shape [B, F_h, F_w, K] of gaussian maps over the
            keypoints, where [F_h, F_w] is the size of the keypoint_encoder
            feature maps.
    """
    conv = snt.Conv2D(
        self._num_keypoints, [1, 1],
        stride=1,
        regularizers={"w": contrib_layers.l2_regularizer(1.0)},
        name="conv_1/conv_1")

    image_features = self._keypoint_encoder(image, is_training=is_training)
    keypoint_features = conv(image_features)
    return get_keypoint_data_from_feature_map(
        keypoint_features, self._gauss_std)


def get_keypoint_data_from_feature_map(feature_map, gauss_std):
  """Returns keypoint information from a feature map.

  Args:
    feature_map: [B, H, W, K] Tensor, should be activations from a convnet.
    gauss_std: float, the standard deviation of the gaussians to be put around
      the keypoints.

  Returns:
    a dict with keys:
      'centers': A tensor of shape [B, K, 2] of the center locations for each
          of the K keypoints.
      'heatmaps': A tensor of shape [B, H, W, K] of gaussian maps over the
          keypoints.
  """
  gauss_mu = _get_keypoint_mus(feature_map)
  map_size = feature_map.shape.as_list()[1:3]
  gauss_maps = _get_gaussian_maps(gauss_mu, map_size, 1.0 / gauss_std)

  return {
      "centers": gauss_mu,
      "heatmaps": gauss_maps,
  }


def _get_keypoint_mus(keypoint_features):
  """Returns the keypoint center points.

  Args:
    keypoint_features: A tensor of shape [B, F_h, F_w, K] where K is the number
      of keypoints to extract.

  Returns:
    A tensor of shape [B, K, 2] of the y, x center points of each keypoint. Each
      center point are in the range [-1, 1]^2. Note: the first element is the y
      coordinate, the second is the x coordinate.
  """
  gauss_y = _get_coord(keypoint_features, 1)
  gauss_x = _get_coord(keypoint_features, 2)
  gauss_mu = tf.stack([gauss_y, gauss_x], axis=2)
  return gauss_mu


def _get_coord(features, axis):
  """Returns the keypoint coordinate encoding for the given axis.

  Args:
    features: A tensor of shape [B, F_h, F_w, K] where K is the number of
      keypoints to extract.
    axis: `int` which axis to extract the coordinate for. Has to be axis 1 or 2.

  Returns:
    A tensor of shape [B, K] containing the keypoint centers along the given
      axis. The location is given in the range [-1, 1].
  """
  if axis != 1 and axis != 2:
    raise ValueError("Axis needs to be 1 or 2.")

  other_axis = 1 if axis == 2 else 2
  axis_size = features.shape[axis]

  # Compute the normalized weight for each row/column along the axis
  g_c_prob = tf.reduce_mean(features, axis=other_axis)
  g_c_prob = tf.nn.softmax(g_c_prob, axis=1)

  # Linear combination of the interval [-1, 1] using the normalized weights to
  # give a single coordinate in the same interval [-1, 1]
  scale = tf.cast(tf.linspace(-1.0, 1.0, axis_size), tf.float32)
  scale = tf.reshape(scale, [1, axis_size, 1])
  coordinate = tf.reduce_sum(g_c_prob * scale, axis=1)
  return coordinate


def _get_gaussian_maps(mu, map_size, inv_std, power=2):
  """Transforms the keypoint center points to a gaussian masks."""
  mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

  y = tf.cast(tf.linspace(-1.0, 1.0, map_size[0]), tf.float32)
  x = tf.cast(tf.linspace(-1.0, 1.0, map_size[1]), tf.float32)

  mu_y, mu_x = tf.expand_dims(mu_y, -1), tf.expand_dims(mu_x, -1)

  y = tf.reshape(y, [1, 1, map_size[0], 1])
  x = tf.reshape(x, [1, 1, 1, map_size[1]])

  g_y = tf.pow(y - mu_y, power)
  g_x = tf.pow(x - mu_x, power)
  dist = (g_y + g_x) * tf.pow(inv_std, power)
  g_yx = tf.exp(-dist)

  g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])
  return g_yx


class Decoder(snt.AbstractModule):
  """Decoder reconstruction network.

  The decoder is a standard convolutional network with ReLu activations.
  """

  def __init__(self, initial_filters, output_size,
               output_channels=3,
               norm_type="batch",
               name="decoder"):
    """Initialize the decoder.

    Args:
      initial_filters: `int` number of initial filters used in the decoder
      output_size: tuple of `int` height and width of the reconstructed image
      output_channels: `int` number of output channels, for RGB use 3 (default)
      norm_type: string, one of 'instance', 'layer', 'batch'.
      name: `str` name of the module
    """
    super(Decoder, self).__init__(name=name)
    self._initial_filters = initial_filters
    self._output_height = output_size[0]
    self._output_width = output_size[1]
    self._output_channels = output_channels
    self._norm_ctor = _NORMALIZATION_CTORS[norm_type]

  def _build(self, features, is_training):
    """Connect the Decoder.

    Args:
      features: Tensor of shape [B, F_h, F_w, N]
      is_training: `bool` whether the module is in training mode.

    Returns:
      A reconstructed image tensor of shape [B, output_height, output_width,
          output_channels]
    """
    height, width = features.shape.as_list()[1:3]

    filters = self._initial_filters
    regularizers = {"w": contrib_layers.l2_regularizer(1.0)}

    layer = 0

    while height <= self._output_height:
      layer += 1
      with tf.variable_scope("conv_{}".format(layer)):
        conv1 = snt.Conv2D(
            filters,
            [3, 3],
            stride=1,
            regularizers=regularizers,
            name="conv_{}".format(layer))
        norm_module = self._norm_ctor(name="normalization")

      features = conv1(features)
      features = _connect_module_with_kwarg_if_supported(
          norm_module, features, "is_training", is_training)
      features = tf.nn.relu(features)

      if height == self._output_height:
        layer += 1
        with tf.variable_scope("conv_{}".format(layer)):
          conv2 = snt.Conv2D(
              self._output_channels,
              [3, 3],
              stride=1,
              regularizers=regularizers,
              name="conv_{}".format(layer))
        features = conv2(features)
        break
      else:
        layer += 1
        with tf.variable_scope("conv_{}".format(layer)):
          conv2 = snt.Conv2D(
              filters,
              [3, 3],
              stride=1,
              regularizers=regularizers,
              name="conv_{}".format(layer))
          norm_module = self._norm_ctor(name="normalization")

        features = conv2(features)
        features = _connect_module_with_kwarg_if_supported(
            norm_module, features, "is_training", is_training)
        features = tf.nn.relu(features)

      height *= 2
      width *= 2
      features = tf.image.resize(features, [height, width])

      if filters >= 8:
        filters /= 2

    assert height == self._output_height
    assert width == self._output_width

    return features
