# Lint as: python3.
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Two dimensional convolutional neural net layers."""

from absl import logging
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


def weight_variable(shape, stddev=0.01):
  """Returns the weight variable."""
  logging.vlog(1, 'weight init for shape %s', str(shape))
  return tf.get_variable(
      'w', shape, initializer=tf.random_normal_initializer(stddev=stddev))


def bias_variable(shape):
  return tf.get_variable(
      'b', shape, initializer=tf.zeros_initializer())


def conv2d(x, w, atrou_rate=1, data_format='NHWC'):
  if atrou_rate > 1:
    return tf.nn.convolution(
        x,
        w,
        dilation_rate=[atrou_rate] * 2,
        padding='SAME',
        data_format=data_format)
  else:
    return tf.nn.conv2d(
        x, w, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)


def make_conv_sep2d_layer(input_node,
                          in_channels,
                          channel_multiplier,
                          out_channels,
                          layer_name,
                          filter_size,
                          filter_size_2=None,
                          batch_norm=False,
                          is_training=True,
                          atrou_rate=1,
                          data_format='NHWC',
                          stddev=0.01):
  """Use separable convolutions."""
  if filter_size_2 is None:
    filter_size_2 = filter_size
  logging.vlog(1, 'layer %s in %d out %d chan mult %d', layer_name, in_channels,
               out_channels, channel_multiplier)
  with tf.variable_scope(layer_name):
    with tf.variable_scope('depthwise'):
      w_depthwise = weight_variable(
          [filter_size, filter_size_2, in_channels, channel_multiplier],
          stddev=stddev)
    with tf.variable_scope('pointwise'):
      w_pointwise = weight_variable(
          [1, 1, in_channels * channel_multiplier, out_channels], stddev=stddev)
    h_conv = tf.nn.separable_conv2d(
        input_node,
        w_depthwise,
        w_pointwise,
        padding='SAME',
        strides=[1, 1, 1, 1],
        rate=[atrou_rate, atrou_rate],
        data_format=data_format)

    if batch_norm:
      h_conv = batch_norm_layer(
          h_conv, layer_name=layer_name, is_training=is_training,
          data_format=data_format)
    else:
      b_conv = bias_variable([out_channels])
      h_conv = tf.nn.bias_add(h_conv, b_conv, data_format=data_format)

    return h_conv


def batch_norm_layer(h_conv, layer_name, is_training=True, data_format='NCHW'):
  """Batch norm layer."""
  logging.vlog(1, 'batch norm for layer %s', layer_name)
  return tf.contrib.layers.batch_norm(
      h_conv,
      is_training=is_training,
      fused=True,
      decay=0.999,
      scope=layer_name,
      data_format=data_format)


def make_conv_layer(input_node,
                    in_channels,
                    out_channels,
                    layer_name,
                    filter_size,
                    filter_size_2=None,
                    non_linearity=True,
                    batch_norm=False,
                    is_training=True,
                    atrou_rate=1,
                    data_format='NHWC',
                    stddev=0.01):
  """Creates a convolution layer."""

  if filter_size_2 is None:
    filter_size_2 = filter_size
  logging.vlog(
      1, 'layer %s in %d out %d', layer_name, in_channels, out_channels)
  with tf.variable_scope(layer_name):
    w_conv = weight_variable(
        [filter_size, filter_size_2, in_channels, out_channels], stddev=stddev)
    h_conv = conv2d(
        input_node, w_conv, atrou_rate=atrou_rate, data_format=data_format)

    if batch_norm:
      h_conv = batch_norm_layer(
          h_conv, layer_name=layer_name, is_training=is_training,
          data_format=data_format)
    else:
      b_conv = bias_variable([out_channels])
      h_conv = tf.nn.bias_add(h_conv, b_conv, data_format=data_format)

    if non_linearity:
      h_conv = tf.nn.elu(h_conv)

    return h_conv
