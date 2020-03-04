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
"""2D Resnet."""

from absl import logging
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from alphafold_casp13 import two_dim_convnet


def make_sep_res_layer(
    input_node,
    in_channels,
    out_channels,
    layer_name,
    filter_size,
    filter_size_2=None,
    batch_norm=False,
    is_training=True,
    divide_channels_by=2,
    atrou_rate=1,
    channel_multiplier=0,
    data_format='NHWC',
    stddev=0.01,
    dropout_keep_prob=1.0):
  """A separable resnet block."""

  with tf.name_scope(layer_name):
    input_times_almost_1 = input_node
    h_conv = input_times_almost_1

    if batch_norm:
      h_conv = two_dim_convnet.batch_norm_layer(
          h_conv, layer_name=layer_name, is_training=is_training,
          data_format=data_format)

    h_conv = tf.nn.elu(h_conv)

    if filter_size_2 is None:
      filter_size_2 = filter_size

    # 1x1 with half size
    h_conv = two_dim_convnet.make_conv_layer(
        h_conv,
        in_channels=in_channels,
        out_channels=in_channels / divide_channels_by,
        layer_name=layer_name + '_1x1h',
        filter_size=1,
        filter_size_2=1,
        non_linearity=True,
        batch_norm=batch_norm,
        is_training=is_training,
        data_format=data_format,
        stddev=stddev)

    # 3x3 with half size
    if channel_multiplier == 0:
      h_conv = two_dim_convnet.make_conv_layer(
          h_conv,
          in_channels=in_channels / divide_channels_by,
          out_channels=in_channels / divide_channels_by,
          layer_name=layer_name + '_%dx%dh' % (filter_size, filter_size_2),
          filter_size=filter_size,
          filter_size_2=filter_size_2,
          non_linearity=True,
          batch_norm=batch_norm,
          is_training=is_training,
          atrou_rate=atrou_rate,
          data_format=data_format,
          stddev=stddev)
    else:
      # We use separable convolution for 3x3
      h_conv = two_dim_convnet.make_conv_sep2d_layer(
          h_conv,
          in_channels=in_channels / divide_channels_by,
          channel_multiplier=channel_multiplier,
          out_channels=in_channels / divide_channels_by,
          layer_name=layer_name + '_sep%dx%dh' % (filter_size, filter_size_2),
          filter_size=filter_size,
          filter_size_2=filter_size_2,
          batch_norm=batch_norm,
          is_training=is_training,
          atrou_rate=atrou_rate,
          data_format=data_format,
          stddev=stddev)

    # 1x1 back to normal size without relu
    h_conv = two_dim_convnet.make_conv_layer(
        h_conv,
        in_channels=in_channels / divide_channels_by,
        out_channels=out_channels,
        layer_name=layer_name + '_1x1',
        filter_size=1,
        filter_size_2=1,
        non_linearity=False,
        batch_norm=False,
        is_training=is_training,
        data_format=data_format,
        stddev=stddev)

    if dropout_keep_prob < 1.0:
      logging.info('dropout keep prob %f', dropout_keep_prob)
      h_conv = tf.nn.dropout(h_conv, keep_prob=dropout_keep_prob)

    return h_conv + input_times_almost_1


def make_two_dim_resnet(
    input_node,
    num_residues=50,
    num_features=40,
    num_predictions=1,
    num_channels=32,
    num_layers=2,
    filter_size=3,
    filter_size_2=None,
    final_non_linearity=False,
    name_prefix='',
    fancy=True,
    batch_norm=False,
    is_training=False,
    atrou_rates=None,
    channel_multiplier=0,
    divide_channels_by=2,
    resize_features_with_1x1=False,
    data_format='NHWC',
    stddev=0.01,
    dropout_keep_prob=1.0):
  """Two dim resnet towers."""
  del num_residues  # Unused.

  if atrou_rates is None:
    atrou_rates = [1]
  if not fancy:
    raise ValueError('non fancy deprecated')

  logging.info('atrou rates %s', atrou_rates)

  logging.info('name prefix %s', name_prefix)
  x_image = input_node
  previous_layer = x_image
  non_linearity = True
  for i_layer in range(num_layers):
    in_channels = num_channels
    out_channels = num_channels

    curr_atrou_rate = atrou_rates[i_layer % len(atrou_rates)]

    if i_layer == 0:
      in_channels = num_features
    if i_layer == num_layers - 1:
      out_channels = num_predictions
      non_linearity = final_non_linearity
    if i_layer == 0 or i_layer == num_layers - 1:
      layer_name = name_prefix + 'conv%d' % (i_layer + 1)
      initial_filter_size = filter_size
      if resize_features_with_1x1:
        initial_filter_size = 1
      previous_layer = two_dim_convnet.make_conv_layer(
          input_node=previous_layer,
          in_channels=in_channels,
          out_channels=out_channels,
          layer_name=layer_name,
          filter_size=initial_filter_size,
          filter_size_2=filter_size_2,
          non_linearity=non_linearity,
          atrou_rate=curr_atrou_rate,
          data_format=data_format,
          stddev=stddev)
    else:
      layer_name = name_prefix + 'res%d' % (i_layer + 1)
      previous_layer = make_sep_res_layer(
          input_node=previous_layer,
          in_channels=in_channels,
          out_channels=out_channels,
          layer_name=layer_name,
          filter_size=filter_size,
          filter_size_2=filter_size_2,
          batch_norm=batch_norm,
          is_training=is_training,
          atrou_rate=curr_atrou_rate,
          channel_multiplier=channel_multiplier,
          divide_channels_by=divide_channels_by,
          data_format=data_format,
          stddev=stddev,
          dropout_keep_prob=dropout_keep_prob)

  y = previous_layer

  return y
