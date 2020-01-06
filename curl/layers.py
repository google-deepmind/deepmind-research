################################################################################
# Copyright 2019 DeepMind Technologies Limited
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
################################################################################
"""Custom layers for CURL."""

from absl import logging
import sonnet as snt
import tensorflow.compat.v1 as tf

tfc = tf.compat.v1


class ResidualStack(snt.AbstractModule):
  """A stack of ResNet V2 blocks."""

  def __init__(self,
               num_hiddens,
               num_residual_layers,
               num_residual_hiddens,
               filter_size=3,
               initializers=None,
               data_format='NHWC',
               activation=tf.nn.relu,
               name='residual_stack'):
    """Instantiate a ResidualStack."""
    super(ResidualStack, self).__init__(name=name)
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens
    self._filter_size = filter_size
    self._initializers = initializers
    self._data_format = data_format
    self._activation = activation

  def _build(self, h):
    for i in range(self._num_residual_layers):
      h_i = self._activation(h)

      h_i = snt.Conv2D(
          output_channels=self._num_residual_hiddens,
          kernel_shape=(self._filter_size, self._filter_size),
          stride=(1, 1),
          initializers=self._initializers,
          data_format=self._data_format,
          name='res_nxn_%d' % i)(
              h_i)
      h_i = self._activation(h_i)

      h_i = snt.Conv2D(
          output_channels=self._num_hiddens,
          kernel_shape=(1, 1),
          stride=(1, 1),
          initializers=self._initializers,
          data_format=self._data_format,
          name='res_1x1_%d' % i)(
              h_i)
      h += h_i
    return self._activation(h)


class SharedConvModule(snt.AbstractModule):
  """Convolutional decoder."""

  def __init__(self,
               filters,
               kernel_size,
               activation,
               strides,
               name='shared_conv_encoder'):
    super(SharedConvModule, self).__init__(name=name)

    self._filters = filters
    self._kernel_size = kernel_size
    self._activation = activation
    self.strides = strides
    assert len(strides) == len(filters) - 1
    self.conv_shapes = None

  def _build(self, x, is_training=True):
    with tf.control_dependencies([tfc.assert_rank(x, 4)]):

      self.conv_shapes = [x.shape.as_list()]  # Needed by deconv module
      conv = x
    for i, (filter_i,
            stride_i) in enumerate(zip(self._filters, self.strides), 1):
      conv = tf.layers.Conv2D(
          filters=filter_i,
          kernel_size=self._kernel_size,
          padding='same',
          activation=self._activation,
          strides=stride_i,
          name='enc_conv_%d' % i)(
              conv)
      self.conv_shapes.append(conv.shape.as_list())
    conv_flat = snt.BatchFlatten()(conv)

    enc_mlp = snt.nets.MLP(
        name='enc_mlp',
        output_sizes=[self._filters[-1]],
        activation=self._activation,
        activate_final=True)
    h = enc_mlp(conv_flat)

    logging.info('Shared conv module layer shapes:')
    logging.info('\n'.join([str(el) for el in self.conv_shapes]))
    logging.info(h.shape.as_list())

    return h
