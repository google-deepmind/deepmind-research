# Copyright 2019 DeepMind Technologies Limited and Google LLC
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
"""Network utilities."""

import functools
import re
import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan


def _sn_custom_getter():
  def name_filter(name):
    match = re.match(r'.*w(_.*)?$', name)
    return match is not None

  return tfgan.features.spectral_normalization_custom_getter(
      name_filter=name_filter)


class ConvGenNet(snt.AbstractModule):
  """As in the SN paper."""

  def __init__(self, name='conv_gen'):
    super(ConvGenNet, self).__init__(name=name)

  def _build(self, inputs, is_training):
    batch_size = inputs.get_shape().as_list()[0]
    first_shape = [4, 4, 512]
    norm_ctor = snt.BatchNormV2
    norm_ctor_config = {'scale': True}
    up_tensor = snt.Linear(np.prod(first_shape))(inputs)
    first_tensor = tf.reshape(up_tensor, shape=[batch_size] + first_shape)

    net = snt.nets.ConvNet2DTranspose(
        output_channels=[256, 128, 64, 3],
        output_shapes=[(8, 8), (16, 16), (32, 32), (32, 32)],
        kernel_shapes=[(4, 4), (4, 4), (4, 4), (3, 3)],
        strides=[2, 2, 2, 1],
        normalization_ctor=norm_ctor,
        normalization_kwargs=norm_ctor_config,
        normalize_final=False,
        paddings=[snt.SAME], activate_final=False, activation=tf.nn.relu)
    output = net(first_tensor, is_training=is_training)
    return tf.nn.tanh(output)


class ConvMetricNet(snt.AbstractModule):
  """Convolutional discriminator (metric) architecture."""

  def __init__(self, num_outputs=2, use_sn=True, name='sn_metric'):
    super(ConvMetricNet, self).__init__(name=name)
    self._num_outputs = num_outputs
    self._use_sn = use_sn

  def _build(self, inputs):

    def build_net():
      net = snt.nets.ConvNet2D(
          output_channels=[64, 64, 128, 128, 256, 256, 512],
          kernel_shapes=[
              (3, 3), (4, 4), (3, 3), (4, 4), (3, 3), (4, 4), (3, 3)],
          strides=[1, 2, 1, 2, 1, 2, 1],
          paddings=[snt.SAME], activate_final=True,
          activation=functools.partial(tf.nn.leaky_relu, alpha=0.1))
      linear = snt.Linear(self._num_outputs)
      output = linear(snt.BatchFlatten()(net(inputs)))
      return output
    if self._use_sn:
      with tf.variable_scope('', custom_getter=_sn_custom_getter()):
        output = build_net()
    else:
      output = build_net()

    return output


class MLPGeneratorNet(snt.AbstractModule):
  """MNIST generator net."""

  def __init__(self, name='mlp_generator'):
    super(MLPGeneratorNet, self).__init__(name=name)

  def _build(self, inputs, is_training=True):
    del is_training
    net = snt.nets.MLP([500, 500, 784], activation=tf.nn.leaky_relu)
    out = net(inputs)
    out = tf.nn.tanh(out)
    return snt.BatchReshape([28, 28, 1])(out)


class MLPMetricNet(snt.AbstractModule):
  """Same as in Grover and Ermon, ICLR workshop 2017."""

  def __init__(self, num_outputs=2, name='mlp_metric'):
    super(MLPMetricNet, self).__init__(name=name)
    self._layer_size = [500, 500, num_outputs]

  def _build(self, inputs):
    net = snt.nets.MLP(self._layer_size,
                       activation=tf.nn.leaky_relu)
    output = net(snt.BatchFlatten()(inputs))
    return output
