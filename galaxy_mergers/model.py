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

"""Fork of a generic ResNet to incorporate additional cosmological features."""

from typing import Mapping, Optional, Sequence, Text

import sonnet.v2 as snt
import tensorflow.compat.v2 as tf


class ResNet(snt.Module):
  """ResNet model."""

  def __init__(self,
               n_repeats: int,
               blocks_per_group_list: Sequence[int],
               num_classes: int,
               bn_config: Optional[Mapping[Text, float]] = None,
               resnet_v2: bool = False,
               channels_per_group_list: Sequence[int] = (256, 512, 1024, 2048),
               use_additional_features: bool = False,
               additional_features_mode: Optional[Text] = "per_block",
               name: Optional[Text] = None):
    """Constructs a ResNet model.

    Args:
      n_repeats: The batch dimension for the input is expected to have the form
        `B = b * n_repeats`. After the conv stack, the logits for the
        `n_repeats` replicas are reduced, leading to an output batch dimension
        of `b`.
      blocks_per_group_list: A sequence of length 4 that indicates the number of
        blocks created in each group.
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers. By default the `decay_rate` is
        `0.9` and `eps` is `1e-5`.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        False.
      channels_per_group_list: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      use_additional_features: If true, additional vector features will be
        concatenated to the residual stack before logits are computed.
      additional_features_mode: Mode for processing additional features.
        Supported modes: 'mlp' and 'per_block'.
      name: Name of the module.
    """
    super(ResNet, self).__init__(name=name)
    self._n_repeats = n_repeats
    if bn_config is None:
      bn_config = {"decay_rate": 0.9, "eps": 1e-5}
    self._bn_config = bn_config
    self._resnet_v2 = resnet_v2

    # Number of blocks in each group for ResNet.
    if len(blocks_per_group_list) != 4:
      raise ValueError(
          "`blocks_per_group_list` must be of length 4 not {}".format(
              len(blocks_per_group_list)))
    self._blocks_per_group_list = blocks_per_group_list

    # Number of channels in each group for ResNet.
    if len(channels_per_group_list) != 4:
      raise ValueError(
          "`channels_per_group_list` must be of length 4 not {}".format(
              len(channels_per_group_list)))
    self._channels_per_group_list = channels_per_group_list
    self._use_additional_features = use_additional_features
    self._additional_features_mode = additional_features_mode

    self._initial_conv = snt.Conv2D(
        output_channels=64,
        kernel_shape=7,
        stride=2,
        with_bias=False,
        padding="SAME",
        name="initial_conv")
    if not self._resnet_v2:
      self._initial_batchnorm = snt.BatchNorm(
          create_scale=True,
          create_offset=True,
          name="initial_batchnorm",
          **bn_config)

    self._block_groups = []
    strides = [1, 2, 2, 2]
    for i in range(4):
      self._block_groups.append(
          snt.nets.resnet.BlockGroup(
              channels=self._channels_per_group_list[i],
              num_blocks=self._blocks_per_group_list[i],
              stride=strides[i],
              bn_config=bn_config,
              resnet_v2=resnet_v2,
              name="block_group_%d" % (i)))

    if self._resnet_v2:
      self._final_batchnorm = snt.BatchNorm(
          create_scale=True,
          create_offset=True,
          name="final_batchnorm",
          **bn_config)

    self._logits = snt.Linear(
        output_size=num_classes,
        w_init=snt.initializers.VarianceScaling(scale=2.0), name="logits")

    if self._use_additional_features:
      self._embedding = LinearBNReLU(output_size=16, name="embedding",
                                     **bn_config)

      if self._additional_features_mode == "mlp":
        self._feature_repr = LinearBNReLU(
            output_size=self._channels_per_group_list[-1], name="features_repr",
            **bn_config)
      elif self._additional_features_mode == "per_block":
        self._feature_repr = []
        for i, ch in enumerate(self._channels_per_group_list):
          self._feature_repr.append(
              LinearBNReLU(output_size=ch, name=f"features_{i}", **bn_config))
      else:
        raise ValueError(f"Unsupported addiitonal features mode: "
                         f"{additional_features_mode}")

  def __call__(self, inputs, features, is_training):
    net = inputs
    net = self._initial_conv(net)
    if not self._resnet_v2:
      net = self._initial_batchnorm(net, is_training=is_training)
      net = tf.nn.relu(net)

    net = tf.nn.max_pool2d(
        net, ksize=3, strides=2, padding="SAME", name="initial_max_pool")

    if self._use_additional_features:
      assert features is not None
      features = self._embedding(features, is_training=is_training)

    for i, block_group in enumerate(self._block_groups):
      net = block_group(net, is_training)

      if (self._use_additional_features and
          self._additional_features_mode == "per_block"):
        features_i = self._feature_repr[i](features, is_training=is_training)
        # support for n_repeats > 1
        features_i = tf.repeat(features_i, self._n_repeats, axis=0)
        net += features_i[:, None, None, :]  # expand to spacial resolution

    if self._resnet_v2:
      net = self._final_batchnorm(net, is_training=is_training)
      net = tf.nn.relu(net)
    net = tf.reduce_mean(net, axis=[1, 2], name="final_avg_pool")
    # Re-split the batch dimension
    net = tf.reshape(net, [-1, self._n_repeats] + net.shape.as_list()[1:])
    # Average over the various repeats of the input (e.g. those could have
    # corresponded to different crops).
    net = tf.reduce_mean(net, axis=1)

    if (self._use_additional_features and
        self._additional_features_mode == "mlp"):
      net += self._feature_repr(features, is_training=is_training)

    return self._logits(net)


class LinearBNReLU(snt.Module):
  """Wrapper class for Linear layer with Batch Norm and ReLU activation."""

  def __init__(self, output_size=64,
               w_init=snt.initializers.VarianceScaling(scale=2.0),
               name="linear", **bn_config):
    """Constructs a LinearBNReLU module.

    Args:
      output_size: Output dimension.
      w_init: weight Initializer for snt.Linear.
      name: Name of the module.
      **bn_config: Optional parameters to be passed to snt.BatchNorm.
    """
    super(LinearBNReLU, self).__init__(name=name)
    self._linear = snt.Linear(output_size=output_size, w_init=w_init,
                              name=f"{name}_linear")
    self._bn = snt.BatchNorm(create_scale=True, create_offset=True,
                             name=f"{name}_bn", **bn_config)

  def __call__(self, x, is_training):
    x = self._linear(x)
    x = self._bn(x, is_training=is_training)
    return tf.nn.relu(x)
