# Copyright 2020 DeepMind Technologies Limited.
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

"""Networks used in BYOL."""

from typing import Any, Mapping, Optional, Sequence, Text

import haiku as hk
import jax
import jax.numpy as jnp


class MLP(hk.Module):
  """One hidden layer perceptron, with normalization."""

  def __init__(
      self,
      name: Text,
      hidden_size: int,
      output_size: int,
      bn_config: Mapping[Text, Any],
  ):
    super().__init__(name=name)
    self._hidden_size = hidden_size
    self._output_size = output_size
    self._bn_config = bn_config

  def __call__(self, inputs: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    out = hk.Linear(output_size=self._hidden_size, with_bias=True)(inputs)
    out = hk.BatchNorm(**self._bn_config)(out, is_training=is_training)
    out = jax.nn.relu(out)
    out = hk.Linear(output_size=self._output_size, with_bias=False)(out)
    return out


def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f'`{name}` must be of length 4 not {len(value)}')


class ResNetTorso(hk.Module):
  """ResNet model."""

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      num_classes: int = None,
      bn_config: Optional[Mapping[str, float]] = None,
      resnet_v2: bool = False,
      bottleneck: bool = True,
      channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
      use_projection: Sequence[bool] = (True, True, True, True),
      width_multiplier: int = 1,
      name: Optional[str] = None,
  ):
    """Constructs a ResNet model.

    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of three elements, `decay_rate`, `eps`, and
        `cross_replica_axis`, to be passed on to the `BatchNorm` layers. By
        default the `decay_rate` is `0.9` and `eps` is `1e-5`, and the axis is
        `None`.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        False.
       bottleneck: Whether the block should bottleneck or not. Defaults to True.
      channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      width_multiplier: An integer multiplying the number of channels per group.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.resnet_v2 = resnet_v2

    bn_config = dict(bn_config or {})
    bn_config.setdefault('decay_rate', 0.9)
    bn_config.setdefault('eps', 1e-5)
    bn_config.setdefault('create_scale', True)
    bn_config.setdefault('create_offset', True)

    # Number of blocks in each group for ResNet.
    check_length(4, blocks_per_group, 'blocks_per_group')
    check_length(4, channels_per_group, 'channels_per_group')

    self.initial_conv = hk.Conv2D(
        output_channels=64 * width_multiplier,
        kernel_shape=7,
        stride=2,
        with_bias=False,
        padding='SAME',
        name='initial_conv')

    if not self.resnet_v2:
      self.initial_batchnorm = hk.BatchNorm(name='initial_batchnorm',
                                            **bn_config)

    self.block_groups = []
    strides = (1, 2, 2, 2)
    for i in range(4):
      self.block_groups.append(
          hk.nets.ResNet.BlockGroup(
              channels=width_multiplier * channels_per_group[i],
              num_blocks=blocks_per_group[i],
              stride=strides[i],
              bn_config=bn_config,
              resnet_v2=resnet_v2,
              bottleneck=bottleneck,
              use_projection=use_projection[i],
              name='block_group_%d' % (i)))

    if self.resnet_v2:
      self.final_batchnorm = hk.BatchNorm(name='final_batchnorm', **bn_config)

    self.logits = hk.Linear(num_classes, w_init=jnp.zeros, name='logits')

  def __call__(self, inputs, is_training, test_local_stats=False):
    out = inputs
    out = self.initial_conv(out)
    if not self.resnet_v2:
      out = self.initial_batchnorm(out, is_training, test_local_stats)
      out = jax.nn.relu(out)

    out = hk.max_pool(out,
                      window_shape=(1, 3, 3, 1),
                      strides=(1, 2, 2, 1),
                      padding='SAME')

    for block_group in self.block_groups:
      out = block_group(out, is_training, test_local_stats)

    if self.resnet_v2:
      out = self.final_batchnorm(out, is_training, test_local_stats)
      out = jax.nn.relu(out)
    out = jnp.mean(out, axis=[1, 2])
    return out


class TinyResNet(ResNetTorso):
  """Tiny resnet for local runs and tests."""

  def __init__(self,
               num_classes: Optional[int] = None,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               width_multiplier: int = 1,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to False.
      width_multiplier: An integer multiplying the number of channels per group.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(1, 1, 1, 1),
                     channels_per_group=(8, 8, 8, 8),
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=False,
                     width_multiplier=width_multiplier,
                     name=name)


class ResNet18(ResNetTorso):
  """ResNet18."""

  def __init__(self,
               num_classes: Optional[int] = None,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               width_multiplier: int = 1,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to False.
      width_multiplier: An integer multiplying the number of channels per group.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(2, 2, 2, 2),
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=False,
                     channels_per_group=(64, 128, 256, 512),
                     width_multiplier=width_multiplier,
                     name=name)


class ResNet34(ResNetTorso):
  """ResNet34."""

  def __init__(self,
               num_classes: Optional[int],
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               width_multiplier: int = 1,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to False.
      width_multiplier: An integer multiplying the number of channels per group.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 4, 6, 3),
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=False,
                     channels_per_group=(64, 128, 256, 512),
                     width_multiplier=width_multiplier,
                     name=name)


class ResNet50(ResNetTorso):
  """ResNet50."""

  def __init__(self,
               num_classes: Optional[int] = None,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               width_multiplier: int = 1,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to False.
      width_multiplier: An integer multiplying the number of channels per group.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 4, 6, 3),
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=True,
                     width_multiplier=width_multiplier,
                     name=name)


class ResNet101(ResNetTorso):
  """ResNet101."""

  def __init__(self,
               num_classes: Optional[int],
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               width_multiplier: int = 1,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to False.
      width_multiplier: An integer multiplying the number of channels per group.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 4, 23, 3),
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=True,
                     width_multiplier=width_multiplier,
                     name=name)


class ResNet152(ResNetTorso):
  """ResNet152."""

  def __init__(self,
               num_classes: Optional[int],
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               width_multiplier: int = 1,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to False.
      width_multiplier: An integer multiplying the number of channels per group.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 8, 36, 3),
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=True,
                     width_multiplier=width_multiplier,
                     name=name)


class ResNet200(ResNetTorso):
  """ResNet200."""

  def __init__(self,
               num_classes: Optional[int],
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               width_multiplier: int = 1,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to False.
      width_multiplier: An integer multiplying the number of channels per group.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 24, 36, 3),
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=True,
                     width_multiplier=width_multiplier,
                     name=name)
