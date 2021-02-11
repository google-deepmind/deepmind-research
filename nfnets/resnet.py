# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""ResNet model family."""
import functools
import haiku as hk
import jax
import jax.numpy as jnp
from nfnets import base


class ResNet(hk.Module):
  """ResNetv2 Models."""

  variant_dict = {'ResNet50': {'depth': [3, 4, 6, 3]},
                  'ResNet101': {'depth': [3, 4, 23, 3]},
                  'ResNet152': {'depth': [3, 8, 36, 3]},
                  'ResNet200': {'depth': [3, 24, 36, 3]},
                  'ResNet288': {'depth': [24, 24, 24, 24]},
                  'ResNet600': {'depth': [50, 50, 50, 50]},
                  }

  def __init__(self, width, num_classes,
               variant='ResNet50',
               which_norm='BatchNorm', norm_kwargs=None,
               activation='relu', drop_rate=0.0,
               fc_init=jnp.zeros, conv_kwargs=None,
               preactivation=True, use_se=False, se_ratio=0.25,
               name='ResNet'):
    super().__init__(name=name)
    self.width = width
    self.num_classes = num_classes
    self.variant = variant
    self.depth_pattern = self.variant_dict[variant]['depth']
    self.activation = getattr(jax.nn, activation)
    self.drop_rate = drop_rate
    self.which_norm = getattr(hk, which_norm)
    if norm_kwargs is not None:
      self.which_norm = functools.partial(self.which_norm, **norm_kwargs)
    if conv_kwargs is not None:
      self.which_conv = functools.partial(hk.Conv2D, **conv_kwargs)
    else:
      self.which_conv = hk.Conv2D
    self.preactivation = preactivation

    # Stem
    self.initial_conv = self.which_conv(16 * self.width, kernel_shape=7,
                                        stride=2, padding='SAME',
                                        with_bias=False, name='initial_conv')
    if not self.preactivation:
      self.initial_bn = self.which_norm(name='initial_bn')
    which_block = ResBlockV2 if self.preactivation else ResBlockV1
    # Body
    self.blocks = []
    for multiplier, blocks_per_stage, stride in zip([64, 128, 256, 512],
                                                    self.depth_pattern,
                                                    [1, 2, 2, 2]):
      for block_index in range(blocks_per_stage):
        self.blocks += [which_block(multiplier * self.width,
                                    use_projection=block_index == 0,
                                    stride=stride if block_index == 0 else 1,
                                    activation=self.activation,
                                    which_norm=self.which_norm,
                                    which_conv=self.which_conv,
                                    use_se=use_se,
                                    se_ratio=se_ratio)]

    # Head
    self.final_bn = self.which_norm(name='final_bn')
    self.fc = hk.Linear(self.num_classes, w_init=fc_init, with_bias=True)

  def __call__(self, x, is_training, test_local_stats=False,
               return_metrics=False):
    """Return the output of the final layer without any [log-]softmax."""
    outputs = {}
    # Stem
    out = self.initial_conv(x)
    if not self.preactivation:
      out = self.activation(self.initial_bn(out, is_training, test_local_stats))
    out = hk.max_pool(out, window_shape=(1, 3, 3, 1),
                      strides=(1, 2, 2, 1), padding='SAME')
    if return_metrics:
      outputs.update(base.signal_metrics(out, 0))
    # Blocks
    for i, block in enumerate(self.blocks):
      out, res_var = block(out, is_training, test_local_stats)
      if return_metrics:
        outputs.update(base.signal_metrics(out, i + 1))
        outputs[f'res_avg_var_{i}'] = res_var
    if self.preactivation:
      out = self.activation(self.final_bn(out, is_training, test_local_stats))
    # Pool, dropout, classify
    pool = jnp.mean(out, axis=[1, 2])
    # Return pool before dropout in case we want to regularize it separately.
    outputs['pool'] = pool
    # Optionally apply dropout
    if self.drop_rate > 0.0 and is_training:
      pool = hk.dropout(hk.next_rng_key(), self.drop_rate, pool)
    outputs['logits'] = self.fc(pool)
    return outputs


class ResBlockV2(hk.Module):
  """ResNet preac block, 1x1->3x3->1x1 with strides and shortcut downsample."""

  def __init__(self, out_ch, stride=1, use_projection=False,
               activation=jax.nn.relu, which_norm=hk.BatchNorm,
               which_conv=hk.Conv2D, use_se=False, se_ratio=0.25,
               name=None):
    super().__init__(name=name)
    self.out_ch = out_ch
    self.stride = stride
    self.use_projection = use_projection
    self.activation = activation
    self.which_norm = which_norm
    self.which_conv = which_conv
    self.use_se = use_se
    self.se_ratio = se_ratio

    self.width = self.out_ch // 4

    self.bn0 = which_norm(name='bn0')
    self.conv0 = which_conv(self.width, kernel_shape=1, with_bias=False,
                            padding='SAME', name='conv0')
    self.bn1 = which_norm(name='bn1')
    self.conv1 = which_conv(self.width, stride=self.stride,
                            kernel_shape=3, with_bias=False,
                            padding='SAME', name='conv1')
    self.bn2 = which_norm(name='bn2')
    self.conv2 = which_conv(self.out_ch, kernel_shape=1, with_bias=False,
                            padding='SAME', name='conv2')
    if self.use_projection:
      self.conv_shortcut = which_conv(self.out_ch, stride=stride,
                                      kernel_shape=1, with_bias=False,
                                      padding='SAME', name='conv_shortcut')
    if self.use_se:
      self.se = base.SqueezeExcite(self.out_ch, self.out_ch, self.se_ratio)

  def __call__(self, x, is_training, test_local_stats):
    bn_args = (is_training, test_local_stats)
    out = self.activation(self.bn0(x, *bn_args))
    if self.use_projection:
      shortcut = self.conv_shortcut(out)
    else:
      shortcut = x
    out = self.conv0(out)
    out = self.conv1(self.activation(self.bn1(out, *bn_args)))
    out = self.conv2(self.activation(self.bn2(out, *bn_args)))
    if self.use_se:
      out = self.se(out) * out
    # Get average residual standard deviation for reporting metrics.
    res_avg_var = jnp.mean(jnp.var(out, axis=[0, 1, 2]))
    return out + shortcut, res_avg_var


class ResBlockV1(ResBlockV2):
  """Post-Ac Residual Block."""

  def __call__(self, x, is_training, test_local_stats):
    bn_args = (is_training, test_local_stats)
    if self.use_projection:
      shortcut = self.conv_shortcut(x)
      shortcut = self.which_norm(name='shortcut_bn')(shortcut, *bn_args)
    else:
      shortcut = x
    out = self.activation(self.bn0(self.conv0(x), *bn_args))
    out = self.activation(self.bn1(self.conv1(out), *bn_args))
    out = self.bn2(self.conv2(out), *bn_args)
    if self.use_se:
      out = self.se(out) * out
    res_avg_var = jnp.mean(jnp.var(out, axis=[0, 1, 2]))
    return self.activation(out + shortcut), res_avg_var
