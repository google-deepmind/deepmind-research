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
"""ResNet (post-activation) with FixUp."""
# pylint: disable=invalid-name

import functools
import haiku as hk
import jax
import jax.numpy as jnp
from nfnets import base


nonlinearities = {
    'swish': jax.nn.silu,
    'relu': jax.nn.relu,
    'identity': lambda x: x}


class FixUp_ResNet(hk.Module):
  """Fixup based ResNet."""

  variant_dict = {'ResNet50': {'depth': [3, 4, 6, 3]},
                  'ResNet101': {'depth': [3, 4, 23, 3]},
                  'ResNet152': {'depth': [3, 8, 36, 3]},
                  'ResNet200': {'depth': [3, 24, 36, 3]},
                  'ResNet288': {'depth': [24, 24, 24, 24]},
                  'ResNet600': {'depth': [50, 50, 50, 50]},
                  }

  def __init__(self, num_classes, variant='ResNet50', width=4,
               stochdepth_rate=0.1, drop_rate=None,
               activation='relu', fc_init=jnp.zeros,
               name='FixUp_ResNet'):
    super().__init__(name=name)
    self.num_classes = num_classes
    self.variant = variant
    self.width = width
    # Get variant info
    block_params = self.variant_dict[self.variant]
    self.width_pattern = [item * self.width for item in [64, 128, 256, 512]]
    self.depth_pattern = block_params['depth']
    self.activation = nonlinearities[activation]
    if drop_rate is None:
      self.drop_rate = block_params['drop_rate']
    else:
      self.drop_rate = drop_rate
    self.which_conv = functools.partial(hk.Conv2D,
                                        with_bias=False)
    # Stem
    ch = int(16 * self.width)
    self.initial_conv = self.which_conv(ch, kernel_shape=7, stride=2,
                                        padding='SAME',
                                        name='initial_conv')

    # Body
    self.blocks = []
    num_blocks = sum(self.depth_pattern)
    index = 0  # Overall block index
    block_args = (self.width_pattern, self.depth_pattern, [1, 2, 2, 2])
    for block_width, stage_depth, stride in zip(*block_args):
      for block_index in range(stage_depth):
        # Block stochastic depth drop-rate
        block_stochdepth_rate = stochdepth_rate * index / num_blocks
        self.blocks += [ResBlock(ch, block_width, num_blocks,
                                 stride=stride if block_index == 0 else 1,
                                 activation=self.activation,
                                 which_conv=self.which_conv,
                                 stochdepth_rate=block_stochdepth_rate,
                                 )]
        ch = block_width
        index += 1

    # Head
    self.fc = hk.Linear(self.num_classes, w_init=fc_init, with_bias=True)

  def __call__(self, x, is_training=True, return_metrics=False):
    """Return the output of the final layer without any [log-]softmax."""
    # Stem
    outputs = {}
    out = self.initial_conv(x)
    bias1 = hk.get_parameter('bias1', (), x.dtype, init=jnp.zeros)
    out = self.activation(out + bias1)
    out = hk.max_pool(out, window_shape=(1, 3, 3, 1),
                      strides=(1, 2, 2, 1), padding='SAME')
    if return_metrics:
      outputs.update(base.signal_metrics(out, 0))
    # Blocks
    for i, block in enumerate(self.blocks):
      out, res_avg_var = block(out, is_training=is_training)
      if return_metrics:
        outputs.update(base.signal_metrics(out, i + 1))
        outputs[f'res_avg_var_{i}'] = res_avg_var
    # Final-conv->activation, pool, dropout, classify
    pool = jnp.mean(out, [1, 2])
    outputs['pool'] = pool
    # Optionally apply dropout
    if self.drop_rate > 0.0 and is_training:
      pool = hk.dropout(hk.next_rng_key(), self.drop_rate, pool)
    bias2 = hk.get_parameter('bias2', (), pool.dtype, init=jnp.zeros)
    outputs['logits'] = self.fc(pool + bias2)
    return outputs

  def count_flops(self, h, w):
    flops = []
    flops += [base.count_conv_flops(3, self.initial_conv, h, w)]
    h, w = h / 2, w / 2
    # Body FLOPs
    for block in self.blocks:
      flops += [block.count_flops(h, w)]
      if block.stride > 1:
        h, w = h / block.stride, w / block.stride
    # Head module FLOPs
    out_ch = self.blocks[-1].out_ch
    flops += [base.count_conv_flops(out_ch, self.final_conv, h, w)]
    # Count flops for classifier
    flops += [self.final_conv.output_channels * self.fc.output_size]
    return flops, sum(flops)


class ResBlock(hk.Module):
  """Post-activation Fixup Block."""

  def __init__(self, in_ch, out_ch, num_blocks, bottleneck_ratio=0.25,
               kernel_size=3, stride=1,
               which_conv=hk.Conv2D, activation=jax.nn.relu,
               stochdepth_rate=None, name=None):
    super().__init__(name=name)
    self.in_ch, self.out_ch = in_ch, out_ch
    self.kernel_size = kernel_size
    self.activation = activation
    # Bottleneck width
    self.width = int(self.out_ch * bottleneck_ratio)
    self.stride = stride
    # Conv 0 (typically expansion conv)
    conv0_init = hk.initializers.RandomNormal(
        stddev=((2 / self.width)**0.5) * (num_blocks**(-0.25)))
    self.conv0 = which_conv(self.width, kernel_shape=1, padding='SAME',
                            name='conv0', w_init=conv0_init)
    # Grouped NxN conv
    conv1_init = hk.initializers.RandomNormal(
        stddev=((2 / (self.width * (kernel_size**2)))**0.5)
        * (num_blocks**(-0.25)))
    self.conv1 = which_conv(self.width, kernel_shape=kernel_size, stride=stride,
                            padding='SAME', name='conv1', w_init=conv1_init)
    # Conv 2, typically projection conv
    self.conv2 = which_conv(self.out_ch, kernel_shape=1, padding='SAME',
                            name='conv2', w_init=hk.initializers.Constant(0))
    # Use shortcut conv on channel change or downsample.
    self.use_projection = stride > 1 or self.in_ch != self.out_ch
    if self.use_projection:
      shortcut_init = hk.initializers.RandomNormal(
          stddev=(2 / self.out_ch) ** 0.5)
      self.conv_shortcut = which_conv(self.out_ch, kernel_shape=1,
                                      stride=stride, padding='SAME',
                                      name='conv_shortcut',
                                      w_init=shortcut_init)
    # Are we using stochastic depth?
    self._has_stochdepth = (stochdepth_rate is not None and
                            stochdepth_rate > 0. and stochdepth_rate < 1.0)
    if self._has_stochdepth:
      self.stoch_depth = base.StochDepth(stochdepth_rate)

  def __call__(self, x, is_training):
    bias1a = hk.get_parameter('bias1a', (), x.dtype, init=jnp.zeros)
    bias1b = hk.get_parameter('bias1b', (), x.dtype, init=jnp.zeros)
    bias2a = hk.get_parameter('bias2a', (), x.dtype, init=jnp.zeros)
    bias2b = hk.get_parameter('bias2b', (), x.dtype, init=jnp.zeros)
    bias3a = hk.get_parameter('bias3a', (), x.dtype, init=jnp.zeros)
    bias3b = hk.get_parameter('bias3b', (), x.dtype, init=jnp.zeros)
    scale = hk.get_parameter('scale', (), x.dtype, init=jnp.ones)

    out = x + bias1a
    shortcut = out
    if self.use_projection:  # Downsample with conv1x1
      shortcut = self.conv_shortcut(shortcut)
    out = self.conv0(out)
    out = self.activation(out + bias1b)
    out = self.conv1(out + bias2a)
    out = self.activation(out + bias2b)
    out = self.conv2(out + bias3a)
    out = out * scale + bias3b
    # Get average residual variance for reporting metrics.
    res_avg_var = jnp.mean(jnp.var(out, axis=[0, 1, 2]))
    # Apply stochdepth if applicable.
    if self._has_stochdepth:
      out = self.stoch_depth(out, is_training)
    # SkipInit Gain
    out = out + shortcut
    return self.activation(out), res_avg_var

  def count_flops(self, h, w):
    # Count conv FLOPs based on input HW
    expand_flops = base.count_conv_flops(self.in_ch, self.conv0, h, w)
    # If block is strided we decrease resolution here.
    dw_flops = base.count_conv_flops(self.width, self.conv1, h, w)
    if self.stride > 1:
      h, w = h / self.stride, w / self.stride
    if self.use_projection:
      sc_flops = base.count_conv_flops(self.in_ch, self.conv_shortcut, h, w)
    else:
      sc_flops = 0
    contract_flops = base.count_conv_flops(self.width, self.conv2, h, w)
    return sum([expand_flops, dw_flops, contract_flops, sc_flops])

