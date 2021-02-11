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
"""Norm-Free Nets."""
# pylint: disable=unused-import
# pylint: disable=invalid-name

import functools
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


from nfnets import base


class NFNet(hk.Module):
  """Normalizer-Free Networks with an improved architecture.

  References:
    [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
    Recognition Without Normalization.
  """

  variant_dict = base.nfnet_params

  def __init__(self, num_classes, variant='F0',
               width=1.0, se_ratio=0.5,
               alpha=0.2, stochdepth_rate=0.1, drop_rate=None,
               activation='gelu', fc_init=None,
               final_conv_mult=2, final_conv_ch=None,
               use_two_convs=True,
               name='NFNet'):
    super().__init__(name=name)
    self.num_classes = num_classes
    self.variant = variant
    self.width = width
    self.se_ratio = se_ratio
    # Get variant info
    block_params = self.variant_dict[self.variant]
    self.train_imsize = block_params['train_imsize']
    self.test_imsize = block_params['test_imsize']
    self.width_pattern = block_params['width']
    self.depth_pattern = block_params['depth']
    self.bneck_pattern = block_params.get('expansion', [0.5] * 4)
    self.group_pattern = block_params.get('group_width', [128] * 4)
    self.big_pattern = block_params.get('big_width', [True] * 4)
    self.activation = base.nonlinearities[activation]
    if drop_rate is None:
      self.drop_rate = block_params['drop_rate']
    else:
      self.drop_rate = drop_rate
    self.which_conv = base.WSConv2D
    # Stem
    ch = self.width_pattern[0] // 2
    self.stem = hk.Sequential([
        self.which_conv(16, kernel_shape=3, stride=2,
                        padding='SAME', name='stem_conv0'),
        self.activation,
        self.which_conv(32, kernel_shape=3, stride=1,
                        padding='SAME', name='stem_conv1'),
        self.activation,
        self.which_conv(64, kernel_shape=3, stride=1,
                        padding='SAME', name='stem_conv2'),
        self.activation,
        self.which_conv(ch, kernel_shape=3, stride=2,
                        padding='SAME', name='stem_conv3'),
    ])

    # Body
    self.blocks = []
    expected_std = 1.0
    num_blocks = sum(self.depth_pattern)
    index = 0  # Overall block index
    stride_pattern = [1, 2, 2, 2]
    block_args = zip(self.width_pattern, self.depth_pattern, self.bneck_pattern,
                     self.group_pattern, self.big_pattern, stride_pattern)
    for (block_width, stage_depth, expand_ratio,
         group_size, big_width, stride) in block_args:
      for block_index in range(stage_depth):
        # Scalar pre-multiplier so each block sees an N(0,1) input at init
        beta = 1./ expected_std
        # Block stochastic depth drop-rate
        block_stochdepth_rate = stochdepth_rate * index / num_blocks
        out_ch = (int(block_width * self.width))
        self.blocks += [NFBlock(ch, out_ch,
                                expansion=expand_ratio, se_ratio=se_ratio,
                                group_size=group_size,
                                stride=stride if block_index == 0 else 1,
                                beta=beta, alpha=alpha,
                                activation=self.activation,
                                which_conv=self.which_conv,
                                stochdepth_rate=block_stochdepth_rate,
                                big_width=big_width,
                                use_two_convs=use_two_convs,
                                )]
        ch = out_ch
        index += 1
         # Reset expected std but still give it 1 block of growth
        if block_index == 0:
          expected_std = 1.0
        expected_std = (expected_std **2 + alpha**2)**0.5

    # Head
    if final_conv_mult is None:
      if final_conv_ch is None:
        raise ValueError('Must provide one of final_conv_mult or final_conv_ch')
      ch = final_conv_ch
    else:
      ch = int(final_conv_mult * ch)
    self.final_conv = self.which_conv(ch, kernel_shape=1,
                                      padding='SAME', name='final_conv')
    # By default, initialize with N(0, 0.01)
    if fc_init is None:
      fc_init = hk.initializers.RandomNormal(mean=0, stddev=0.01)
    self.fc = hk.Linear(self.num_classes, w_init=fc_init, with_bias=True)

  def __call__(self, x, is_training=True, return_metrics=False):
    """Return the output of the final layer without any [log-]softmax."""
    # Stem
    outputs = {}
    out = self.stem(x)
    if return_metrics:
      outputs.update(base.signal_metrics(out, 0))
    # Blocks
    for i, block in enumerate(self.blocks):
      out, res_avg_var = block(out, is_training=is_training)
      if return_metrics:
        outputs.update(base.signal_metrics(out, i + 1))
        outputs[f'res_avg_var_{i}'] = res_avg_var
    # Final-conv->activation, pool, dropout, classify
    out = self.activation(self.final_conv(out))
    pool = jnp.mean(out, [1, 2])
    outputs['pool'] = pool
    # Optionally apply dropout
    if self.drop_rate > 0.0 and is_training:
      pool = hk.dropout(hk.next_rng_key(), self.drop_rate, pool)
    outputs['logits'] = self.fc(pool)
    return outputs

  def count_flops(self, h, w):
    flops = []
    ch = 3
    for module in self.stem.layers:
      if isinstance(module, hk.Conv2D):
        flops += [base.count_conv_flops(ch, module, h, w)]
        if any([item > 1 for item in module.stride]):
          h, w = h / module.stride[0], w / module.stride[1]
        ch = module.output_channels
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


class NFBlock(hk.Module):
  """Normalizer-Free Net Block."""

  def __init__(self, in_ch, out_ch, expansion=0.5, se_ratio=0.5,
               kernel_shape=3, group_size=128, stride=1,
               beta=1.0, alpha=0.2,
               which_conv=base.WSConv2D, activation=jax.nn.gelu,
               big_width=True, use_two_convs=True,
               stochdepth_rate=None, name=None):
    super().__init__(name=name)
    self.in_ch, self.out_ch = in_ch, out_ch
    self.expansion = expansion
    self.se_ratio = se_ratio
    self.kernel_shape = kernel_shape
    self.activation = activation
    self.beta, self.alpha = beta, alpha
    # Mimic resnet style bigwidth scaling?
    width = int((self.out_ch if big_width else self.in_ch) * expansion)
    # Round expanded with based on group count
    self.groups = width // group_size
    self.width = group_size * self.groups
    self.stride = stride
    self.use_two_convs = use_two_convs
    # Conv 0 (typically expansion conv)
    self.conv0 = which_conv(self.width, kernel_shape=1, padding='SAME',
                            name='conv0')
    # Grouped NxN conv
    self.conv1 = which_conv(self.width, kernel_shape=kernel_shape,
                            stride=stride, padding='SAME',
                            feature_group_count=self.groups, name='conv1')
    if self.use_two_convs:
      self.conv1b = which_conv(self.width, kernel_shape=kernel_shape,
                               stride=1, padding='SAME',
                               feature_group_count=self.groups, name='conv1b')
    # Conv 2, typically projection conv
    self.conv2 = which_conv(self.out_ch, kernel_shape=1, padding='SAME',
                            name='conv2')
    # Use shortcut conv on channel change or downsample.
    self.use_projection = stride > 1 or self.in_ch != self.out_ch
    if self.use_projection:
      self.conv_shortcut = which_conv(self.out_ch, kernel_shape=1,
                                      padding='SAME', name='conv_shortcut')
    # Squeeze + Excite Module
    self.se = base.SqueezeExcite(self.out_ch, self.out_ch, self.se_ratio)

    # Are we using stochastic depth?
    self._has_stochdepth = (stochdepth_rate is not None and
                            stochdepth_rate > 0. and stochdepth_rate < 1.0)
    if self._has_stochdepth:
      self.stoch_depth = base.StochDepth(stochdepth_rate)

  def __call__(self, x, is_training):
    out = self.activation(x) * self.beta
    if self.stride > 1:  # Average-pool downsample.
      shortcut = hk.avg_pool(out, window_shape=(1, 2, 2, 1),
                             strides=(1, 2, 2, 1), padding='SAME')
      if self.use_projection:
        shortcut = self.conv_shortcut(shortcut)
    elif self.use_projection:
      shortcut = self.conv_shortcut(out)
    else:
      shortcut = x
    out = self.conv0(out)
    out = self.conv1(self.activation(out))
    if self.use_two_convs:
      out = self.conv1b(self.activation(out))
    out = self.conv2(self.activation(out))
    out = (self.se(out) * 2) * out  # Multiply by 2 for rescaling
    # Get average residual standard deviation for reporting metrics.
    res_avg_var = jnp.mean(jnp.var(out, axis=[0, 1, 2]))
    # Apply stochdepth if applicable.
    if self._has_stochdepth:
      out = self.stoch_depth(out, is_training)
    # SkipInit Gain
    out = out * hk.get_parameter('skip_gain', (), out.dtype, init=jnp.zeros)
    return out * self.alpha + shortcut, res_avg_var

  def count_flops(self, h, w):
    # Count conv FLOPs based on input HW
    expand_flops = base.count_conv_flops(self.in_ch, self.conv0, h, w)
    # If block is strided we decrease resolution here.
    dw_flops = base.count_conv_flops(self.width, self.conv1, h, w)
    if self.stride > 1:
      h, w = h / self.stride, w / self.stride
    if self.use_two_convs:
      dw_flops += base.count_conv_flops(self.width, self.conv1b, h, w)

    if self.use_projection:
      sc_flops = base.count_conv_flops(self.in_ch, self.conv_shortcut, h, w)
    else:
      sc_flops = 0
    # SE flops happen on avg-pooled activations
    se_flops = self.se.fc0.output_size * self.out_ch
    se_flops += self.se.fc0.output_size * self.se.fc1.output_size
    contract_flops = base.count_conv_flops(self.width, self.conv2, h, w)
    return sum([expand_flops, dw_flops, se_flops, contract_flops, sc_flops])

