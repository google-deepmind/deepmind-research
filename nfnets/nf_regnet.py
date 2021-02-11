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
"""Normalizer-Free RegNets."""
# pylint: disable=invalid-name

import haiku as hk
import jax
import jax.numpy as jnp
from nfnets import base


class NF_RegNet(hk.Module):
  """Normalizer-Free RegNets."""

  variant_dict = base.nf_regnet_params

  def __init__(self, num_classes, variant='B0',
               width=0.75, expansion=2.25, group_size=8, se_ratio=0.5,
               alpha=0.2, stochdepth_rate=0.1, drop_rate=None,
               activation='swish', fc_init=jnp.zeros,
               name='NF_RegNet'):
    super().__init__(name=name)
    self.num_classes = num_classes
    self.variant = variant
    self.width = width
    self.expansion = expansion
    self.group_size = group_size
    self.se_ratio = se_ratio
    # Get variant info
    block_params = self.variant_dict[self.variant]
    self.train_imsize = block_params['train_imsize']
    self.test_imsize = block_params['test_imsize']
    self.width_pattern = block_params['width']
    self.depth_pattern = block_params['depth']
    self.activation = base.nonlinearities[activation]
    if drop_rate is None:
      self.drop_rate = block_params['drop_rate']
    else:
      self.drop_rate = drop_rate
    self.which_conv = base.WSConv2D
    # Stem
    ch = int(self.width_pattern[0] * self.width)
    self.initial_conv = self.which_conv(ch, kernel_shape=3, stride=2,
                                        padding='SAME', name='initial_conv')

    # Body
    self.blocks = []
    expected_std = 1.0
    num_blocks = sum(self.depth_pattern)
    index = 0  # Overall block index
    for block_width, stage_depth in zip(self.width_pattern, self.depth_pattern):
      for block_index in range(stage_depth):
        # Scalar pre-multiplier so each block sees an N(0,1) input at init
        beta = 1./ expected_std
        # Block stochastic depth drop-rate
        block_stochdepth_rate = stochdepth_rate * index / num_blocks
        # Use a bottleneck expansion ratio of 1 for first block following EffNet
        expand_ratio = 1 if index == 0 else expansion
        out_ch = (int(block_width * self.width))
        self.blocks += [NFBlock(ch, out_ch,
                                expansion=expand_ratio, se_ratio=se_ratio,
                                group_size=self.group_size,
                                stride=2 if block_index == 0 else 1,
                                beta=beta, alpha=alpha,
                                activation=self.activation,
                                which_conv=self.which_conv,
                                stochdepth_rate=block_stochdepth_rate,
                                )]
        ch = out_ch
        index += 1
         # Reset expected std but still give it 1 block of growth
        if block_index == 0:
          expected_std = 1.0
        expected_std = (expected_std **2 + alpha**2)**0.5

    # Head with final conv mimicking EffNets
    self.final_conv = self.which_conv(int(1280 * ch // 440), kernel_shape=1,
                                      padding='SAME', name='final_conv')
    self.fc = hk.Linear(self.num_classes, w_init=fc_init, with_bias=True)

  def __call__(self, x, is_training=True, return_metrics=False):
    """Return the output of the final layer without any [log-]softmax."""
    # Stem
    outputs = {}
    out = self.initial_conv(x)
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


class NFBlock(hk.Module):
  """Normalizer-Free RegNet Block."""

  def __init__(self, in_ch, out_ch, expansion=2.25, se_ratio=0.5,
               kernel_size=3, group_size=8, stride=1,
               beta=1.0, alpha=0.2,
               which_conv=base.WSConv2D, activation=jax.nn.relu,
               stochdepth_rate=None, name=None):
    super().__init__(name=name)
    self.in_ch, self.out_ch = in_ch, out_ch
    self.expansion = expansion
    self.se_ratio = se_ratio
    self.kernel_size = kernel_size
    self.activation = activation
    self.beta, self.alpha = beta, alpha
    # Round expanded with based on group count
    width = int(self.in_ch * expansion)
    self.groups = width // group_size
    self.width = group_size * self.groups
    self.stride = stride
    # Conv 0 (typically expansion conv)
    self.conv0 = which_conv(self.width, kernel_shape=1, padding='SAME',
                            name='conv0')
    # Grouped NxN conv
    self.conv1 = which_conv(self.width, kernel_shape=kernel_size, stride=stride,
                            padding='SAME', feature_group_count=self.groups,
                            name='conv1')
    # Conv 2, typically projection conv
    self.conv2 = which_conv(self.out_ch, kernel_shape=1, padding='SAME',
                            name='conv2')
    # Use shortcut conv on channel change or downsample.
    self.use_projection = stride > 1 or self.in_ch != self.out_ch
    if self.use_projection:
      self.conv_shortcut = which_conv(self.out_ch, kernel_shape=1,
                                      padding='SAME', name='conv_shortcut')
    # Squeeze + Excite Module
    self.se = base.SqueezeExcite(self.width, self.width, self.se_ratio)

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
    out = 2 * self.se(out) * out  # Multiply by 2 for rescaling
    out = self.conv2(self.activation(out))
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
    if self.use_projection:
      sc_flops = base.count_conv_flops(self.in_ch, self.conv_shortcut, h, w)
    else:
      sc_flops = 0
    # SE flops happen on avg-pooled activations
    se_flops = self.se.fc0.output_size * self.width
    se_flops += self.se.fc0.output_size * self.se.fc1.output_size
    contract_flops = base.count_conv_flops(self.width, self.conv2, h, w)
    return sum([expand_flops, dw_flops, se_flops, contract_flops, sc_flops])

