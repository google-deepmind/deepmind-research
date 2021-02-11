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
"""Architecture definitions for different models."""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


# Model settings for NF-RegNets
nf_regnet_params = {
    'B0': {'width': [48, 104, 208, 440], 'depth': [1, 3, 6, 6],
           'train_imsize': 192, 'test_imsize': 224,
           'drop_rate': 0.2},
    'B1': {'width': [48, 104, 208, 440], 'depth': [2, 4, 7, 7],
           'train_imsize': 224, 'test_imsize': 256,
           'drop_rate': 0.2},
    'B2': {'width': [56, 112, 232, 488], 'depth': [2, 4, 8, 8],
           'train_imsize': 240, 'test_imsize': 272,
           'drop_rate': 0.3},
    'B3': {'width': [56, 128, 248, 528], 'depth': [2, 5, 9, 9],
           'train_imsize': 288, 'test_imsize': 320,
           'drop_rate': 0.3},
    'B4': {'width': [64, 144, 288, 616], 'depth': [2, 6, 11, 11],
           'train_imsize': 320, 'test_imsize': 384,
           'drop_rate': 0.4},
    'B5': {'width': [80, 168, 336, 704], 'depth': [3, 7, 14, 14],
           'train_imsize': 384, 'test_imsize': 456,
           'drop_rate': 0.4},
    'B6': {'width': [88, 184, 376, 792], 'depth': [3, 8, 16, 16],
           'train_imsize': 448, 'test_imsize': 528,
           'drop_rate': 0.5},
    'B7': {'width': [96, 208, 416, 880], 'depth': [4, 10, 19, 19],
           'train_imsize': 512, 'test_imsize': 600,
           'drop_rate': 0.5},
    'B8': {'width': [104, 232, 456, 968], 'depth': [4, 11, 22, 22],
           'train_imsize': 600, 'test_imsize': 672,
           'drop_rate': 0.5},
}

nfnet_params = {}


# F-series models
nfnet_params.update(**{
    'F0': {
        'width': [256, 512, 1536, 1536], 'depth': [1, 2, 6, 3],
        'train_imsize': 192, 'test_imsize': 256,
        'RA_level': '405', 'drop_rate': 0.2},
    'F1': {
        'width': [256, 512, 1536, 1536], 'depth': [2, 4, 12, 6],
        'train_imsize': 224, 'test_imsize': 320,
        'RA_level': '410', 'drop_rate': 0.3},
    'F2': {
        'width': [256, 512, 1536, 1536], 'depth': [3, 6, 18, 9],
        'train_imsize': 256, 'test_imsize': 352,
        'RA_level': '410', 'drop_rate': 0.4},
    'F3': {
        'width': [256, 512, 1536, 1536], 'depth': [4, 8, 24, 12],
        'train_imsize': 320, 'test_imsize': 416,
        'RA_level': '415', 'drop_rate': 0.4},
    'F4': {
        'width': [256, 512, 1536, 1536], 'depth': [5, 10, 30, 15],
        'train_imsize': 384, 'test_imsize': 512,
        'RA_level': '415', 'drop_rate': 0.5},
    'F5': {
        'width': [256, 512, 1536, 1536], 'depth': [6, 12, 36, 18],
        'train_imsize': 416, 'test_imsize': 544,
        'RA_level': '415', 'drop_rate': 0.5},
    'F6': {
        'width': [256, 512, 1536, 1536], 'depth': [7, 14, 42, 21],
        'train_imsize': 448, 'test_imsize': 576,
        'RA_level': '415', 'drop_rate': 0.5},
    'F7': {
        'width': [256, 512, 1536, 1536], 'depth': [8, 16, 48, 24],
        'train_imsize': 480, 'test_imsize': 608,
        'RA_level': '415', 'drop_rate': 0.5},
})

# Minor variants FN+, slightly wider
nfnet_params.update(**{
    **{f'{key}+': {**nfnet_params[key], 'width': [384, 768, 2048, 2048],}
       for key in nfnet_params}
})

# Nonlinearities with magic constants (gamma) baked in.
# Note that not all nonlinearities will be stable, especially if they are
# not perfectly monotonic. Good choices include relu, silu, and gelu.
nonlinearities = {
    'identity': lambda x: x,
    'celu': lambda x: jax.nn.celu(x) * 1.270926833152771,
    'elu': lambda x: jax.nn.elu(x) * 1.2716004848480225,
    'gelu': lambda x: jax.nn.gelu(x) * 1.7015043497085571,
    'glu': lambda x: jax.nn.glu(x) * 1.8484294414520264,
    'leaky_relu': lambda x: jax.nn.leaky_relu(x) * 1.70590341091156,
    'log_sigmoid': lambda x: jax.nn.log_sigmoid(x) * 1.9193484783172607,
    'log_softmax': lambda x: jax.nn.log_softmax(x) * 1.0002083778381348,
    'relu': lambda x: jax.nn.relu(x) * 1.7139588594436646,
    'relu6': lambda x: jax.nn.relu6(x) * 1.7131484746932983,
    'selu': lambda x: jax.nn.selu(x) * 1.0008515119552612,
    'sigmoid': lambda x: jax.nn.sigmoid(x) * 4.803835391998291,
    'silu': lambda x: jax.nn.silu(x) * 1.7881293296813965,
    'soft_sign': lambda x: jax.nn.soft_sign(x) * 2.338853120803833,
    'softplus': lambda x: jax.nn.softplus(x) * 1.9203323125839233,
    'tanh': lambda x: jnp.tanh(x) * 1.5939117670059204,
}


class WSConv2D(hk.Conv2D):
  """2D Convolution with Scaled Weight Standardization and affine gain+bias."""

  @hk.transparent
  def standardize_weight(self, weight, eps=1e-4):
    """Apply scaled WS with affine gain."""
    mean = jnp.mean(weight, axis=(0, 1, 2), keepdims=True)
    var = jnp.var(weight, axis=(0, 1, 2), keepdims=True)
    fan_in = np.prod(weight.shape[:-1])
    # Get gain
    gain = hk.get_parameter('gain', shape=(weight.shape[-1],),
                            dtype=weight.dtype, init=jnp.ones)
    # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var)
    scale = jax.lax.rsqrt(jnp.maximum(var * fan_in, eps)) * gain
    shift = mean * scale
    return weight * scale - shift

  def __call__(self, inputs: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    w_shape = self.kernel_shape + (
        inputs.shape[self.channel_index] // self.feature_group_count,
        self.output_channels)
    # Use fan-in scaled init, but WS is largely insensitive to this choice.
    w_init = hk.initializers.VarianceScaling(1.0, 'fan_in', 'normal')
    w = hk.get_parameter('w', w_shape, inputs.dtype, init=w_init)
    weight = self.standardize_weight(w, eps)
    out = jax.lax.conv_general_dilated(
        inputs, weight, window_strides=self.stride, padding=self.padding,
        lhs_dilation=self.lhs_dilation, rhs_dilation=self.kernel_dilation,
        dimension_numbers=self.dimension_numbers,
        feature_group_count=self.feature_group_count)
    # Always add bias
    bias_shape = (self.output_channels,)
    bias = hk.get_parameter('bias', bias_shape, inputs.dtype, init=jnp.zeros)
    return out + bias


def signal_metrics(x, i):
  """Things to measure about a NCHW tensor activation."""
  metrics = {}
  # Average channel-wise mean-squared
  metrics[f'avg_sq_mean_{i}'] = jnp.mean(jnp.mean(x, axis=[0, 1, 2])**2)
  # Average channel variance
  metrics[f'avg_var_{i}'] = jnp.mean(jnp.var(x, axis=[0, 1, 2]))
  return metrics


def count_conv_flops(in_ch, conv, h, w):
  """For a conv layer with in_ch inputs, count the FLOPS."""
  # How many outputs are we producing? Note this is wrong for VALID padding.
  output_shape = conv.output_channels * (h * w) / np.prod(conv.stride)
  # At each OHW location we do computation equal to (I//G) * kh * kw
  flop_per_loc = (in_ch / conv.feature_group_count)
  flop_per_loc *= np.prod(conv.kernel_shape)
  return output_shape * flop_per_loc


class SqueezeExcite(hk.Module):
  """Simple Squeeze+Excite module."""

  def __init__(self, in_ch, out_ch, se_ratio=0.5,
               hidden_ch=None, activation=jax.nn.relu,
               name=None):
    super().__init__(name=name)
    self.in_ch, self.out_ch = in_ch, out_ch
    if se_ratio is None:
      if hidden_ch is None:
        raise ValueError('Must provide one of se_ratio or hidden_ch')
      self.hidden_ch = hidden_ch
    else:
      self.hidden_ch = max(1, int(self.in_ch * se_ratio))
    self.activation = activation
    self.fc0 = hk.Linear(self.hidden_ch, with_bias=True)
    self.fc1 = hk.Linear(self.out_ch, with_bias=True)

  def __call__(self, x):
    h = jnp.mean(x, axis=[1, 2])  # Mean pool over HW extent
    h = self.fc1(self.activation(self.fc0(h)))
    h = jax.nn.sigmoid(h)[:, None, None]  # Broadcast along H, W
    return h


class StochDepth(hk.Module):
  """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

  def __init__(self, drop_rate, scale_by_keep=False, name=None):
    super().__init__(name=name)
    self.drop_rate = drop_rate
    self.scale_by_keep = scale_by_keep

  def __call__(self, x, is_training) -> jnp.ndarray:
    if not is_training:
      return x
    batch_size = x.shape[0]
    r = jax.random.uniform(hk.next_rng_key(), [batch_size, 1, 1, 1],
                           dtype=x.dtype)
    keep_prob = 1. - self.drop_rate
    binary_tensor = jnp.floor(keep_prob + r)
    if self.scale_by_keep:
      x = x / keep_prob
    return x * binary_tensor
