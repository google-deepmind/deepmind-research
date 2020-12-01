# Copyright 2020 Deepmind Technologies Limited.
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

"""Modules and networks for mesh generation."""
import sonnet as snt
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import function
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def dequantize_verts(verts, n_bits, add_noise=False):
  """Quantizes vertices and outputs integers with specified n_bits."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  verts = tf.cast(verts, tf.float32)
  verts = verts * (max_range - min_range) / range_quantize + min_range
  if add_noise:
    verts += tf.random_uniform(tf.shape(verts)) * (1 / float(range_quantize))
  return verts


def quantize_verts(verts, n_bits):
  """Dequantizes integer vertices to floats."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  verts_quantize = (
      (verts - min_range) * range_quantize / (max_range - min_range))
  return tf.cast(verts_quantize, tf.int32)


def top_k_logits(logits, k):
  """Masks logits such that logits not in top-k are small."""
  if k == 0:
    return logits
  else:
    values, _ = tf.math.top_k(logits, k=k)
    k_largest = tf.reduce_min(values)
    logits = tf.where(tf.less_equal(logits, k_largest),
                      tf.ones_like(logits)*-1e9, logits)
    return logits


def top_p_logits(logits, p):
  """Masks logits using nucleus (top-p) sampling."""
  if p == 1:
    return logits
  else:
    logit_shape = tf.shape(logits)
    seq, dim = logit_shape[1], logit_shape[2]
    logits = tf.reshape(logits, [-1, dim])
    sort_indices = tf.argsort(logits, axis=-1, direction='DESCENDING')
    probs = tf.gather(tf.nn.softmax(logits), sort_indices, batch_dims=1)
    cumprobs = tf.cumsum(probs, axis=-1, exclusive=True)
    # The top 1 candidate always will not be masked.
    # This way ensures at least 1 indices will be selected.
    sort_mask = tf.cast(tf.greater(cumprobs, p), logits.dtype)
    batch_indices = tf.tile(
        tf.expand_dims(tf.range(tf.shape(logits)[0]), axis=-1), [1, dim])
    top_p_mask = tf.scatter_nd(
        tf.stack([batch_indices, sort_indices], axis=-1), sort_mask,
        tf.shape(logits))
    logits -= top_p_mask * 1e9
    return tf.reshape(logits, [-1, seq, dim])


_function_cache = {}  # For multihead_self_attention_memory_efficient


def multihead_self_attention_memory_efficient(x,
                                              bias,
                                              num_heads,
                                              head_size=None,
                                              cache=None,
                                              epsilon=1e-6,
                                              forget=True,
                                              test_vars=None,
                                              name=None):
  """Memory-efficient Multihead scaled-dot-product self-attention.

  Based on Tensor2Tensor version but adds optional caching.

  Returns multihead-self-attention(layer_norm(x))

  Computes one attention head at a time to avoid exhausting memory.

  If forget=True, then forget all forwards activations and recompute on
  the backwards pass.

  Args:
    x: a Tensor with shape [batch, length, input_size]
    bias: an attention bias tensor broadcastable to [batch, 1, length, length]
    num_heads: an integer
    head_size: an optional integer - defaults to input_size/num_heads
    cache: Optional dict containing tensors which are the results of previous
        attentions, used for fast decoding. Expects the dict to contain two
        keys ('k' and 'v'), for the initial call the values for these keys
        should be empty Tensors of the appropriate shape.
        'k' [batch_size, 0, key_channels] 'v' [batch_size, 0, value_channels]
    epsilon: a float, for layer norm
    forget: a boolean - forget forwards activations and recompute on backprop
    test_vars: optional tuple of variables for testing purposes
    name: an optional string

  Returns:
    A Tensor.
  """
  io_size = x.get_shape().as_list()[-1]
  if head_size is None:
    assert io_size % num_heads == 0
    head_size = io_size / num_heads

  def forward_internal(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
    """Forward function."""
    n = common_layers.layer_norm_compute(x, epsilon, norm_scale, norm_bias)
    wqkv_split = tf.unstack(wqkv, num=num_heads)
    wo_split = tf.unstack(wo, num=num_heads)
    y = 0
    if cache is not None:
      cache_k = []
      cache_v = []
    for h in range(num_heads):
      with tf.control_dependencies([y] if h > 0 else []):
        combined = tf.nn.conv1d(n, wqkv_split[h], 1, 'SAME')
        q, k, v = tf.split(combined, 3, axis=2)
        if cache is not None:
          k = tf.concat([cache['k'][:, h], k], axis=1)
          v = tf.concat([cache['v'][:, h], v], axis=1)
          cache_k.append(k)
          cache_v.append(v)
        o = common_attention.scaled_dot_product_attention_simple(
            q, k, v, attention_bias)
        y += tf.nn.conv1d(o, wo_split[h], 1, 'SAME')
    if cache is not None:
      cache['k'] = tf.stack(cache_k, axis=1)
      cache['v'] = tf.stack(cache_v, axis=1)
    return y

  key = (
      'multihead_self_attention_memory_efficient %s %s' % (num_heads, epsilon))
  if not forget:
    forward_fn = forward_internal
  elif key in _function_cache:
    forward_fn = _function_cache[key]
  else:

    @function.Defun(compiled=True)
    def grad_fn(x, wqkv, wo, attention_bias, norm_scale, norm_bias, dy):
      """Custom gradient function."""
      with tf.control_dependencies([dy]):
        n = common_layers.layer_norm_compute(x, epsilon, norm_scale, norm_bias)
        wqkv_split = tf.unstack(wqkv, num=num_heads)
        wo_split = tf.unstack(wo, num=num_heads)
        deps = []
        dwqkvs = []
        dwos = []
        dn = 0
        for h in range(num_heads):
          with tf.control_dependencies(deps):
            combined = tf.nn.conv1d(n, wqkv_split[h], 1, 'SAME')
            q, k, v = tf.split(combined, 3, axis=2)
            o = common_attention.scaled_dot_product_attention_simple(
                q, k, v, attention_bias)
            partial_y = tf.nn.conv1d(o, wo_split[h], 1, 'SAME')
            pdn, dwqkvh, dwoh = tf.gradients(
                ys=[partial_y],
                xs=[n, wqkv_split[h], wo_split[h]],
                grad_ys=[dy])
            dn += pdn
            dwqkvs.append(dwqkvh)
            dwos.append(dwoh)
            deps = [dn, dwqkvh, dwoh]
        dwqkv = tf.stack(dwqkvs)
        dwo = tf.stack(dwos)
        with tf.control_dependencies(deps):
          dx, dnorm_scale, dnorm_bias = tf.gradients(
              ys=[n], xs=[x, norm_scale, norm_bias], grad_ys=[dn])
        return (dx, dwqkv, dwo, tf.zeros_like(attention_bias), dnorm_scale,
                dnorm_bias)

    @function.Defun(
        grad_func=grad_fn, compiled=True, separate_compiled_gradients=True)
    def forward_fn(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
      return forward_internal(x, wqkv, wo, attention_bias, norm_scale,
                              norm_bias)

    _function_cache[key] = forward_fn

  if bias is not None:
    bias = tf.squeeze(bias, 1)
  with tf.variable_scope(name, default_name='multihead_attention', values=[x]):
    if test_vars is not None:
      wqkv, wo, norm_scale, norm_bias = list(test_vars)
    else:
      wqkv = tf.get_variable(
          'wqkv', [num_heads, 1, io_size, 3 * head_size],
          initializer=tf.random_normal_initializer(stddev=io_size**-0.5))
      wo = tf.get_variable(
          'wo', [num_heads, 1, head_size, io_size],
          initializer=tf.random_normal_initializer(
              stddev=(head_size * num_heads)**-0.5))
      norm_scale, norm_bias = common_layers.layer_norm_vars(io_size)
    y = forward_fn(x, wqkv, wo, bias, norm_scale, norm_bias)
    y.set_shape(x.get_shape())  #  pytype: disable=attribute-error
    return y


class TransformerEncoder(snt.AbstractModule):
  """Transformer encoder.

  Sonnet Transformer encoder module as described in Vaswani et al. 2017. Uses
  the Tensor2Tensor multihead_attention function for full self attention
  (no masking). Layer norm is applied inside the residual path as in sparse
  transformers (Child 2019).

  This module expects inputs to be already embedded, and does not add position
  embeddings.
  """

  def __init__(self,
               hidden_size=256,
               fc_size=1024,
               num_heads=4,
               layer_norm=True,
               num_layers=8,
               dropout_rate=0.2,
               re_zero=True,
               memory_efficient=False,
               name='transformer_encoder'):
    """Initializes TransformerEncoder.

    Args:
      hidden_size: Size of embedding vectors.
      fc_size: Size of fully connected layer.
      num_heads: Number of attention heads.
      layer_norm: If True, apply layer normalization
      num_layers: Number of Transformer blocks, where each block contains a
        multi-head attention layer and a MLP.
      dropout_rate: Dropout rate applied immediately after the ReLU in each
        fully-connected layer.
      re_zero: If True, alpha scale residuals with zero init.
      memory_efficient: If True, recompute gradients for memory savings.
      name: Name of variable scope
    """
    super(TransformerEncoder, self).__init__(name=name)
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.layer_norm = layer_norm
    self.fc_size = fc_size
    self.num_layers = num_layers
    self.dropout_rate = dropout_rate
    self.re_zero = re_zero
    self.memory_efficient = memory_efficient

  def _build(self, inputs, is_training=False):
    """Passes inputs through Transformer encoder network.

    Args:
      inputs: Tensor of shape [batch_size, sequence_length, embed_size]. Zero
        embeddings are masked in self-attention.
      is_training: If True, dropout is applied.

    Returns:
      output: Tensor of shape [batch_size, sequence_length, embed_size].
    """
    if is_training:
      dropout_rate = self.dropout_rate
    else:
      dropout_rate = 0.

    # Identify elements with all zeros as padding, and create bias to mask
    # out padding elements in self attention.
    encoder_padding = common_attention.embedding_to_padding(inputs)
    encoder_self_attention_bias = (
        common_attention.attention_bias_ignore_padding(encoder_padding))

    x = inputs
    for layer_num in range(self.num_layers):
      with tf.variable_scope('layer_{}'.format(layer_num)):

        # Multihead self-attention from Tensor2Tensor.
        res = x
        if self.memory_efficient:
          res = multihead_self_attention_memory_efficient(
              res,
              bias=encoder_self_attention_bias,
              num_heads=self.num_heads,
              head_size=self.hidden_size // self.num_heads,
              forget=True if is_training else False,
              name='self_attention'
              )
        else:
          if self.layer_norm:
            res = common_layers.layer_norm(res, name='self_attention')
          res = common_attention.multihead_attention(
              res,
              memory_antecedent=None,
              bias=encoder_self_attention_bias,
              total_key_depth=self.hidden_size,
              total_value_depth=self.hidden_size,
              output_depth=self.hidden_size,
              num_heads=self.num_heads,
              dropout_rate=0.,
              make_image_summary=False,
              name='self_attention')
        if self.re_zero:
          res *= tf.get_variable('self_attention/alpha', initializer=0.)
        if dropout_rate:
          res = tf.nn.dropout(res, rate=dropout_rate)
        x += res

        # MLP
        res = x
        if self.layer_norm:
          res = common_layers.layer_norm(res, name='fc')
        res = tf.layers.dense(
            res, self.fc_size, activation=tf.nn.relu, name='fc_1')
        res = tf.layers.dense(res, self.hidden_size, name='fc_2')
        if self.re_zero:
          res *= tf.get_variable('fc/alpha', initializer=0.)
        if dropout_rate:
          res = tf.nn.dropout(res, rate=dropout_rate)
        x += res

    if self.layer_norm:
      output = common_layers.layer_norm(x, name='output')
    else:
      output = x
    return output


class TransformerDecoder(snt.AbstractModule):
  """Transformer decoder.

  Sonnet Transformer decoder module as described in Vaswani et al. 2017. Uses
  the Tensor2Tensor multihead_attention function for masked self attention, and
  non-masked cross attention attention. Layer norm is applied inside the
  residual path as in sparse transformers (Child 2019).

  This module expects inputs to be already embedded, and does not
  add position embeddings.
  """

  def __init__(self,
               hidden_size=256,
               fc_size=1024,
               num_heads=4,
               layer_norm=True,
               num_layers=8,
               dropout_rate=0.2,
               re_zero=True,
               memory_efficient=False,
               name='transformer_decoder'):
    """Initializes TransformerDecoder.

    Args:
      hidden_size: Size of embedding vectors.
      fc_size: Size of fully connected layer.
      num_heads: Number of attention heads.
      layer_norm: If True, apply layer normalization. If mem_efficient_attention
        is True, then layer norm is always applied.
      num_layers: Number of Transformer blocks, where each block contains a
        multi-head attention layer and a MLP.
      dropout_rate: Dropout rate applied immediately after the ReLU in each
        fully-connected layer.
      re_zero: If True, alpha scale residuals with zero init.
      memory_efficient: If True, recompute gradients for memory savings.
      name: Name of variable scope
    """
    super(TransformerDecoder, self).__init__(name=name)
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.layer_norm = layer_norm
    self.fc_size = fc_size
    self.num_layers = num_layers
    self.dropout_rate = dropout_rate
    self.re_zero = re_zero
    self.memory_efficient = memory_efficient

  def _build(self,
             inputs,
             sequential_context_embeddings=None,
             is_training=False,
             cache=None):
    """Passes inputs through Transformer decoder network.

    Args:
      inputs: Tensor of shape [batch_size, sequence_length, embed_size]. Zero
        embeddings are masked in self-attention.
      sequential_context_embeddings: Optional tensor with global context
        (e.g image embeddings) of shape
        [batch_size, context_seq_length, context_embed_size].
      is_training: If True, dropout is applied.
      cache: Optional dict containing tensors which are the results of previous
        attentions, used for fast decoding. Expects the dict to contain two
        keys ('k' and 'v'), for the initial call the values for these keys
        should be empty Tensors of the appropriate shape.
        'k' [batch_size, 0, key_channels] 'v' [batch_size, 0, value_channels]

    Returns:
      output: Tensor of shape [batch_size, sequence_length, embed_size].
    """
    if is_training:
      dropout_rate = self.dropout_rate
    else:
      dropout_rate = 0.

    # create bias to mask future elements for causal self-attention.
    seq_length = tf.shape(inputs)[1]
    decoder_self_attention_bias = common_attention.attention_bias_lower_triangle(
        seq_length)

    # If using sequential_context, identify elements with all zeros as padding,
    # and create bias to mask out padding elements in self attention.
    if sequential_context_embeddings is not None:
      encoder_padding = common_attention.embedding_to_padding(
          sequential_context_embeddings)
      encoder_decoder_attention_bias = (
          common_attention.attention_bias_ignore_padding(encoder_padding))

    x = inputs
    for layer_num in range(self.num_layers):
      with tf.variable_scope('layer_{}'.format(layer_num)):

        # If using cached decoding, access cache for current layer, and create
        # bias that enables un-masked attention into the cache
        if cache is not None:
          layer_cache = cache[layer_num]
          layer_decoder_bias = tf.zeros([1, 1, 1, 1])
        # Otherwise use standard masked bias
        else:
          layer_cache = None
          layer_decoder_bias = decoder_self_attention_bias

        # Multihead self-attention from Tensor2Tensor.
        res = x
        if self.memory_efficient:
          res = multihead_self_attention_memory_efficient(
              res,
              bias=layer_decoder_bias,
              cache=layer_cache,
              num_heads=self.num_heads,
              head_size=self.hidden_size // self.num_heads,
              forget=True if is_training else False,
              name='self_attention'
              )
        else:
          if self.layer_norm:
            res = common_layers.layer_norm(res, name='self_attention')
          res = common_attention.multihead_attention(
              res,
              memory_antecedent=None,
              bias=layer_decoder_bias,
              total_key_depth=self.hidden_size,
              total_value_depth=self.hidden_size,
              output_depth=self.hidden_size,
              num_heads=self.num_heads,
              cache=layer_cache,
              dropout_rate=0.,
              make_image_summary=False,
              name='self_attention')
        if self.re_zero:
          res *= tf.get_variable('self_attention/alpha', initializer=0.)
        if dropout_rate:
          res = tf.nn.dropout(res, rate=dropout_rate)
        x += res

        # Optional cross attention into sequential context
        if sequential_context_embeddings is not None:
          res = x
          if self.layer_norm:
            res = common_layers.layer_norm(res, name='cross_attention')
          res = common_attention.multihead_attention(
              res,
              memory_antecedent=sequential_context_embeddings,
              bias=encoder_decoder_attention_bias,
              total_key_depth=self.hidden_size,
              total_value_depth=self.hidden_size,
              output_depth=self.hidden_size,
              num_heads=self.num_heads,
              dropout_rate=0.,
              make_image_summary=False,
              name='cross_attention')
          if self.re_zero:
            res *= tf.get_variable('cross_attention/alpha', initializer=0.)
          if dropout_rate:
            res = tf.nn.dropout(res, rate=dropout_rate)
          x += res

        # FC layers
        res = x
        if self.layer_norm:
          res = common_layers.layer_norm(res, name='fc')
        res = tf.layers.dense(
            res, self.fc_size, activation=tf.nn.relu, name='fc_1')
        res = tf.layers.dense(res, self.hidden_size, name='fc_2')
        if self.re_zero:
          res *= tf.get_variable('fc/alpha', initializer=0.)
        if dropout_rate:
          res = tf.nn.dropout(res, rate=dropout_rate)
        x += res

    if self.layer_norm:
      output = common_layers.layer_norm(x, name='output')
    else:
      output = x
    return output

  def create_init_cache(self, batch_size):
    """Creates empty cache dictionary for use in fast decoding."""

    def compute_cache_shape_invariants(tensor):
      """Helper function to get dynamic shapes for cache tensors."""
      shape_list = tensor.shape.as_list()
      if len(shape_list) == 4:
        return tf.TensorShape(
            [shape_list[0], shape_list[1], None, shape_list[3]])
      elif len(shape_list) == 3:
        return tf.TensorShape([shape_list[0], None, shape_list[2]])

    # Build cache
    k = common_attention.split_heads(
        tf.zeros([batch_size, 0, self.hidden_size]), self.num_heads)
    v = common_attention.split_heads(
        tf.zeros([batch_size, 0, self.hidden_size]), self.num_heads)
    cache = [{'k': k, 'v': v} for _ in range(self.num_layers)]
    shape_invariants = tf.nest.map_structure(
        compute_cache_shape_invariants, cache)
    return cache, shape_invariants


def conv_residual_block(inputs,
                        output_channels=None,
                        downsample=False,
                        kernel_size=3,
                        re_zero=True,
                        dropout_rate=0.,
                        name='conv_residual_block'):
  """Convolutional block with residual connections for 2D or 3D inputs.

  Args:
    inputs: Input tensor of shape [batch_size, height, width, channels] or
      [batch_size, height, width, depth, channels].
    output_channels: Number of output channels.
    downsample: If True, downsample by 1/2 in this block.
    kernel_size: Spatial size of convolutional kernels.
    re_zero: If True, alpha scale residuals with zero init.
    dropout_rate: Dropout rate applied after second ReLU in residual path.
    name: Name for variable scope.

  Returns:
    outputs: Output tensor of shape [batch_size, height, width, output_channels]
      or [batch_size, height, width, depth, output_channels].
  """
  with tf.variable_scope(name):
    input_shape = inputs.get_shape().as_list()
    num_dims = len(input_shape) - 2

    if num_dims == 2:
      conv = tf.layers.conv2d
    elif num_dims == 3:
      conv = tf.layers.conv3d

    input_channels = input_shape[-1]
    if output_channels is None:
      output_channels = input_channels
    if downsample:
      shortcut = conv(
          inputs,
          filters=output_channels,
          strides=2,
          kernel_size=kernel_size,
          padding='same',
          name='conv_shortcut')
    else:
      shortcut = inputs

    res = inputs
    res = tf.nn.relu(res)
    res = conv(
        res, filters=input_channels, kernel_size=kernel_size, padding='same',
        name='conv_1')

    res = tf.nn.relu(res)
    if dropout_rate:
      res = tf.nn.dropout(res, rate=dropout_rate)
    if downsample:
      out_strides = 2
    else:
      out_strides = 1
    res = conv(
        res,
        filters=output_channels,
        kernel_size=kernel_size,
        padding='same',
        strides=out_strides,
        name='conv_2')
    if re_zero:
      res *= tf.get_variable('alpha', initializer=0.)
  return shortcut + res


class ResNet(snt.AbstractModule):
  """ResNet architecture for 2D image or 3D voxel inputs."""

  def __init__(self,
               num_dims,
               hidden_sizes=(64, 256),
               num_blocks=(2, 2),
               dropout_rate=0.1,
               re_zero=True,
               name='res_net'):
    """Initializes ResNet.

    Args:
      num_dims: Number of spatial dimensions. 2 for images or 3 for voxels.
      hidden_sizes: Sizes of hidden layers in resnet blocks.
      num_blocks: Number of resnet blocks at each size.
      dropout_rate: Dropout rate applied immediately after the ReLU in each
        fully-connected layer.
      re_zero: If True, alpha scale residuals with zero init.
      name: Name of variable scope
    """
    super(ResNet, self).__init__(name=name)
    self.num_dims = num_dims
    self.hidden_sizes = hidden_sizes
    self.num_blocks = num_blocks
    self.dropout_rate = dropout_rate
    self.re_zero = re_zero

  def _build(self, inputs, is_training=False):
    """Passes inputs through resnet.

    Args:
      inputs: Tensor of shape [batch_size, height, width, channels] or
        [batch_size, height, width, depth, channels].
      is_training: If True, dropout is applied.

    Returns:
      output: Tensor of shape [batch_size, height, width, depth, output_size].
    """
    if is_training:
      dropout_rate = self.dropout_rate
    else:
      dropout_rate = 0.

    # Initial projection with large kernel as in original resnet architecture
    if self.num_dims == 3:
      conv = tf.layers.conv3d
    elif self.num_dims == 2:
      conv = tf.layers.conv2d
    x = conv(
        inputs,
        filters=self.hidden_sizes[0],
        kernel_size=7,
        strides=2,
        padding='same',
        name='conv_input')

    if self.num_dims == 2:
      x = tf.layers.max_pooling2d(
          x, strides=2, pool_size=3, padding='same', name='pool_input')

    for d, (hidden_size,
            blocks) in enumerate(zip(self.hidden_sizes, self.num_blocks)):

      with tf.variable_scope('resolution_{}'.format(d)):

        # Downsample at the start of each collection of blocks
        x = conv_residual_block(
            x,
            downsample=False if d == 0 else True,
            dropout_rate=dropout_rate,
            output_channels=hidden_size,
            re_zero=self.re_zero,
            name='block_1_downsample')
        for i in range(blocks - 1):
          x = conv_residual_block(
              x,
              dropout_rate=dropout_rate,
              output_channels=hidden_size,
              re_zero=self.re_zero,
              name='block_{}'.format(i + 2))
    return x


class VertexModel(snt.AbstractModule):
  """Autoregressive generative model of quantized mesh vertices.

  Operates on flattened vertex sequences with a stopping token:

  [z_0, y_0, x_0, z_1, y_1, x_1, ..., z_n, y_n, z_n, STOP]

  Input vertex coordinates are embedded and tagged with learned coordinate and
  position indicators. A transformer decoder outputs logits for a quantized
  vertex distribution.
  """

  def __init__(self,
               decoder_config,
               quantization_bits,
               class_conditional=False,
               num_classes=55,
               max_num_input_verts=2500,
               use_discrete_embeddings=True,
               name='vertex_model'):
    """Initializes VertexModel.

    Args:
      decoder_config: Dictionary with TransformerDecoder config
      quantization_bits: Number of quantization used in mesh preprocessing.
      class_conditional: If True, then condition on learned class embeddings.
      num_classes: Number of classes to condition on.
      max_num_input_verts: Maximum number of vertices. Used for learned position
        embeddings.
      use_discrete_embeddings: If True, use discrete rather than continuous
        vertex embeddings.
      name: Name of variable scope
    """
    super(VertexModel, self).__init__(name=name)
    self.embedding_dim = decoder_config['hidden_size']
    self.class_conditional = class_conditional
    self.num_classes = num_classes
    self.max_num_input_verts = max_num_input_verts
    self.quantization_bits = quantization_bits
    self.use_discrete_embeddings = use_discrete_embeddings

    with self._enter_variable_scope():
      self.decoder = TransformerDecoder(**decoder_config)

  @snt.reuse_variables
  def _embed_class_label(self, labels):
    """Embeds class label with learned embedding matrix."""
    init_dict = {'embeddings': tf.glorot_uniform_initializer}
    return snt.Embed(
        vocab_size=self.num_classes,
        embed_dim=self.embedding_dim,
        initializers=init_dict,
        densify_gradients=True,
        name='class_label')(labels)

  @snt.reuse_variables
  def _prepare_context(self, context, is_training=False):
    """Prepare class label context."""
    if self.class_conditional:
      global_context_embedding = self._embed_class_label(context['class_label'])
    else:
      global_context_embedding = None
    return global_context_embedding, None

  @snt.reuse_variables
  def _embed_inputs(self, vertices, global_context_embedding=None):
    """Embeds flat vertices and adds position and coordinate information."""
    # Dequantize inputs and get shapes
    input_shape = tf.shape(vertices)
    batch_size, seq_length = input_shape[0], input_shape[1]

    # Coord indicators (x, y, z)
    coord_embeddings = snt.Embed(
        vocab_size=3,
        embed_dim=self.embedding_dim,
        initializers={'embeddings': tf.glorot_uniform_initializer},
        densify_gradients=True,
        name='coord_embeddings')(tf.mod(tf.range(seq_length), 3))

    # Position embeddings
    pos_embeddings = snt.Embed(
        vocab_size=self.max_num_input_verts,
        embed_dim=self.embedding_dim,
        initializers={'embeddings': tf.glorot_uniform_initializer},
        densify_gradients=True,
        name='coord_embeddings')(tf.floordiv(tf.range(seq_length), 3))

    # Discrete vertex value embeddings
    if self.use_discrete_embeddings:
      vert_embeddings = snt.Embed(
          vocab_size=2**self.quantization_bits + 1,
          embed_dim=self.embedding_dim,
          initializers={'embeddings': tf.glorot_uniform_initializer},
          densify_gradients=True,
          name='value_embeddings')(vertices)
    # Continuous vertex value embeddings
    else:
      vert_embeddings = tf.layers.dense(
          dequantize_verts(vertices[..., None], self.quantization_bits),
          self.embedding_dim,
          use_bias=True,
          name='value_embeddings')

    # Step zero embeddings
    if global_context_embedding is None:
      zero_embed = tf.get_variable(
          'embed_zero', shape=[1, 1, self.embedding_dim])
      zero_embed_tiled = tf.tile(zero_embed, [batch_size, 1, 1])
    else:
      zero_embed_tiled = global_context_embedding[:, None]

    # Aggregate embeddings
    embeddings = vert_embeddings + (coord_embeddings + pos_embeddings)[None]
    embeddings = tf.concat([zero_embed_tiled, embeddings], axis=1)

    return embeddings

  @snt.reuse_variables
  def _project_to_logits(self, inputs):
    """Projects transformer outputs to logits for predictive distribution."""
    return tf.layers.dense(
        inputs,
        2**self.quantization_bits + 1,  # + 1 for stopping token
        use_bias=True,
        kernel_initializer=tf.zeros_initializer(),
        name='project_to_logits')

  @snt.reuse_variables
  def _create_dist(self,
                   vertices,
                   global_context_embedding=None,
                   sequential_context_embeddings=None,
                   temperature=1.,
                   top_k=0,
                   top_p=1.,
                   is_training=False,
                   cache=None):
    """Outputs categorical dist for quantized vertex coordinates."""

    # Embed inputs
    decoder_inputs = self._embed_inputs(vertices, global_context_embedding)
    if cache is not None:
      decoder_inputs = decoder_inputs[:, -1:]

    # pass through decoder
    outputs = self.decoder(
        decoder_inputs, cache=cache,
        sequential_context_embeddings=sequential_context_embeddings,
        is_training=is_training)

    # Get logits and optionally process for sampling
    logits = self._project_to_logits(outputs)
    logits /= temperature
    logits = top_k_logits(logits, top_k)
    logits = top_p_logits(logits, top_p)
    cat_dist = tfd.Categorical(logits=logits)
    return cat_dist

  def _build(self, batch, is_training=False):
    """Pass batch through vertex model and get log probabilities under model.

    Args:
      batch: Dictionary containing:
        'vertices_flat': int32 vertex tensors of shape [batch_size, seq_length].
      is_training: If True, use dropout.

    Returns:
      pred_dist: tfd.Categorical predictive distribution with batch shape
          [batch_size, seq_length].
    """
    global_context, seq_context = self._prepare_context(
        batch, is_training=is_training)
    pred_dist = self._create_dist(
        batch['vertices_flat'][:, :-1],  # Last element not used for preds
        global_context_embedding=global_context,
        sequential_context_embeddings=seq_context,
        is_training=is_training)
    return pred_dist

  def sample(self,
             num_samples,
             context=None,
             max_sample_length=None,
             temperature=1.,
             top_k=0,
             top_p=1.,
             recenter_verts=True,
             only_return_complete=True):
    """Autoregressive sampling with caching.

    Args:
      num_samples: Number of samples to produce.
      context: Dictionary of context, such as class labels. See _prepare_context
        for details.
      max_sample_length: Maximum length of sampled vertex sequences. Sequences
        that do not complete are truncated.
      temperature: Scalar softmax temperature > 0.
      top_k: Number of tokens to keep for top-k sampling.
      top_p: Proportion of probability mass to keep for top-p sampling.
      recenter_verts: If True, center vertex samples around origin. This should
        be used if model is trained using shift augmentations.
      only_return_complete: If True, only return completed samples. Otherwise
        return all samples along with completed indicator.

    Returns:
      outputs: Output dictionary with fields:
        'completed': Boolean tensor of shape [num_samples]. If True then
          corresponding sample completed within max_sample_length.
        'vertices': Tensor of samples with shape [num_samples, num_verts, 3].
        'num_vertices': Tensor indicating number of vertices for each example
          in padded vertex samples.
        'vertices_mask': Tensor of shape [num_samples, num_verts] that masks
          corresponding invalid elements in 'vertices'.
    """
    # Obtain context for decoder
    global_context, seq_context = self._prepare_context(
        context, is_training=False)

    # num_samples is the minimum value of num_samples and the batch size of
    # context inputs (if present).
    if global_context is not None:
      num_samples = tf.minimum(num_samples, tf.shape(global_context)[0])
      global_context = global_context[:num_samples]
      if seq_context is not None:
        seq_context = seq_context[:num_samples]
    elif seq_context is not None:
      num_samples = tf.minimum(num_samples, tf.shape(seq_context)[0])
      seq_context = seq_context[:num_samples]

    def _loop_body(i, samples, cache):
      """While-loop body for autoregression calculation."""
      cat_dist = self._create_dist(
          samples,
          global_context_embedding=global_context,
          sequential_context_embeddings=seq_context,
          cache=cache,
          temperature=temperature,
          top_k=top_k,
          top_p=top_p)
      next_sample = cat_dist.sample()
      samples = tf.concat([samples, next_sample], axis=1)
      return i + 1, samples, cache

    def _stopping_cond(i, samples, cache):
      """Stopping condition for sampling while-loop."""
      del i, cache  # Unused
      return tf.reduce_any(tf.reduce_all(tf.not_equal(samples, 0), axis=-1))

    # Initial values for loop variables
    samples = tf.zeros([num_samples, 0], dtype=tf.int32)
    max_sample_length = max_sample_length or self.max_num_input_verts
    cache, cache_shape_invariants = self.decoder.create_init_cache(num_samples)
    _, v, _ = tf.while_loop(
        cond=_stopping_cond,
        body=_loop_body,
        loop_vars=(0, samples, cache),
        shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, None]),
                          cache_shape_invariants),
        maximum_iterations=max_sample_length * 3 + 1,
        back_prop=False,
        parallel_iterations=1)

    # Check if samples completed. Samples are complete if the stopping token
    # is produced.
    completed = tf.reduce_any(tf.equal(v, 0), axis=-1)

    # Get the number of vertices in the sample. This requires finding the
    # index of the stopping token. For complete samples use to argmax to get
    # first nonzero index.
    stop_index_completed = tf.argmax(
        tf.cast(tf.equal(v, 0), tf.int32), axis=-1, output_type=tf.int32)
    # For incomplete samples the stopping index is just the maximum index.
    stop_index_incomplete = (
        max_sample_length * 3 * tf.ones_like(stop_index_completed))
    stop_index = tf.where(
        completed, stop_index_completed, stop_index_incomplete)
    num_vertices = tf.floordiv(stop_index, 3)

    # Convert to 3D vertices by reshaping and re-ordering x -> y -> z
    v = v[:, :(tf.reduce_max(num_vertices) * 3)] - 1
    verts_dequantized = dequantize_verts(v, self.quantization_bits)
    vertices = tf.reshape(verts_dequantized, [num_samples, -1, 3])
    vertices = tf.stack(
        [vertices[..., 2], vertices[..., 1], vertices[..., 0]], axis=-1)

    # Pad samples to max sample length. This is required in order to concatenate
    # Samples across different replicator instances. Pad with stopping tokens
    # for incomplete samples.
    pad_size = max_sample_length - tf.shape(vertices)[1]
    vertices = tf.pad(vertices, [[0, 0], [0, pad_size], [0, 0]])

    # 3D Vertex mask
    vertices_mask = tf.cast(
        tf.range(max_sample_length)[None] < num_vertices[:, None], tf.float32)

    if recenter_verts:
      vert_max = tf.reduce_max(
          vertices - 1e10 * (1. - vertices_mask)[..., None], axis=1,
          keepdims=True)
      vert_min = tf.reduce_min(
          vertices + 1e10 * (1. - vertices_mask)[..., None], axis=1,
          keepdims=True)
      vert_centers = 0.5 * (vert_max + vert_min)
      vertices -= vert_centers
    vertices *= vertices_mask[..., None]

    if only_return_complete:
      vertices = tf.boolean_mask(vertices, completed)
      num_vertices = tf.boolean_mask(num_vertices, completed)
      vertices_mask = tf.boolean_mask(vertices_mask, completed)
      completed = tf.boolean_mask(completed, completed)

    # Outputs
    outputs = {
        'completed': completed,
        'vertices': vertices,
        'num_vertices': num_vertices,
        'vertices_mask': vertices_mask,
    }
    return outputs


class ImageToVertexModel(VertexModel):
  """Generative model of quantized mesh vertices with image conditioning.

  Operates on flattened vertex sequences with a stopping token:

  [z_0, y_0, x_0, z_1, y_1, x_1, ..., z_n, y_n, z_n, STOP]

  Input vertex coordinates are embedded and tagged with learned coordinate and
  position indicators. A transformer decoder outputs logits for a quantized
  vertex distribution. Image inputs are encoded and used to condition the
  vertex decoder.
  """

  def __init__(self,
               res_net_config,
               decoder_config,
               quantization_bits,
               use_discrete_embeddings=True,
               max_num_input_verts=2500,
               name='image_to_vertex_model'):
    """Initializes VoxelToVertexModel.

    Args:
      res_net_config: Dictionary with ResNet config.
      decoder_config: Dictionary with TransformerDecoder config.
      quantization_bits: Number of quantization used in mesh preprocessing.
      use_discrete_embeddings: If True, use discrete rather than continuous
        vertex embeddings.
      max_num_input_verts: Maximum number of vertices. Used for learned position
        embeddings.
      name: Name of variable scope
    """
    super(ImageToVertexModel, self).__init__(
        decoder_config=decoder_config,
        quantization_bits=quantization_bits,
        max_num_input_verts=max_num_input_verts,
        use_discrete_embeddings=use_discrete_embeddings,
        name=name)

    with self._enter_variable_scope():
      self.res_net = ResNet(num_dims=2, **res_net_config)

  @snt.reuse_variables
  def _prepare_context(self, context, is_training=False):

    # Pass images through encoder
    image_embeddings = self.res_net(
        context['image'] - 0.5, is_training=is_training)

    # Add 2D coordinate grid embedding
    processed_image_resolution = tf.shape(image_embeddings)[1]
    x = tf.linspace(-1., 1., processed_image_resolution)
    image_coords = tf.stack(tf.meshgrid(x, x), axis=-1)
    image_coord_embeddings = tf.layers.dense(
        image_coords,
        self.embedding_dim,
        use_bias=True,
        name='image_coord_embeddings')
    image_embeddings += image_coord_embeddings[None]

    # Reshape spatial grid to sequence
    batch_size = tf.shape(image_embeddings)[0]
    sequential_context_embedding = tf.reshape(
        image_embeddings, [batch_size, -1, self.embedding_dim])

    return None, sequential_context_embedding


class VoxelToVertexModel(VertexModel):
  """Generative model of quantized mesh vertices with voxel conditioning.

  Operates on flattened vertex sequences with a stopping token:

  [z_0, y_0, x_0, z_1, y_1, x_1, ..., z_n, y_n, z_n, STOP]

  Input vertex coordinates are embedded and tagged with learned coordinate and
  position indicators. A transformer decoder outputs logits for a quantized
  vertex distribution. Image inputs are encoded and used to condition the
  vertex decoder.
  """

  def __init__(self,
               res_net_config,
               decoder_config,
               quantization_bits,
               use_discrete_embeddings=True,
               max_num_input_verts=2500,
               name='voxel_to_vertex_model'):
    """Initializes VoxelToVertexModel.

    Args:
      res_net_config: Dictionary with ResNet config.
      decoder_config: Dictionary with TransformerDecoder config.
      quantization_bits: Integer number of bits used for vertex quantization.
      use_discrete_embeddings: If True, use discrete rather than continuous
        vertex embeddings.
      max_num_input_verts: Maximum number of vertices. Used for learned position
        embeddings.
      name: Name of variable scope
    """
    super(VoxelToVertexModel, self).__init__(
        decoder_config=decoder_config,
        quantization_bits=quantization_bits,
        max_num_input_verts=max_num_input_verts,
        use_discrete_embeddings=use_discrete_embeddings,
        name=name)

    with self._enter_variable_scope():
      self.res_net = ResNet(num_dims=3, **res_net_config)

  @snt.reuse_variables
  def _prepare_context(self, context, is_training=False):

    # Embed binary input voxels
    voxel_embeddings = snt.Embed(
        vocab_size=2,
        embed_dim=self.pre_embed_dim,
        initializers={'embeddings': tf.glorot_uniform_initializer},
        densify_gradients=True,
        name='voxel_embeddings')(context['voxels'])

    # Pass embedded voxels through voxel encoder
    voxel_embeddings = self.res_net(
        voxel_embeddings, is_training=is_training)

    # Add 3D coordinate grid embedding
    processed_voxel_resolution = tf.shape(voxel_embeddings)[1]
    x = tf.linspace(-1., 1., processed_voxel_resolution)
    voxel_coords = tf.stack(tf.meshgrid(x, x, x), axis=-1)
    voxel_coord_embeddings = tf.layers.dense(
        voxel_coords,
        self.embedding_dim,
        use_bias=True,
        name='voxel_coord_embeddings')
    voxel_embeddings += voxel_coord_embeddings[None]

    # Reshape spatial grid to sequence
    batch_size = tf.shape(voxel_embeddings)[0]
    sequential_context_embedding = tf.reshape(
        voxel_embeddings, [batch_size, -1, self.embedding_dim])

    return None, sequential_context_embedding


class FaceModel(snt.AbstractModule):
  """Autoregressive generative model of n-gon meshes.

  Operates on sets of input vertices as well as flattened face sequences with
  new face and stopping tokens:

  [f_0^0, f_0^1, f_0^2, NEW, f_1^0, f_1^1, ..., STOP]

  Input vertices are encoded using a Transformer encoder.

  Input face sequences are embedded and tagged with learned position indicators,
  as well as their corresponding vertex embeddings. A transformer decoder
  outputs a pointer which is compared to each vertex embedding to obtain a
  distribution over vertex indices.
  """

  def __init__(self,
               encoder_config,
               decoder_config,
               class_conditional=True,
               num_classes=55,
               decoder_cross_attention=True,
               use_discrete_vertex_embeddings=True,
               quantization_bits=8,
               max_seq_length=5000,
               name='face_model'):
    """Initializes FaceModel.

    Args:
      encoder_config: Dictionary with TransformerEncoder config.
      decoder_config: Dictionary with TransformerDecoder config.
      class_conditional: If True, then condition on learned class embeddings.
      num_classes: Number of classes to condition on.
      decoder_cross_attention: If True, the use cross attention from decoder
        querys into encoder outputs.
      use_discrete_vertex_embeddings: If True, use discrete vertex embeddings.
      quantization_bits: Number of quantization bits for discrete vertex
        embeddings.
      max_seq_length: Maximum face sequence length. Used for learned position
        embeddings.
      name: Name of variable scope
    """
    super(FaceModel, self).__init__(name=name)
    self.embedding_dim = decoder_config['hidden_size']
    self.class_conditional = class_conditional
    self.num_classes = num_classes
    self.max_seq_length = max_seq_length
    self.decoder_cross_attention = decoder_cross_attention
    self.use_discrete_vertex_embeddings = use_discrete_vertex_embeddings
    self.quantization_bits = quantization_bits

    with self._enter_variable_scope():
      self.decoder = TransformerDecoder(**decoder_config)
      self.encoder = TransformerEncoder(**encoder_config)

  @snt.reuse_variables
  def _embed_class_label(self, labels):
    """Embeds class label with learned embedding matrix."""
    init_dict = {'embeddings': tf.glorot_uniform_initializer}
    return snt.Embed(
        vocab_size=self.num_classes,
        embed_dim=self.embedding_dim,
        initializers=init_dict,
        densify_gradients=True,
        name='class_label')(labels)

  @snt.reuse_variables
  def _prepare_context(self, context, is_training=False):
    """Prepare class label and vertex context."""
    if self.class_conditional:
      global_context_embedding = self._embed_class_label(context['class_label'])
    else:
      global_context_embedding = None
    vertex_embeddings = self._embed_vertices(
        context['vertices'], context['vertices_mask'],
        is_training=is_training)
    if self.decoder_cross_attention:
      sequential_context_embeddings = (
          vertex_embeddings *
          tf.pad(context['vertices_mask'], [[0, 0], [2, 0]],
                 constant_values=1)[..., None])
    else:
      sequential_context_embeddings = None
    return (vertex_embeddings, global_context_embedding,
            sequential_context_embeddings)

  @snt.reuse_variables
  def _embed_vertices(self, vertices, vertices_mask, is_training=False):
    """Embeds vertices with transformer encoder."""
    # num_verts = tf.shape(vertices)[1]
    if self.use_discrete_vertex_embeddings:
      vertex_embeddings = 0.
      verts_quantized = quantize_verts(vertices, self.quantization_bits)
      for c in range(3):
        vertex_embeddings += snt.Embed(
            vocab_size=256,
            embed_dim=self.embedding_dim,
            initializers={'embeddings': tf.glorot_uniform_initializer},
            densify_gradients=True,
            name='coord_{}'.format(c))(verts_quantized[..., c])
    else:
      vertex_embeddings = tf.layers.dense(
          vertices, self.embedding_dim, use_bias=True, name='vertex_embeddings')
    vertex_embeddings *= vertices_mask[..., None]

    # Pad vertex embeddings with learned embeddings for stopping and new face
    # tokens
    stopping_embeddings = tf.get_variable(
        'stopping_embeddings', shape=[1, 2, self.embedding_dim])
    stopping_embeddings = tf.tile(stopping_embeddings,
                                  [tf.shape(vertices)[0], 1, 1])
    vertex_embeddings = tf.concat(
        [stopping_embeddings, vertex_embeddings], axis=1)

    # Pass through Transformer encoder
    vertex_embeddings = self.encoder(vertex_embeddings, is_training=is_training)
    return vertex_embeddings

  @snt.reuse_variables
  def _embed_inputs(self, faces_long, vertex_embeddings,
                    global_context_embedding=None):
    """Embeds face sequences and adds within and between face positions."""

    # Face value embeddings are gathered vertex embeddings
    face_embeddings = tf.gather(vertex_embeddings, faces_long, batch_dims=1)

    # Position embeddings
    pos_embeddings = snt.Embed(
        vocab_size=self.max_seq_length,
        embed_dim=self.embedding_dim,
        initializers={'embeddings': tf.glorot_uniform_initializer},
        densify_gradients=True,
        name='coord_embeddings')(tf.range(tf.shape(faces_long)[1]))

    # Step zero embeddings
    batch_size = tf.shape(face_embeddings)[0]
    if global_context_embedding is None:
      zero_embed = tf.get_variable(
          'embed_zero', shape=[1, 1, self.embedding_dim])
      zero_embed_tiled = tf.tile(zero_embed, [batch_size, 1, 1])
    else:
      zero_embed_tiled = global_context_embedding[:, None]

    # Aggregate embeddings
    embeddings = face_embeddings + pos_embeddings[None]
    embeddings = tf.concat([zero_embed_tiled, embeddings], axis=1)

    return embeddings

  @snt.reuse_variables
  def _project_to_pointers(self, inputs):
    """Projects transformer outputs to pointer vectors."""
    return tf.layers.dense(
        inputs,
        self.embedding_dim,
        use_bias=True,
        kernel_initializer=tf.zeros_initializer(),
        name='project_to_pointers'
        )

  @snt.reuse_variables
  def _create_dist(self,
                   vertex_embeddings,
                   vertices_mask,
                   faces_long,
                   global_context_embedding=None,
                   sequential_context_embeddings=None,
                   temperature=1.,
                   top_k=0,
                   top_p=1.,
                   is_training=False,
                   cache=None):
    """Outputs categorical dist for vertex indices."""

    # Embed inputs
    decoder_inputs = self._embed_inputs(
        faces_long, vertex_embeddings, global_context_embedding)

    # Pass through Transformer decoder
    if cache is not None:
      decoder_inputs = decoder_inputs[:, -1:]
    decoder_outputs = self.decoder(
        decoder_inputs,
        cache=cache,
        sequential_context_embeddings=sequential_context_embeddings,
        is_training=is_training)

    # Get pointers
    pred_pointers = self._project_to_pointers(decoder_outputs)

    # Get logits and mask
    logits = tf.matmul(pred_pointers, vertex_embeddings, transpose_b=True)
    logits /= tf.sqrt(float(self.embedding_dim))
    f_verts_mask = tf.pad(
        vertices_mask, [[0, 0], [2, 0]], constant_values=1.)[:, None]
    logits *= f_verts_mask
    logits -= (1. - f_verts_mask) * 1e9
    logits /= temperature
    logits = top_k_logits(logits, top_k)
    logits = top_p_logits(logits, top_p)
    return tfd.Categorical(logits=logits)

  def _build(self, batch, is_training=False):
    """Pass batch through face model and get log probabilities.

    Args:
      batch: Dictionary containing:
        'vertices_dequantized': Tensor of shape [batch_size, num_vertices, 3].
        'faces': int32 tensor of shape [batch_size, seq_length] with flattened
          faces.
        'vertices_mask': float32 tensor with shape
          [batch_size, num_vertices] that masks padded elements in 'vertices'.
      is_training: If True, use dropout.

    Returns:
      pred_dist: tfd.Categorical predictive distribution with batch shape
          [batch_size, seq_length].
    """
    vertex_embeddings, global_context, seq_context = self._prepare_context(
        batch, is_training=is_training)
    pred_dist = self._create_dist(
        vertex_embeddings,
        batch['vertices_mask'],
        batch['faces'][:, :-1],
        global_context_embedding=global_context,
        sequential_context_embeddings=seq_context,
        is_training=is_training)
    return pred_dist

  def sample(self,
             context,
             max_sample_length=None,
             temperature=1.,
             top_k=0,
             top_p=1.,
             only_return_complete=True):
    """Sample from face model using caching.

    Args:
      context: Dictionary of context, including 'vertices' and 'vertices_mask'.
        See _prepare_context for details.
      max_sample_length: Maximum length of sampled vertex sequences. Sequences
        that do not complete are truncated.
      temperature: Scalar softmax temperature > 0.
      top_k: Number of tokens to keep for top-k sampling.
      top_p: Proportion of probability mass to keep for top-p sampling.
      only_return_complete: If True, only return completed samples. Otherwise
        return all samples along with completed indicator.

    Returns:
      outputs: Output dictionary with fields:
        'completed': Boolean tensor of shape [num_samples]. If True then
          corresponding sample completed within max_sample_length.
        'faces': Tensor of samples with shape [num_samples, num_verts, 3].
        'num_face_indices': Tensor indicating number of vertices for each
          example in padded vertex samples.
    """
    vertex_embeddings, global_context, seq_context = self._prepare_context(
        context, is_training=False)
    num_samples = tf.shape(vertex_embeddings)[0]

    def _loop_body(i, samples, cache):
      """While-loop body for autoregression calculation."""
      pred_dist = self._create_dist(
          vertex_embeddings,
          context['vertices_mask'],
          samples,
          global_context_embedding=global_context,
          sequential_context_embeddings=seq_context,
          cache=cache,
          temperature=temperature,
          top_k=top_k,
          top_p=top_p)
      next_sample = pred_dist.sample()[:, -1:]
      samples = tf.concat([samples, next_sample], axis=1)
      return i + 1, samples, cache

    def _stopping_cond(i, samples, cache):
      """Stopping conditions for autoregressive calculation."""
      del i, cache  # Unused
      return tf.reduce_any(tf.reduce_all(tf.not_equal(samples, 0), axis=-1))

    # While loop sampling with caching
    samples = tf.zeros([num_samples, 0], dtype=tf.int32)
    max_sample_length = max_sample_length or self.max_seq_length
    cache, cache_shape_invariants = self.decoder.create_init_cache(num_samples)
    _, f, _ = tf.while_loop(
        cond=_stopping_cond,
        body=_loop_body,
        loop_vars=(0, samples, cache),
        shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, None]),
                          cache_shape_invariants),
        back_prop=False,
        parallel_iterations=1,
        maximum_iterations=max_sample_length)

    # Record completed samples
    complete_samples = tf.reduce_any(tf.equal(f, 0), axis=-1)

    # Find number of faces
    sample_length = tf.shape(f)[-1]
    # Get largest new face (1) index as stopping point for incomplete samples.
    max_one_ind = tf.reduce_max(
        tf.range(sample_length)[None] * tf.cast(tf.equal(f, 1), tf.int32),
        axis=-1)
    zero_inds = tf.cast(
        tf.argmax(tf.cast(tf.equal(f, 0), tf.int32), axis=-1), tf.int32)
    num_face_indices = tf.where(complete_samples, zero_inds, max_one_ind) + 1

    # Mask faces beyond stopping token with zeros
    # This mask has a -1 in order to replace the last new face token with zero
    faces_mask = tf.cast(
        tf.range(sample_length)[None] < num_face_indices[:, None] - 1, tf.int32)
    f *= faces_mask
    # This is the real mask
    faces_mask = tf.cast(
        tf.range(sample_length)[None] < num_face_indices[:, None], tf.int32)

    # Pad to maximum size with zeros
    pad_size = max_sample_length - sample_length
    f = tf.pad(f, [[0, 0], [0, pad_size]])

    if only_return_complete:
      f = tf.boolean_mask(f, complete_samples)
      num_face_indices = tf.boolean_mask(num_face_indices, complete_samples)
      context = tf.nest.map_structure(
          lambda x: tf.boolean_mask(x, complete_samples), context)
      complete_samples = tf.boolean_mask(complete_samples, complete_samples)

    # outputs
    outputs = {
        'context': context,
        'completed': complete_samples,
        'faces': f,
        'num_face_indices': num_face_indices,
    }
    return outputs
