# Lint as: python3.
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Network for predicting C-beta contacts."""

from absl import logging
import sonnet
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from alphafold_casp13 import asa_output
from alphafold_casp13 import secstruct
from alphafold_casp13 import two_dim_convnet
from alphafold_casp13 import two_dim_resnet


def call_on_tuple(f):
  """Unpacks a tuple input parameter into arguments for a function f.

  Mimics tuple unpacking in lambdas, which existed in Python 2 but has been
  removed in Python 3.

  Args:
    f: A function taking multiple arguments.

  Returns:
    A function equivalent to f accepting a tuple, which is then unpacked.
  """
  return lambda args: f(*args)


class ContactsNet(sonnet.AbstractModule):
  """A network to go from sequence to distance histograms."""

  def __init__(self,
               binary_code_bits,
               data_format,
               distance_multiplier,
               features,
               features_forward,
               max_range,
               min_range,
               num_bins,
               reshape_layer,
               resolution_noise_scale,
               scalars,
               targets,
               network_2d_deep,
               torsion_bins=None,
               skip_connect=0,
               position_specific_bias_size=0,
               filters_1d=(),
               collapsed_batch_norm=False,
               is_ca_feature=False,
               asa_multiplier=0.0,
               secstruct_multiplier=0.0,
               torsion_multiplier=0.0,
               name='contacts_net'):
    """Construct position prediction network."""
    super(ContactsNet, self).__init__(name=name)

    self._filters_1d = filters_1d
    self._collapsed_batch_norm = collapsed_batch_norm
    self._is_ca_feature = is_ca_feature
    self._binary_code_bits = binary_code_bits
    self._data_format = data_format
    self._distance_multiplier = distance_multiplier
    self._features = features
    self._features_forward = features_forward
    self._max_range = max_range
    self._min_range = min_range
    self._num_bins = num_bins
    self._position_specific_bias_size = position_specific_bias_size
    self._reshape_layer = reshape_layer
    self._resolution_noise_scale = resolution_noise_scale
    self._scalars = scalars
    self._torsion_bins = torsion_bins
    self._skip_connect = skip_connect
    self._targets = targets
    self._network_2d_deep = network_2d_deep

    self.asa_multiplier = asa_multiplier
    self.secstruct_multiplier = secstruct_multiplier
    self.torsion_multiplier = torsion_multiplier

    with self._enter_variable_scope():
      if self.secstruct_multiplier > 0:
        self._secstruct = secstruct.Secstruct()
      if self.asa_multiplier > 0:
        self._asa = asa_output.ASAOutputLayer()
      if self._position_specific_bias_size:
        self._position_specific_bias = tf.get_variable(
            'position_specific_bias',
            [self._position_specific_bias_size, self._num_bins or 1],
            initializer=tf.zeros_initializer())

  def quant_threshold(self, threshold=8.0):
    """Find the bin that is 8A+: we sum mass below this bin gives contact prob.

    Args:
      threshold: The distance threshold.
    Returns:
      Index of bin.
    """
    # Note that this misuses the max_range as the range.
    return int(
        (threshold - self._min_range) * self._num_bins / float(self._max_range))

  def _build(self, crop_size_x=0, crop_size_y=0, placeholders=None):
    """Puts the network into the graph.

    Args:
      crop_size_x: Crop a chunk out in one dimension. 0 means no cropping.
      crop_size_y: Crop a chunk out in one dimension. 0 means no cropping.
      placeholders: A dict containing the placeholders needed.

    Returns:
      A Tensor with logits of size [batch_size, num_residues, 3].
    """
    crop_placeholder = placeholders['crop_placeholder']
    inputs_1d = placeholders['inputs_1d_placeholder']
    if self._is_ca_feature and 'aatype' in self._features:
      logging.info('Collapsing aatype to is_ca_feature %s',
                   inputs_1d.shape.as_list()[-1])
      assert inputs_1d.shape.as_list()[-1] <= 21 + (
          1 if 'seq_length' in self._features else 0)
      inputs_1d = inputs_1d[:, :, 7:8]
    logits = self.compute_outputs(
        inputs_1d=inputs_1d,
        residue_index=placeholders['residue_index_placeholder'],
        inputs_2d=placeholders['inputs_2d_placeholder'],
        crop_x=crop_placeholder[:, 0:2],
        crop_y=crop_placeholder[:, 2:4],
        use_on_the_fly_stats=True,
        crop_size_x=crop_size_x,
        crop_size_y=crop_size_y,
        data_format='NHWC',  # Force NHWC for evals.
    )
    return logits

  def compute_outputs(self, inputs_1d, residue_index, inputs_2d, crop_x, crop_y,
                      use_on_the_fly_stats, crop_size_x, crop_size_y,
                      data_format='NHWC'):
    """Given the inputs for a block, compute the network outputs."""
    hidden_1d = inputs_1d
    hidden_1d_list = [hidden_1d]
    if len(hidden_1d_list) != 1:
      hidden_1d = tf.concat(hidden_1d_list, 2)

    output_dimension = self._num_bins or 1
    if self._distance_multiplier > 0:
      output_dimension += 1
    logits, activations = self._build_2d_embedding(
        hidden_1d=hidden_1d,
        residue_index=residue_index,
        inputs_2d=inputs_2d,
        output_dimension=output_dimension,
        use_on_the_fly_stats=use_on_the_fly_stats,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_size_x=crop_size_x, crop_size_y=crop_size_y,
        data_format=data_format)
    logits = tf.debugging.check_numerics(
        logits, 'NaN in resnet activations', name='resnet_activations')
    if (self.secstruct_multiplier > 0 or
        self.asa_multiplier > 0 or
        self.torsion_multiplier > 0):
      # Make a 1d embedding by reducing the 2D activations.
      # We do this in the x direction and the y direction separately.

      collapse_dim = 1
      join_dim = -1
      embedding_1d = tf.concat(
          # First targets are crop_x (axis 2) which we must reduce on axis 1
          [tf.concat([tf.reduce_max(activations, axis=collapse_dim),
                      tf.reduce_mean(activations, axis=collapse_dim)],
                     axis=join_dim),
           # Next targets are crop_y (axis 1) which we must reduce on axis 2
           tf.concat([tf.reduce_max(activations, axis=collapse_dim+1),
                      tf.reduce_mean(activations, axis=collapse_dim+1)],
                     axis=join_dim)],
          axis=collapse_dim)  # Join the two crops together.
      if self._collapsed_batch_norm:
        embedding_1d = tf.contrib.layers.batch_norm(
            embedding_1d, is_training=use_on_the_fly_stats,
            fused=True, decay=0.999, scope='collapsed_batch_norm',
            data_format='NHWC')
      for i, nfil in enumerate(self._filters_1d):
        embedding_1d = tf.contrib.layers.fully_connected(
            embedding_1d,
            num_outputs=nfil,
            normalizer_fn=(
                tf.contrib.layers.batch_norm if self._collapsed_batch_norm
                else None),
            normalizer_params={'is_training': use_on_the_fly_stats,
                               'updates_collections': None},
            scope='collapsed_embed_%d' % i)

      if self.torsion_multiplier > 0:
        self.torsion_logits = tf.contrib.layers.fully_connected(
            embedding_1d,
            num_outputs=self._torsion_bins * self._torsion_bins,
            activation_fn=None,
            scope='torsion_logits')
        self.torsion_output = tf.nn.softmax(self.torsion_logits)
      if self.secstruct_multiplier > 0:
        self._secstruct.make_layer_new(embedding_1d)
      if self.asa_multiplier > 0:
        self.asa_logits = self._asa.compute_asa_output(embedding_1d)
    return logits

  @staticmethod
  def _concatenate_2d(hidden_1d, residue_index, hidden_2d, crop_x, crop_y,
                      binary_code_bits, crop_size_x, crop_size_y):
    # Form the pairwise expansion of the 1D embedding
    # And the residue offsets and (one) absolute position.
    with tf.name_scope('Features2D'):
      range_scale = 100.0  # Crude normalization factor.
      n = tf.shape(hidden_1d)[1]
      # pylint: disable=g-long-lambda
      hidden_1d_cropped_y = tf.map_fn(
          call_on_tuple(lambda c, h: tf.pad(
              h[tf.maximum(0, c[0]):c[1]],
              [[tf.maximum(0, -c[0]),
                tf.maximum(0, crop_size_y -(n - c[0]))], [0, 0]])),
          elems=(crop_y, hidden_1d), dtype=tf.float32,
          back_prop=True)
      range_n_y = tf.map_fn(
          call_on_tuple(lambda ri, c: tf.pad(
              ri[tf.maximum(0, c[0]):c[1]],
              [[tf.maximum(0, -c[0]),
                tf.maximum(0, crop_size_y -(n - c[0]))]])),
          elems=(residue_index, crop_y), dtype=tf.int32,
          back_prop=False)
      hidden_1d_cropped_x = tf.map_fn(
          call_on_tuple(lambda c, h: tf.pad(
              h[tf.maximum(0, c[0]):c[1]],
              [[tf.maximum(0, -c[0]),
                tf.maximum(0, crop_size_x -(n - c[0]))], [0, 0]])),
          elems=(crop_x, hidden_1d), dtype=tf.float32,
          back_prop=True)
      range_n_x = tf.map_fn(
          call_on_tuple(lambda ri, c: tf.pad(
              ri[tf.maximum(0, c[0]):c[1]],
              [[tf.maximum(0, -c[0]),
                tf.maximum(0, crop_size_x -(n - c[0]))]])),
          elems=(residue_index, crop_x), dtype=tf.int32,
          back_prop=False)
      # pylint: enable=g-long-lambda
      n_x = crop_size_x
      n_y = crop_size_y

      offset = (tf.expand_dims(tf.cast(range_n_x, tf.float32), 1) -
                tf.expand_dims(tf.cast(range_n_y, tf.float32), 2)) / range_scale
      position_features = [
          tf.tile(
              tf.reshape(
                  (tf.cast(range_n_y, tf.float32) - range_scale) / range_scale,
                  [-1, n_y, 1, 1]), [1, 1, n_x, 1],
              name='TileRange'),
          tf.tile(
              tf.reshape(offset, [-1, n_y, n_x, 1]), [1, 1, 1, 1],
              name='TileOffset')
      ]
      channels = 2
      if binary_code_bits:
        # Binary coding of position.
        exp_range_n_y = tf.expand_dims(range_n_y, 2)
        bin_y = tf.stop_gradient(
            tf.concat([tf.math.floormod(exp_range_n_y // (1 << i), 2)
                       for i in range(binary_code_bits)], 2))
        exp_range_n_x = tf.expand_dims(range_n_x, 2)
        bin_x = tf.stop_gradient(
            tf.concat([tf.math.floormod(exp_range_n_x // (1 << i), 2)
                       for i in range(binary_code_bits)], 2))
        position_features += [
            tf.tile(
                tf.expand_dims(tf.cast(bin_y, tf.float32), 2), [1, 1, n_x, 1],
                name='TileBinRangey'),
            tf.tile(
                tf.expand_dims(tf.cast(bin_x, tf.float32), 1), [1, n_y, 1, 1],
                name='TileBinRangex')
        ]
        channels += 2 * binary_code_bits

      augmentation_features = position_features + [
          tf.tile(tf.expand_dims(hidden_1d_cropped_x, 1),
                  [1, n_y, 1, 1], name='Tile1Dx'),
          tf.tile(tf.expand_dims(hidden_1d_cropped_y, 2),
                  [1, 1, n_x, 1], name='Tile1Dy')]
      channels += 2 * hidden_1d.shape.as_list()[-1]
      channels += hidden_2d.shape.as_list()[-1]
      hidden_2d = tf.concat(
          [hidden_2d] + augmentation_features, 3, name='Stack2Dfeatures')
    logging.info('2d stacked features are depth %d %s', channels, hidden_2d)
    hidden_2d.set_shape([None, None, None, channels])
    return hidden_2d

  def _build_2d_embedding(self, hidden_1d, residue_index, inputs_2d,
                          output_dimension, use_on_the_fly_stats, crop_x,
                          crop_y, crop_size_x, crop_size_y, data_format):
    """Returns NHWC logits and NHWC preactivations."""
    logging.info('2d %s %s', inputs_2d, data_format)

    # Stack with diagonal has already happened.
    inputs_2d_cropped = inputs_2d

    features_forward = None
    hidden_2d = inputs_2d_cropped
    hidden_2d = self._concatenate_2d(
        hidden_1d, residue_index, hidden_2d, crop_x, crop_y,
        self._binary_code_bits, crop_size_x, crop_size_y)

    config_2d_deep = self._network_2d_deep
    num_features = hidden_2d.shape.as_list()[3]
    if data_format == 'NCHW':
      logging.info('NCHW shape deep pre %s', hidden_2d)
      hidden_2d = tf.transpose(hidden_2d, perm=[0, 3, 1, 2])
      hidden_2d.set_shape([None, num_features, None, None])
      logging.info('NCHW shape deep post %s', hidden_2d)
    layers_forward = None
    if config_2d_deep.extra_blocks:
      # Optionally put some extra double-size blocks at the beginning.
      with tf.variable_scope('Deep2DExtra'):
        hidden_2d = two_dim_resnet.make_two_dim_resnet(
            input_node=hidden_2d,
            num_residues=None,  # Unused
            num_features=num_features,
            num_predictions=2 * config_2d_deep.num_filters,
            num_channels=2 * config_2d_deep.num_filters,
            num_layers=config_2d_deep.extra_blocks *
            config_2d_deep.num_layers_per_block,
            filter_size=3,
            batch_norm=config_2d_deep.use_batch_norm,
            is_training=use_on_the_fly_stats,
            fancy=True,
            final_non_linearity=True,
            atrou_rates=[1, 2, 4, 8],
            data_format=data_format,
            dropout_keep_prob=1.0
        )
        num_features = 2 * config_2d_deep.num_filters
        if self._skip_connect:
          layers_forward = hidden_2d
        if features_forward is not None:
          hidden_2d = tf.concat([hidden_2d, features_forward], 1
                                if data_format == 'NCHW' else 3)
    with tf.variable_scope('Deep2D'):
      logging.info('2d hidden shape is %s', str(hidden_2d.shape.as_list()))
      contact_pre_logits = two_dim_resnet.make_two_dim_resnet(
          input_node=hidden_2d,
          num_residues=None,  # Unused
          num_features=num_features,
          num_predictions=(config_2d_deep.num_filters
                           if self._reshape_layer else output_dimension),
          num_channels=config_2d_deep.num_filters,
          num_layers=config_2d_deep.num_blocks *
          config_2d_deep.num_layers_per_block,
          filter_size=3,
          batch_norm=config_2d_deep.use_batch_norm,
          is_training=use_on_the_fly_stats,
          fancy=True,
          final_non_linearity=self._reshape_layer,
          atrou_rates=[1, 2, 4, 8],
          data_format=data_format,
          dropout_keep_prob=1.0
      )

      contact_logits = self._output_from_pre_logits(
          contact_pre_logits, features_forward, layers_forward,
          output_dimension, data_format, crop_x, crop_y, use_on_the_fly_stats)
      if data_format == 'NCHW':
        contact_pre_logits = tf.transpose(contact_pre_logits, perm=[0, 2, 3, 1])
    # Both of these will be NHWC
    return contact_logits, contact_pre_logits

  def _output_from_pre_logits(self, contact_pre_logits, features_forward,
                              layers_forward, output_dimension, data_format,
                              crop_x, crop_y, use_on_the_fly_stats):
    """Given pre-logits, compute the final distogram/contact activations."""
    config_2d_deep = self._network_2d_deep
    if self._reshape_layer:
      in_channels = config_2d_deep.num_filters
      concat_features = [contact_pre_logits]
      if features_forward is not None:
        concat_features.append(features_forward)
        in_channels += self._features_forward
      if layers_forward is not None:
        concat_features.append(layers_forward)
        in_channels += 2 * config_2d_deep.num_filters
      if len(concat_features) > 1:
        contact_pre_logits = tf.concat(concat_features,
                                       1 if data_format == 'NCHW' else 3)

      contact_logits = two_dim_convnet.make_conv_layer(
          contact_pre_logits,
          in_channels=in_channels,
          out_channels=output_dimension,
          layer_name='output_reshape_1x1h',
          filter_size=1,
          filter_size_2=1,
          non_linearity=False,
          batch_norm=config_2d_deep.use_batch_norm,
          is_training=use_on_the_fly_stats,
          data_format=data_format)
    else:
      contact_logits = contact_pre_logits

    if data_format == 'NCHW':
      contact_logits = tf.transpose(contact_logits, perm=[0, 2, 3, 1])

    if self._position_specific_bias_size:
      # Make 2D pos-specific biases: NHWC.
      biases = build_crops_biases(
          self._position_specific_bias_size,
          self._position_specific_bias, crop_x, crop_y, back_prop=True)
      contact_logits += biases

    # Will be NHWC.
    return contact_logits

  def update_crop_fetches(self, fetches):
    """Add auxiliary outputs for a crop to the fetches."""
    if self.secstruct_multiplier > 0:
      fetches['secstruct_probs'] = self._secstruct.get_q8_probs()
    if self.asa_multiplier > 0:
      fetches['asa_output'] = self._asa.asa_output
    if self.torsion_multiplier > 0:
      fetches['torsion_probs'] = self.torsion_output


def build_crops_biases(bias_size, raw_biases, crop_x, crop_y, back_prop):
  """Take the offset-specific biases and reshape them to match current crops.

  Args:
    bias_size: how many bias variables we're storing.
    raw_biases: the bias variable
    crop_x: B x 2 array of start/end for the batch
    crop_y: B x 2 array of start/end for the batch
    back_prop: whether to backprop through the map_fn.

  Returns:
    Reshaped biases.
  """
  # First pad the biases with a copy of the final value to the maximum length.
  max_off_diag = tf.reduce_max(
      tf.maximum(tf.abs(crop_x[:, 1] - crop_y[:, 0]),
                 tf.abs(crop_y[:, 1] - crop_x[:, 0])))
  padded_bias_size = tf.maximum(bias_size, max_off_diag)
  biases = tf.concat(
      [raw_biases,
       tf.tile(raw_biases[-1:, :],
               [padded_bias_size - bias_size, 1])], axis=0)
  # Now prepend a mirror image (excluding 0th elt) for below-diagonal.
  biases = tf.concat([tf.reverse(biases[1:, :], axis=[0]), biases], axis=0)

  # Which diagonal of the full matrix each crop starts on (top left):
  start_diag = crop_x[:, 0:1] - crop_y[:, 0:1]  # B x 1
  crop_size_x = tf.reduce_max(crop_x[:, 1] - crop_x[:, 0])
  crop_size_y = tf.reduce_max(crop_y[:, 1] - crop_y[:, 0])

  # Relative offset of each row within a crop:
  # (off-diagonal decreases as y increases)
  increment = tf.expand_dims(-tf.range(0, crop_size_y), 0)  # 1 x crop_size_y

  # Index of diagonal of first element of each row, flattened.
  row_offsets = tf.reshape(start_diag + increment, [-1])  # B*crop_size_y
  logging.info('row_offsets  %s', row_offsets)

  # Make it relative to the start of the biases array. (0-th diagonal is in
  # the middle at position padded_bias_size - 1)
  row_offsets += padded_bias_size - 1

  # Map_fn to build the individual rows.
  # B*cropsizey x cropsizex x num_bins
  cropped_biases = tf.map_fn(lambda i: biases[i:i+crop_size_x, :],
                             elems=row_offsets, dtype=tf.float32,
                             back_prop=back_prop)
  logging.info('cropped_biases %s', cropped_biases)
  return tf.reshape(
      cropped_biases, [-1, crop_size_y, crop_size_x, tf.shape(raw_biases)[-1]])
