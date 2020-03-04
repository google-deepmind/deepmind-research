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
"""Contact prediction convnet experiment example."""

from absl import logging
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from alphafold_casp13 import contacts_dataset
from alphafold_casp13 import contacts_network


def _int_ph(shape, name):
  return tf.placeholder(
      dtype=tf.int32, shape=shape, name=('%s_placeholder' % name))


def _float_ph(shape, name):
  return tf.placeholder(
      dtype=tf.float32, shape=shape, name=('%s_placeholder' % name))


class Contacts(object):
  """Contact prediction experiment."""

  def __init__(
      self, tfrecord, stats_file, network_config, crop_size_x, crop_size_y,
      feature_normalization, normalization_exclusion):
    """Builds the TensorFlow graph."""
    self.network_config = network_config
    self.crop_size_x = crop_size_x
    self.crop_size_y = crop_size_y

    self._feature_normalization = feature_normalization
    self._normalization_exclusion = normalization_exclusion
    self._model = contacts_network.ContactsNet(**network_config)
    self._features = network_config.features
    self._scalars = network_config.scalars
    self._targets = network_config.targets
    # Add extra targets we need.
    required_targets = ['domain_name', 'resolution', 'chain_name']
    if self.model.torsion_multiplier > 0:
      required_targets.extend([
          'phi_angles', 'phi_mask', 'psi_angles', 'psi_mask'])
    if self.model.secstruct_multiplier > 0:
      required_targets.extend(['sec_structure', 'sec_structure_mask'])
    if self.model.asa_multiplier > 0:
      required_targets.extend(['solv_surf', 'solv_surf_mask'])
    extra_targets = [t for t in required_targets if t not in self._targets]
    if extra_targets:
      targets = list(self._targets)
      targets.extend(extra_targets)
      self._targets = tuple(targets)
    logging.info('Targets %s %s extra %s',
                 type(self._targets), self._targets, extra_targets)
    logging.info('Evaluating on %s, stats: %s', tfrecord, stats_file)
    self._build_evaluation_graph(tfrecord=tfrecord, stats_file=stats_file)

  @property
  def model(self):
    return self._model

  def _get_feature_normalization(self, features):
    return {key: self._feature_normalization
            for key in features
            if key not in list(self._normalization_exclusion)}

  def _build_evaluation_graph(self, tfrecord, stats_file):
    """Constructs the graph in pieces so it can be fed."""
    with tf.name_scope('competitionsep'):
      # Construct the dataset and mapping ops.
      dataset = contacts_dataset.create_tf_dataset(
          tf_record_filename=tfrecord,
          features=tuple(self._features) + tuple(
              self._scalars) + tuple(self._targets))

      def normalize(data):
        return contacts_dataset.normalize_from_stats_file(
            features=data,
            stats_file_path=stats_file,
            feature_normalization=self._get_feature_normalization(
                self._features),
            copy_unnormalized=list(set(self._features) & set(self._targets)))

      def convert_to_legacy(features):
        return contacts_dataset.convert_to_legacy_proteins_dataset_format(
            features, self._features, self._scalars, self._targets)

      dataset = dataset.map(normalize)
      dataset = dataset.map(convert_to_legacy)
      dataset = dataset.batch(1)

      # Get a batch of tensors in the legacy ProteinsDataset format.
      iterator = tf.data.make_one_shot_iterator(dataset)
      self._input_batch = iterator.get_next()

      self.num_eval_examples = sum(
          1 for _ in tf.python_io.tf_record_iterator(tfrecord))

      logging.info('Eval batch:\n%s', self._input_batch)
      feature_dim_1d = self._input_batch.inputs_1d.shape.as_list()[-1]
      feature_dim_2d = self._input_batch.inputs_2d.shape.as_list()[-1]
      feature_dim_2d *= 3  # The diagonals will be stacked before feeding.

      # Now placeholders for the graph to compute the outputs for one crop.
      self.inputs_1d_placeholder = _float_ph(
          shape=[None, None, feature_dim_1d], name='inputs_1d')
      self.residue_index_placeholder = _int_ph(
          shape=[None, None], name='residue_index')
      self.inputs_2d_placeholder = _float_ph(
          shape=[None, None, None, feature_dim_2d], name='inputs_2d')
      # 4 ints: x_start, x_end, y_start, y_end.
      self.crop_placeholder = _int_ph(shape=[None, 4], name='crop')

      # Finally placeholders for the graph to score the complete contact map.
      self.probs_placeholder = _float_ph(shape=[None, None, None], name='probs')
      self.softmax_probs_placeholder = _float_ph(
          shape=[None, None, None, self.network_config.num_bins],
          name='softmax_probs')
      self.cb_placeholder = _float_ph(shape=[None, None, 3], name='cb')
      self.cb_mask_placeholder = _float_ph(shape=[None, None], name='cb_mask')
      self.lengths_placeholder = _int_ph(shape=[None], name='lengths')

      if self.model.secstruct_multiplier > 0:
        self.sec_structure_placeholder = _float_ph(
            shape=[None, None, 8], name='sec_structure')
        self.sec_structure_logits_placeholder = _float_ph(
            shape=[None, None, 8], name='sec_structure_logits')
        self.sec_structure_mask_placeholder = _float_ph(
            shape=[None, None, 1], name='sec_structure_mask')

      if self.model.asa_multiplier > 0:
        self.solv_surf_placeholder = _float_ph(
            shape=[None, None, 1], name='solv_surf')
        self.solv_surf_logits_placeholder = _float_ph(
            shape=[None, None, 1], name='solv_surf_logits')
        self.solv_surf_mask_placeholder = _float_ph(
            shape=[None, None, 1], name='solv_surf_mask')

      if self.model.torsion_multiplier > 0:
        self.torsions_truth_placeholder = _float_ph(
            shape=[None, None, 2], name='torsions_truth')
        self.torsions_mask_placeholder = _float_ph(
            shape=[None, None, 1], name='torsions_mask')
        self.torsion_logits_placeholder = _float_ph(
            shape=[None, None, self.network_config.torsion_bins ** 2],
            name='torsion_logits')

      # Build a dict to pass all the placeholders into build.
      placeholders = {
          'inputs_1d_placeholder': self.inputs_1d_placeholder,
          'residue_index_placeholder': self.residue_index_placeholder,
          'inputs_2d_placeholder': self.inputs_2d_placeholder,
          'crop_placeholder': self.crop_placeholder,
          'probs_placeholder': self.probs_placeholder,
          'softmax_probs_placeholder': self.softmax_probs_placeholder,
          'cb_placeholder': self.cb_placeholder,
          'cb_mask_placeholder': self.cb_mask_placeholder,
          'lengths_placeholder': self.lengths_placeholder,
      }
      if self.model.secstruct_multiplier > 0:
        placeholders.update({
            'sec_structure': self.sec_structure_placeholder,
            'sec_structure_logits_placeholder':
            self.sec_structure_logits_placeholder,
            'sec_structure_mask': self.sec_structure_mask_placeholder,})
      if self.model.asa_multiplier > 0:
        placeholders.update({
            'solv_surf': self.solv_surf_placeholder,
            'solv_surf_logits_placeholder': self.solv_surf_logits_placeholder,
            'solv_surf_mask': self.solv_surf_mask_placeholder,})
      if self.model.torsion_multiplier > 0:
        placeholders.update({
            'torsions_truth': self.torsions_truth_placeholder,
            'torsion_logits_placeholder': self.torsion_logits_placeholder,
            'torsions_truth_mask': self.torsions_mask_placeholder,})

      activations = self._model(
          crop_size_x=self.crop_size_x,
          crop_size_y=self.crop_size_y,
          placeholders=placeholders)
      self.eval_probs_softmax = tf.nn.softmax(
          activations[:, :, :, :self.network_config.num_bins])
      self.eval_probs = tf.reduce_sum(
          self.eval_probs_softmax[:, :, :, :self._model.quant_threshold()],
          axis=3)

  def get_one_example(self, sess):
    """Pull one example off the queue so we can feed it for evaluation."""
    request_dict = {
        'inputs_1d': self._input_batch.inputs_1d,
        'inputs_2d': self._input_batch.inputs_2d,
        'sequence_lengths': self._input_batch.sequence_lengths,
        'beta_positions': self._input_batch.targets.beta_positions,
        'beta_mask': self._input_batch.targets.beta_mask,
        'domain_name': self._input_batch.targets.domain_name,
        'chain_name': self._input_batch.targets.chain_name,
        'sequences': self._input_batch.sequences,
    }
    if hasattr(self._input_batch.targets, 'residue_index'):
      request_dict.update(
          {'residue_index': self._input_batch.targets.residue_index})
    if hasattr(self._input_batch.targets, 'phi_angles'):
      request_dict.update(
          {'phi_angles': self._input_batch.targets.phi_angles,
           'psi_angles': self._input_batch.targets.psi_angles,
           'phi_mask': self._input_batch.targets.phi_mask,
           'psi_mask': self._input_batch.targets.psi_mask})
    if hasattr(self._input_batch.targets, 'sec_structure'):
      request_dict.update(
          {'sec_structure': self._input_batch.targets.sec_structure,
           'sec_structure_mask': self._input_batch.targets.sec_structure_mask,})
    if hasattr(self._input_batch.targets, 'solv_surf'):
      request_dict.update(
          {'solv_surf': self._input_batch.targets.solv_surf,
           'solv_surf_mask': self._input_batch.targets.solv_surf_mask,})
    if hasattr(self._input_batch.targets, 'alpha_positions'):
      request_dict.update(
          {'alpha_positions': self._input_batch.targets.alpha_positions,
           'alpha_mask': self._input_batch.targets.alpha_mask,})
    batch = sess.run(request_dict)
    return batch
