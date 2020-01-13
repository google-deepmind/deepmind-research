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
"""TF wrapper for protein tf.Example datasets."""

import collections
import enum
import json

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


_ProteinDescription = collections.namedtuple(
    '_ProteinDescription', (
        'sequence_lengths', 'key', 'sequences', 'inputs_1d', 'inputs_2d',
        'inputs_2d_diagonal', 'crops', 'scalars', 'targets'))


class FeatureType(enum.Enum):
  ZERO_DIM = 0  # Shape [x]
  ONE_DIM = 1  # Shape [num_res, x]
  TWO_DIM = 2  # Shape [num_res, num_res, x]

# Placeholder values that will be replaced with their true value at runtime.
NUM_RES = 'num residues placeholder'

# Sizes of the protein features. NUM_RES is allowed as a placeholder to be
# replaced with the number of residues.
FEATURES = {
    'aatype': (tf.float32, [NUM_RES, 21]),
    'alpha_mask': (tf.int64, [NUM_RES, 1]),
    'alpha_positions': (tf.float32, [NUM_RES, 3]),
    'beta_mask': (tf.int64, [NUM_RES, 1]),
    'beta_positions': (tf.float32, [NUM_RES, 3]),
    'between_segment_residues': (tf.int64, [NUM_RES, 1]),
    'chain_name': (tf.string, [1]),
    'deletion_probability': (tf.float32, [NUM_RES, 1]),
    'domain_name': (tf.string, [1]),
    'gap_matrix': (tf.float32, [NUM_RES, NUM_RES, 1]),
    'hhblits_profile': (tf.float32, [NUM_RES, 22]),
    'hmm_profile': (tf.float32, [NUM_RES, 30]),
    'key': (tf.string, [1]),
    'mutual_information': (tf.float32, [NUM_RES, NUM_RES, 1]),
    'non_gapped_profile': (tf.float32, [NUM_RES, 21]),
    'num_alignments': (tf.int64, [NUM_RES, 1]),
    'num_effective_alignments': (tf.float32, [1]),
    'phi_angles': (tf.float32, [NUM_RES, 1]),
    'phi_mask': (tf.int64, [NUM_RES, 1]),
    'profile': (tf.float32, [NUM_RES, 21]),
    'profile_with_prior': (tf.float32, [NUM_RES, 22]),
    'profile_with_prior_without_gaps': (tf.float32, [NUM_RES, 21]),
    'pseudo_bias': (tf.float32, [NUM_RES, 22]),
    'pseudo_frob': (tf.float32, [NUM_RES, NUM_RES, 1]),
    'pseudolikelihood': (tf.float32, [NUM_RES, NUM_RES, 484]),
    'psi_angles': (tf.float32, [NUM_RES, 1]),
    'psi_mask': (tf.int64, [NUM_RES, 1]),
    'residue_index': (tf.int64, [NUM_RES, 1]),
    'resolution': (tf.float32, [1]),
    'reweighted_profile': (tf.float32, [NUM_RES, 22]),
    'sec_structure': (tf.int64, [NUM_RES, 8]),
    'sec_structure_mask': (tf.int64, [NUM_RES, 1]),
    'seq_length': (tf.int64, [NUM_RES, 1]),
    'sequence': (tf.string, [1]),
    'solv_surf': (tf.float32, [NUM_RES, 1]),
    'solv_surf_mask': (tf.int64, [NUM_RES, 1]),
    'superfamily': (tf.string, [1]),
}

FEATURE_TYPES = {k: v[0] for k, v in FEATURES.items()}
FEATURE_SIZES = {k: v[1] for k, v in FEATURES.items()}


def shape(feature_name, num_residues, features=None):
  """Get the shape for the given feature name.

  Args:
    feature_name: String identifier for the feature. If the feature name ends
      with "_unnormalized", theis suffix is stripped off.
    num_residues: The number of residues in the current domain - some elements
      of the shape can be dynamic and will be replaced by this value.
    features: A feature_name to (tf_dtype, shape) lookup; defaults to FEATURES.

  Returns:
    List of ints representation the tensor size.
  """
  features = features or FEATURES
  if feature_name.endswith('_unnormalized'):
    feature_name = feature_name[:-13]

  unused_dtype, raw_sizes = features[feature_name]
  replacements = {NUM_RES: num_residues}

  sizes = [replacements.get(dimension, dimension) for dimension in raw_sizes]
  return sizes


def dim(feature_name):
  """Determine the type of feature.

  Args:
    feature_name: String identifier for the feature to lookup. If the feature
      name ends with "_unnormalized", theis suffix is stripped off.

  Returns:
    A FeatureType enum describing whether the feature is of size num_res or
    num_res * num_res.

  Raises:
    ValueError: If the feature is of an unknown type.
  """
  if feature_name.endswith('_unnormalized'):
    feature_name = feature_name[:-13]

  num_dims = len(FEATURE_SIZES[feature_name])
  if num_dims == 1:
    return FeatureType.ZERO_DIM
  elif num_dims == 2 and FEATURE_SIZES[feature_name][0] == NUM_RES:
    return FeatureType.ONE_DIM
  elif num_dims == 3 and FEATURE_SIZES[feature_name][0] == NUM_RES:
    return FeatureType.TWO_DIM
  else:
    raise ValueError('Expect feature sizes to be 2 or 3, got %i' %
                     len(FEATURE_SIZES[feature_name]))


def _concat_or_zeros(tensor_list, axis, tensor_shape, name):
  """Concatenates the tensors if given, otherwise returns a tensor of zeros."""
  if tensor_list:
    return tf.concat(tensor_list, axis=axis, name=name)
  return tf.zeros(tensor_shape, name=name + '_zeros')


def parse_tfexample(raw_data, features):
  """Read a single TF Example proto and return a subset of its features.

  Args:
    raw_data: A serialized tf.Example proto.
    features: A dictionary of features, mapping string feature names to a tuple
      (dtype, shape). This dictionary should be a subset of
      protein_features.FEATURES (or the dictionary itself for all features).

  Returns:
    A dictionary of features mapping feature names to features. Only the given
    features are returned, all other ones are filtered out.
  """
  feature_map = {
      k: tf.io.FixedLenSequenceFeature(shape=(), dtype=v[0], allow_missing=True)
      for k, v in features.items()
  }
  parsed_features = tf.io.parse_single_example(raw_data, feature_map)

  # Find out what is the number of sequences and the number of alignments.
  num_residues = tf.cast(parsed_features['seq_length'][0], dtype=tf.int32)

  # Reshape the tensors according to the sequence length and num alignments.
  for k, v in parsed_features.items():
    new_shape = shape(feature_name=k, num_residues=num_residues)
    # Make sure the feature we are reshaping is not empty.
    assert_non_empty = tf.assert_greater(
        tf.size(v), 0, name='assert_%s_non_empty' % k,
        message='The feature %s is not set in the tf.Example. Either do not '
        'request the feature or use a tf.Example that has the feature set.' % k)
    with tf.control_dependencies([assert_non_empty]):
      parsed_features[k] = tf.reshape(v, new_shape, name='reshape_%s' % k)

  return parsed_features


def create_tf_dataset(tf_record_filename, features):
  """Creates an instance of tf.data.Dataset backed by a protein dataset SSTable.

  Args:
    tf_record_filename: A string with filename of the TFRecord file.
    features: A list of strings of feature names to be returned in the dataset.

  Returns:
    A tf.data.Dataset object. Its items are dictionaries from feature names to
    feature values.
  """
  # Make sure these features are always read.
  required_features = ['aatype', 'sequence', 'seq_length']
  features = list(set(features) | set(required_features))
  features = {name: FEATURES[name] for name in features}

  tf_dataset = tf.data.TFRecordDataset(filenames=[tf_record_filename])
  tf_dataset = tf_dataset.map(lambda raw: parse_tfexample(raw, features))

  return tf_dataset


def normalize_from_stats_file(
    features, stats_file_path, feature_normalization, copy_unnormalized=None):
  """Normalizes the features set in the feature_normalization by the norm stats.

  Args:
    features: A dictionary mapping feature names to feature tensors.
    stats_file_path: A string with the path of the statistics JSON file.
    feature_normalization: A dictionary specifying the normalization type for
      each input feature. Acceptable values are 'std' and 'none'. If not
      specified default to 'none'. Any extra features that are not present in
      features will be ignored.
    copy_unnormalized: A list of features whose unnormalized copy should be
      added. For any feature F in this list a feature F + "_unnormalized" will
      be added in the output dictionary containing the unnormalized feature.
      This is useful if you have a feature you want to have both in
      desired_features (normalized) and also in desired_targets (unnormalized).
      See convert_to_legacy_proteins_dataset_format for more details.

  Returns:
    A dictionary mapping features names to feature tensors. The ones that were
    specified in feature_normalization will be normalized.

  Raises:
    ValueError: If an unknown normalization mode is used.
  """
  with tf.io.gfile.GFile(stats_file_path, 'r') as f:
    norm_stats = json.loads(f.read())

  if not copy_unnormalized:
    copy_unnormalized = []
  # We need this unnormalized in convert_to_legacy_proteins_dataset_format.
  copy_unnormalized.append('num_alignments')

  for feature in copy_unnormalized:
    if feature in features:
      features[feature + '_unnormalized'] = features[feature]

  range_epsilon = 1e-12
  for key, value in features.items():
    if key not in feature_normalization or feature_normalization[key] == 'none':
      pass
    elif feature_normalization[key] == 'std':
      value = tf.cast(value, dtype=tf.float32)
      train_mean = tf.cast(norm_stats['mean'][key], dtype=tf.float32)
      train_range = tf.sqrt(tf.cast(norm_stats['var'][key], dtype=tf.float32))
      value -= train_mean
      value = tf.where(
          train_range > range_epsilon, value / train_range, value)
      features[key] = value
    else:
      raise ValueError('Unknown normalization mode %s for feature %s.'
                       % (feature_normalization[key], key))
  return features


def convert_to_legacy_proteins_dataset_format(
    features, desired_features, desired_scalars, desired_targets):
  """Converts the output of tf.Dataset to the legacy format.

  Args:
    features: A dictionary mapping feature names to feature tensors.
    desired_features: A list with the names of the desired features. These will
      be filtered out of features and returned in one of the inputs_1d or
      inputs_2d. The features concatenated in `inputs_1d`, `inputs_2d` will be
      concatenated in the same order as they were given in `desired_features`.
    desired_scalars: A list naming the desired scalars. These will
      be filtered out of features and returned in scalars. If features contain
      an unnormalized version of a desired scalar, it will be used.
    desired_targets: A list naming the desired targets. These will
      be filtered out of features and returned in targets. If features contain
      an unnormalized version of a desired target, it will be used.

  Returns:
    A _ProteinDescription namedtuple consisting of:
      sequence_length: A scalar int32 tensor with the sequence length.
      key: A string tensor with the sequence key or empty if not set features.
      sequences: A string tensor with the protein sequence.
      inputs_1d: All 1D features in a single tensor of shape
        [num_res, 1d_channels].
      inputs_2d: All 2D features in a single tensor of shape
        [num_res, num_res, 2d_channels].
      inputs_2d_diagonal: All 2D diagonal features in a single tensor of shape
        [num_res, num_res, 2d_diagonal_channels]. If no diagonal features found
        in features, the tensor will be set to inputs_2d.
      crops: A int32 tensor with the crop poisitions. If not set in features,
        it will be set to [0, num_res, 0, num_res].
      scalars: All requested scalar tensors in a list.
      targets: All requested target tensors in a list.

  Raises:
    ValueError: If the feature size is invalid.
  """
  tensors_1d = []
  tensors_2d = []
  tensors_2d_diagonal = []
  for key in desired_features:
    # Determine if the feature is 1D or 2D.
    feature_dim = dim(key)
    if feature_dim == FeatureType.ONE_DIM:
      tensors_1d.append(tf.cast(features[key], dtype=tf.float32))
    elif feature_dim == FeatureType.TWO_DIM:
      if key not in features:
        if not(key + '_cropped' in features and key + '_diagonal' in features):
          raise ValueError(
              'The 2D feature %s is not in the features dictionary and neither '
              'are its cropped and diagonal versions.' % key)
        else:
          tensors_2d.append(
              tf.cast(features[key + '_cropped'], dtype=tf.float32))
          tensors_2d_diagonal.append(
              tf.cast(features[key + '_diagonal'], dtype=tf.float32))
      else:
        tensors_2d.append(tf.cast(features[key], dtype=tf.float32))
    else:
      raise ValueError('Unexpected FeatureType returned: %s' % str(feature_dim))

  # Determine num_res from the sequence as seq_length was possibly normalized.
  num_res = tf.strings.length(features['sequence'])[0]

  # Concatenate feature tensors into a single tensor
  inputs_1d = _concat_or_zeros(
      tensors_1d, axis=1, tensor_shape=[num_res, 0],
      name='inputs_1d_concat')
  inputs_2d = _concat_or_zeros(
      tensors_2d, axis=2, tensor_shape=[num_res, num_res, 0],
      name='inputs_2d_concat')
  if tensors_2d_diagonal:
    # The legacy dataset outputs the two diagonal crops stacked
    # A1, B1, C1, A2, B2, C2. So convert the A1, A2, B1, B2, C1, C2 format.
    diagonal_crops1 = [t[:, :, :(t.shape[2] // 2)] for t in tensors_2d_diagonal]
    diagonal_crops2 = [t[:, :, (t.shape[2] // 2):] for t in tensors_2d_diagonal]
    inputs_2d_diagonal = tf.concat(diagonal_crops1 + diagonal_crops2, axis=2)
  else:
    inputs_2d_diagonal = inputs_2d

  sequence = features['sequence']
  sequence_key = features.get('key', tf.constant(['']))[0]
  if 'crops' in features:
    crops = features['crops']
  else:
    crops = tf.stack([0, tf.shape(sequence)[0], 0, tf.shape(sequence)[0]])

  scalar_tensors = []
  for key in desired_scalars:
    scalar_tensors.append(features.get(key + '_unnormalized', features[key]))

  target_tensors = []
  for key in desired_targets:
    target_tensors.append(features.get(key + '_unnormalized', features[key]))

  scalar_class = collections.namedtuple('_ScalarClass', desired_scalars)
  target_class = collections.namedtuple('_TargetClass', desired_targets)

  return _ProteinDescription(
      sequence_lengths=num_res,
      key=sequence_key,
      sequences=sequence,
      inputs_1d=inputs_1d,
      inputs_2d=inputs_2d,
      inputs_2d_diagonal=inputs_2d_diagonal,
      crops=crops,
      scalars=scalar_class(*scalar_tensors),
      targets=target_class(*target_tensors))
