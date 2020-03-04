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
"""Write contact map predictions to a tf.io.gfile.

Either write a binary contact map as an RR format text file, or a
histogram prediction as a pickle of a dict containing a numpy array.
"""

import os

import numpy as np
import six.moves.cPickle as pickle
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


RR_FORMAT = """PFRMAT RR
TARGET {}
AUTHOR DM-ORIGAMI-TEAM
METHOD {}
MODEL 1
{}
"""


def save_rr_file(filename, probs, domain, sequence,
                 method='dm-contacts-resnet'):
  """Save a contact probability matrix as an RR file."""
  assert len(sequence) == probs.shape[0]
  assert len(sequence) == probs.shape[1]
  with tf.io.gfile.GFile(filename, 'w') as f:
    f.write(RR_FORMAT.format(domain, method, sequence))
    for i in range(probs.shape[0]):
      for j in range(i + 1, probs.shape[1]):
        f.write('{:d} {:d} {:d} {:d} {:f}\n'.format(
            i + 1, j + 1, 0, 8, probs[j, i]))
    f.write('END\n')


def save_torsions(torsions_dir, filebase, sequence, torsions_probs):
  """Save Torsions to a file as pickle of a dict."""
  filename = os.path.join(torsions_dir, filebase + '.torsions')
  t_dict = dict(probs=torsions_probs, sequence=sequence)
  with tf.io.gfile.GFile(filename, 'w') as fh:
    pickle.dump(t_dict, fh, protocol=2)


def save_distance_histogram(
    filename, probs, domain, sequence, min_range, max_range, num_bins):
  """Save a distance histogram prediction matrix as a pickle file."""
  dh_dict = {
      'min_range': min_range,
      'max_range': max_range,
      'num_bins': num_bins,
      'domain': domain,
      'sequence': sequence,
      'probs': probs.astype(np.float32)}
  save_distance_histogram_from_dict(filename, dh_dict)


def save_distance_histogram_from_dict(filename, dh_dict):
  """Save a distance histogram prediction matrix as a pickle file."""
  fields = ['min_range', 'max_range', 'num_bins', 'domain', 'sequence', 'probs']
  missing_fields = [f for f in fields if f not in dh_dict]
  assert not missing_fields, 'Fields {} missing from dictionary'.format(
      missing_fields)
  assert len(dh_dict['sequence']) == dh_dict['probs'].shape[0]
  assert len(dh_dict['sequence']) == dh_dict['probs'].shape[1]
  assert dh_dict['num_bins'] == dh_dict['probs'].shape[2]
  assert dh_dict['min_range'] >= 0.0
  assert dh_dict['max_range'] > 0.0
  with tf.io.gfile.GFile(filename, 'wb') as fw:
    pickle.dump(dh_dict, fw, protocol=2)


def contact_map_from_distogram(distogram_dict):
  """Split the boundary bin."""
  num_bins = distogram_dict['probs'].shape[-1]
  bin_size_angstrom = distogram_dict['max_range'] / num_bins
  threshold_cts = (8.0 - distogram_dict['min_range']) / bin_size_angstrom
  threshold_bin = int(threshold_cts)  # Round down
  pred_contacts = np.sum(distogram_dict['probs'][:, :, :threshold_bin], axis=-1)
  if threshold_bin < threshold_cts:  # Add on the fraction of the boundary bin.
    pred_contacts += distogram_dict['probs'][:, :, threshold_bin] * (
        threshold_cts - threshold_bin)
  return pred_contacts

