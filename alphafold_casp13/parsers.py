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
"""Parsers for various standard biology or AlphaFold-specific formats."""

import pickle

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


def distance_histogram_dict(f):
  """Parses distance histogram dict pickle.

  Distance histograms are stored as pickles of dicts.

  Write one of these with contacts/write_rr_file.write_pickle_file()

  Args:
    f: File-like handle to distance histogram dict pickle.

  Returns:
    Dict with fields:
      probs: (an L x L x num_bins) histogram.
      num_bins: number of bins for each residue pair
      min_range: left hand edge of the distance histogram
      max_range: the extent of the histogram NOT the right hand edge.
  """
  contact_dict = pickle.load(f, encoding='latin1')

  num_res = len(contact_dict['sequence'])

  if not all(key in contact_dict.keys()
             for key in ['probs', 'num_bins', 'min_range', 'max_range']):
    raise ValueError('The pickled contact dict doesn\'t contain all required '
                     'keys: probs, num_bins, min_range, max_range but %s.' %
                     contact_dict.keys())
  if contact_dict['probs'].ndim != 3:
    raise ValueError(
        'Probs is not rank 3 but %d' % contact_dict['probs'].ndim)
  if contact_dict['num_bins'] != contact_dict['probs'].shape[2]:
    raise ValueError(
        'The probs shape doesn\'t match num_bins in the third dimension. '
        'Expected %d got %d.' % (contact_dict['num_bins'],
                                 contact_dict['probs'].shape[2]))
  if contact_dict['probs'].shape[:2] != (num_res, num_res):
    raise ValueError(
        'The first two probs dims (%i, %i) aren\'t equal to len(sequence) %i'
        % (contact_dict['probs'].shape[0], contact_dict['probs'].shape[1],
           num_res))
  return contact_dict


def parse_distance_histogram_dict(filepath):
  """Parses distance histogram piclkle from filepath."""
  with tf.io.gfile.GFile(filepath, 'rb') as f:
    return distance_histogram_dict(f)
