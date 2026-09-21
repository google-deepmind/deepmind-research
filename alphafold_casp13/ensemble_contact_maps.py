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

"""Form a weighted average of several distograms.

Can also/instead form a weighted average of a set of distance histogram pickle
files, so long as they have identical hyperparameters.
"""

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from alphafold_casp13 import distogram_io
from alphafold_casp13 import parsers

flags.DEFINE_list(
    'pickle_dirs', [],
    'Comma separated list of directories with pickle files to ensemble.')
flags.DEFINE_list(
    'weights', [],
    'Comma separated list of weights for the pickle files from different dirs.')
flags.DEFINE_string(
    'output_dir', None, 'Directory where to save results of the evaluation.')
FLAGS = flags.FLAGS


def ensemble_distance_histograms(pickle_dirs, weights, output_dir):
  """Find all the contact maps in the first dir, then ensemble across dirs."""
  if len(pickle_dirs) <= 1:
    logging.warning('Pointless to ensemble %d pickle_dirs %s',
                    len(pickle_dirs), pickle_dirs)
    # Carry on if there's one dir, otherwise do nothing.
    if not pickle_dirs:
      return

  tf.io.gfile.makedirs(output_dir)
  one_dir_pickle_files = tf.io.gfile.glob(
      os.path.join(pickle_dirs[0], '*.pickle'))
  assert one_dir_pickle_files, pickle_dirs[0]
  original_files = len(one_dir_pickle_files)
  logging.info('Found %d files %d in first of %d dirs',
               original_files, len(one_dir_pickle_files), len(pickle_dirs))
  targets = [os.path.splitext(os.path.basename(f))[0]
             for f in one_dir_pickle_files]
  skipped = 0
  wrote = 0
  for t in targets:
    dump_file = os.path.join(output_dir, t + '.pickle')
    pickle_files = [os.path.join(pickle_dir, t + '.pickle')
                    for pickle_dir in pickle_dirs]
    _, new_dict = ensemble_one_distance_histogram(pickle_files, weights)
    if new_dict is not None:
      wrote += 1
      distogram_io.save_distance_histogram_from_dict(dump_file, new_dict)
      msg = 'Distograms Wrote %s %d / %d Skipped %d %s' % (
          t, wrote, len(one_dir_pickle_files), skipped, dump_file)
      logging.info(msg)


def ensemble_one_distance_histogram(pickle_files, weights):
  """Average the given pickle_files and dump."""
  dicts = []
  sequence = None
  max_dim = None
  for picklefile in pickle_files:
    if not tf.io.gfile.exists(picklefile):
      logging.warning('missing %s', picklefile)
      break
    logging.info('loading pickle file %s', picklefile)
    distance_histogram_dict = parsers.parse_distance_histogram_dict(picklefile)
    if sequence is None:
      sequence = distance_histogram_dict['sequence']
    else:
      assert sequence == distance_histogram_dict['sequence'], '%s vs %s' % (
          sequence, distance_histogram_dict['sequence'])
    dicts.append(distance_histogram_dict)
    assert dicts[-1]['probs'].shape[0] == dicts[-1]['probs'].shape[1], (
        '%d vs %d' % (dicts[-1]['probs'].shape[0], dicts[-1]['probs'].shape[1]))
    assert (dicts[0]['probs'].shape[0:2] == dicts[-1]['probs'].shape[0:2]
           ), ('%d vs %d' % (dicts[0]['probs'].shape, dicts[-1]['probs'].shape))
    if max_dim is None or max_dim < dicts[-1]['probs'].shape[2]:
      max_dim = dicts[-1]['probs'].shape[2]
  if len(dicts) != len(pickle_files):
    logging.warning('length mismatch\n%s\nVS\n%s', dicts, pickle_files)
    return sequence, None
  ensemble_hist = (
      sum(w * c['probs'] for w, c in zip(weights, dicts)) / sum(weights))
  new_dict = dict(dicts[0])
  new_dict['probs'] = ensemble_hist
  return sequence, new_dict


def main(argv):
  del argv  # Unused.
  num_dirs = len(FLAGS.pickle_dirs)
  if FLAGS.weights:
    assert len(FLAGS.weights) == num_dirs, (
        'Supply as many weights as pickle_dirs, or no weights')
    weights = [float(w) for w in FLAGS.weights]
  else:
    weights = [1.0 for w in range(num_dirs)]

  ensemble_distance_histograms(
      pickle_dirs=FLAGS.pickle_dirs,
      weights=weights,
      output_dir=FLAGS.output_dir)

if __name__ == '__main__':
  app.run(main)
