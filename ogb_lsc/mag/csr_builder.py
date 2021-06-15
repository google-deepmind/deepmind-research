# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builds CSR matrices which store the MAG graphs."""

import pathlib

from absl import app
from absl import flags
from absl import logging
import numpy as np
import scipy.sparse

# pylint: disable=g-bad-import-order
import data_utils

Path = pathlib.Path


FLAGS = flags.FLAGS

_DATA_FILES_AND_PARAMETERS = {
    'author_affiliated_with_institution_edges.npy': {
        'content_names': ('author', 'institution'),
        'use_boolean': False
    },
    'author_writes_paper_edges.npy': {
        'content_names': ('author', 'paper'),
        'use_boolean': False
    },
    'paper_cites_paper_edges.npy': {
        'content_names': ('paper', 'paper'),
        'use_boolean': True
    },
}

flags.DEFINE_string('data_root', None, 'Data root directory')
flags.DEFINE_boolean('skip_existing', True, 'Skips existing CSR files')

flags.mark_flags_as_required(['data_root'])


def _read_edge_data(path):
  try:
    return np.load(path, mmap_mode='r')
  except FileNotFoundError:
    # If the file path can't be found by np.load, use the file handle w/o mmap.
    with path.open('rb') as fid:
      return np.load(fid)


def _build_coo(edges_data, use_boolean=False):
  if use_boolean:
    mat_coo = scipy.sparse.coo_matrix(
        (np.ones_like(edges_data[1, :],
                      dtype=bool), (edges_data[0, :], edges_data[1, :])))
  else:
    mat_coo = scipy.sparse.coo_matrix(
        (edges_data[1, :], (edges_data[0, :], edges_data[1, :])))
  return mat_coo


def _get_output_paths(directory, content_names, use_boolean):
  boolean_str = '_b' if use_boolean else ''
  transpose_str = '_t' if len(set(content_names)) == 1 else ''
  output_prefix = '_'.join(content_names)
  output_prefix_t = '_'.join(content_names[::-1])
  output_filename = f'{output_prefix}{boolean_str}.npz'
  output_filename_t = f'{output_prefix_t}{boolean_str}{transpose_str}.npz'
  output_path = directory / output_filename
  output_path_t = directory / output_filename_t
  return output_path, output_path_t


def _write_csr(path, csr):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('wb') as fid:
    scipy.sparse.save_npz(fid, csr)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  raw_data_dir = Path(FLAGS.data_root) / data_utils.RAW_DIR
  preprocessed_dir = Path(FLAGS.data_root) / data_utils.PREPROCESSED_DIR

  for input_filename, parameters in _DATA_FILES_AND_PARAMETERS.items():
    input_path = raw_data_dir / input_filename
    output_path, output_path_t = _get_output_paths(preprocessed_dir,
                                                   **parameters)
    if FLAGS.skip_existing and output_path.exists() and output_path_t.exists():
      # If both files exist, skip. When only one exists, that's handled below.
      logging.info(
          '%s and %s exist: skipping. Use flag `--skip_existing=False`'
          'to force overwrite existing.', output_path, output_path_t)
      continue
    logging.info('Reading edge data from: %s', input_path)
    edge_data = _read_edge_data(input_path)
    logging.info('Building CSR matrices')
    mat_coo = _build_coo(edge_data, use_boolean=parameters['use_boolean'])
    # Convert matrices to CSR and write to disk.
    if not FLAGS.skip_existing or not output_path.exists():
      logging.info('Writing CSR matrix to: %s', output_path)
      mat_csr = mat_coo.tocsr()
      _write_csr(output_path, mat_csr)
      del mat_csr  # Free up memory asap.
    else:
      logging.info(
          '%s exists: skipping. Use flag `--skip_existing=False`'
          'to force overwrite existing.', output_path)
    if not FLAGS.skip_existing or not output_path_t.exists():
      logging.info('Writing (transposed) CSR matrix to: %s', output_path_t)
      mat_csr_t = mat_coo.transpose().tocsr()
      _write_csr(output_path_t, mat_csr_t)
      del mat_csr_t  # Free up memory asap.
    else:
      logging.info(
          '%s exists: skipping. Use flag `--skip_existing=False`'
          'to force overwrite existing.', output_path_t)
    del mat_coo  # Free up memory asap.


if __name__ == '__main__':
  app.run(main)
