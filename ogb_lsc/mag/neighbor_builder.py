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

"""Find neighborhoods around paper feature embeddings."""

import pathlib

from absl import app
from absl import flags
from absl import logging
import annoy
import numpy as np
import scipy.sparse as sp

# pylint: disable=g-bad-import-order
import data_utils

Path = pathlib.Path


_PAPER_PAPER_B_PATH = 'ogb_mag_adjacencies/paper_paper_b.npz'

FLAGS = flags.FLAGS
flags.DEFINE_string('data_root', None, 'Data root directory')


def _read_paper_pca_features():
  data_root = Path(FLAGS.data_root)
  path = data_root / data_utils.PCA_PAPER_FEATURES_FILENAME
  with open(path, 'rb') as fid:
    return np.load(fid)


def _read_adjacency_indices():
  # Get adjacencies.
  return data_utils.get_arrays(
      data_root=FLAGS.data_root,
      use_fused_node_labels=False,
      use_fused_node_adjacencies=False,
      return_pca_embeddings=False,
  )


def build_annoy_index(features):
  """Build the Annoy index."""
  logging.info('Building annoy index')
  num_vectors, vector_size = features.shape
  annoy_index = annoy.AnnoyIndex(vector_size, 'euclidean')
  for i, x in enumerate(features):
    annoy_index.add_item(i, x)
    if i % 1000000 == 0:
      logging.info('Adding: %d / %d (%.3g %%)', i, num_vectors,
                   100 * i / num_vectors)
  n_trees = 10
  _ = annoy_index.build(n_trees)
  return annoy_index


def _get_annoy_index_path():
  return Path(FLAGS.data_root) / data_utils.PREPROCESSED_DIR / 'annoy_index.ann'


def save_annoy_index(annoy_index):
  logging.info('Saving annoy index')
  index_path = _get_annoy_index_path()
  index_path.parent.mkdir(parents=True, exist_ok=True)
  annoy_index.save(str(index_path))


def read_annoy_index(features):
  index_path = _get_annoy_index_path()
  vector_size = features.shape[1]
  annoy_index = annoy.AnnoyIndex(vector_size, 'euclidean')
  annoy_index.load(str(index_path))
  return annoy_index


def compute_neighbor_indices_and_distances(features):
  """Use the pre-built Annoy index to compute neighbor indices and distances."""
  logging.info('Computing neighbors and distances')
  annoy_index = read_annoy_index(features)
  num_vectors = features.shape[0]

  k = 20
  pad_k = 5
  search_k = -1
  neighbor_indices = np.zeros([num_vectors, k + pad_k + 1], dtype=np.int32)
  neighbor_distances = np.zeros([num_vectors, k + pad_k + 1], dtype=np.float32)
  for i in range(num_vectors):
    neighbor_indices[i], neighbor_distances[i] = annoy_index.get_nns_by_item(
        i, k + pad_k + 1, search_k=search_k, include_distances=True)
    if i % 10000 == 0:
      logging.info('Finding neighbors %d / %d', i, num_vectors)
  return neighbor_indices, neighbor_distances


def _write_neighbors(neighbor_indices, neighbor_distances):
  """Write neighbor indices and distances."""
  logging.info('Writing neighbors')
  indices_path = Path(FLAGS.data_root) / data_utils.NEIGHBOR_INDICES_FILENAME
  distances_path = (
      Path(FLAGS.data_root) / data_utils.NEIGHBOR_DISTANCES_FILENAME)
  indices_path.parent.mkdir(parents=True, exist_ok=True)
  distances_path.parent.mkdir(parents=True, exist_ok=True)
  with open(indices_path, 'wb') as fid:
    np.save(fid, neighbor_indices)
  with open(distances_path, 'wb') as fid:
    np.save(fid, neighbor_distances)


def _write_fused_edges(fused_paper_adjacency_matrix):
  """Write fused edges."""
  data_root = Path(FLAGS.data_root)
  edges_path = data_root / data_utils.FUSED_PAPER_EDGES_FILENAME
  edges_t_path = data_root / data_utils.FUSED_PAPER_EDGES_T_FILENAME
  edges_path.parent.mkdir(parents=True, exist_ok=True)
  edges_t_path.parent.mkdir(parents=True, exist_ok=True)
  with open(edges_path, 'wb') as fid:
    sp.save_npz(fid, fused_paper_adjacency_matrix)
  with open(edges_t_path, 'wb') as fid:
    sp.save_npz(fid, fused_paper_adjacency_matrix.T)


def _write_fused_nodes(fused_node_labels):
  """Write fused nodes."""
  labels_path = Path(FLAGS.data_root) / data_utils.FUSED_NODE_LABELS_FILENAME
  labels_path.parent.mkdir(parents=True, exist_ok=True)
  with open(labels_path, 'wb') as fid:
    np.save(fid, fused_node_labels)


def main(unused_argv):
  paper_pca_features = _read_paper_pca_features()
  # Find neighbors.
  annoy_index = build_annoy_index(paper_pca_features)
  save_annoy_index(annoy_index)
  neighbor_indices, neighbor_distances = compute_neighbor_indices_and_distances(
      paper_pca_features)
  del paper_pca_features
  _write_neighbors(neighbor_indices, neighbor_distances)

  data = _read_adjacency_indices()
  paper_paper_csr = data['paper_paper_index']
  paper_label = data['paper_label']
  train_indices = data['train_indices']
  valid_indices = data['valid_indices']
  test_indices = data['test_indices']
  del data

  fused_paper_adjacency_matrix = data_utils.generate_fused_paper_adjacency_matrix(
      neighbor_indices, neighbor_distances, paper_paper_csr)
  _write_fused_edges(fused_paper_adjacency_matrix)
  del fused_paper_adjacency_matrix
  del paper_paper_csr

  fused_node_labels = data_utils.generate_fused_node_labels(
      neighbor_indices, neighbor_distances, paper_label, train_indices,
      valid_indices, test_indices)
  _write_fused_nodes(fused_node_labels)


if __name__ == '__main__':
  flags.mark_flag_as_required('data_root')
  app.run(main)
