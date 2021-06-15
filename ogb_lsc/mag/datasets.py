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

"""MAG240M-LSC datasets."""

import threading
from typing import NamedTuple, Optional


import jax
import jraph
from ml_collections import config_dict
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# pylint: disable=g-bad-import-order
# pytype: disable=import-error
import batching_utils
import data_utils


# We only want to load these arrays once for all threads.
# `get_arrays` uses an LRU cache which is not thread safe.
LOADING_RAW_ARRAYS_LOCK = threading.Lock()

NUM_CLASSES = data_utils.NUM_CLASSES


_MAX_DEPTH_IN_SUBGRAPH = 3


class Batch(NamedTuple):
  """NamedTuple to represent batches of data."""
  graph: jraph.GraphsTuple
  node_labels: np.ndarray
  label_mask: np.ndarray
  central_node_mask: np.ndarray
  node_indices: np.ndarray
  absolute_node_indices: np.ndarray


def build_dataset_iterator(
    data_root: str,
    split: str,
    dynamic_batch_size_config: config_dict.ConfigDict,
    online_subsampling_kwargs: dict,  # pylint: disable=g-bare-generic
    debug: bool = False,
    is_training: bool = True,
    k_fold_split_id: Optional[int] = None,
    ratio_unlabeled_data_to_labeled_data: float = 0.0,
    use_all_labels_when_not_training: bool = False,
    use_dummy_adjacencies: bool = False,
):
  """Returns an iterator over Batches from the dataset."""

  if split == 'test':
    use_all_labels_when_not_training = True

  if not is_training:
    ratio_unlabeled_data_to_labeled_data = 0.0

  # Load the master data arrays.
  with LOADING_RAW_ARRAYS_LOCK:
    array_dict = data_utils.get_arrays(
        data_root, k_fold_split_id=k_fold_split_id,
        use_dummy_adjacencies=use_dummy_adjacencies)

  node_labels = array_dict['paper_label'].reshape(-1)
  train_indices = array_dict['train_indices'].astype(np.int32)
  is_train_index = np.zeros(node_labels.shape[0], dtype=np.int32)
  is_train_index[train_indices] = 1
  valid_indices = array_dict['valid_indices'].astype(np.int32)
  is_valid_index = np.zeros(node_labels.shape[0], dtype=np.int32)
  is_valid_index[valid_indices] = 1
  is_train_or_valid_index = is_train_index + is_valid_index

  def sstable_to_intermediate_graph(graph):
    indices = tf.cast(graph.nodes['index'], tf.int32)
    first_index = indices[..., 0]

    # Add an additional absolute index, but adding offsets to authors, and
    # institution indices.
    absolute_index = graph.nodes['index']
    is_author = graph.nodes['type'] == 1
    absolute_index = tf.where(
        is_author, absolute_index + data_utils.NUM_PAPERS, absolute_index)
    is_institution = graph.nodes['type'] == 2
    absolute_index = tf.where(
        is_institution,
        absolute_index + data_utils.NUM_PAPERS + data_utils.NUM_AUTHORS,
        absolute_index)

    is_same_as_central_node = tf.math.equal(indices, first_index)
    input_nodes = graph.nodes
    graph = graph._replace(
        nodes={
            'one_hot_type':
                tf.one_hot(tf.cast(input_nodes['type'], tf.int32), 3),
            'one_hot_depth':
                tf.one_hot(
                    tf.cast(input_nodes['depth'], tf.int32),
                    _MAX_DEPTH_IN_SUBGRAPH),
            'year':
                tf.expand_dims(input_nodes['year'], axis=-1),
            'label':
                tf.one_hot(
                    tf.cast(input_nodes['label'], tf.int32),
                    NUM_CLASSES),
            'is_same_as_central_node':
                is_same_as_central_node,
            # Only first node in graph has a valid label.
            'is_central_node':
                tf.one_hot(0,
                           tf.shape(input_nodes['label'])[0]),
            'index':
                input_nodes['index'],
            'absolute_index': absolute_index,
        },
        globals=tf.expand_dims(graph.globals, axis=-1),
    )

    return graph

  ds = data_utils.get_graph_subsampling_dataset(
      split,
      array_dict,
      shuffle_indices=is_training,
      ratio_unlabeled_data_to_labeled_data=ratio_unlabeled_data_to_labeled_data,
      max_nodes=dynamic_batch_size_config.n_node - 1,  # Keep space for pads.
      max_edges=dynamic_batch_size_config.n_edge,
      **online_subsampling_kwargs)
  if debug:
    ds = ds.take(50)
  ds = ds.map(
      sstable_to_intermediate_graph,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if is_training:
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.shuffle(buffer_size=1 if debug else 128)
    ds = ds.repeat()
  ds = ds.prefetch(1 if debug else tf.data.experimental.AUTOTUNE)
  np_ds = iter(tfds.as_numpy(ds))
  batched_np_ds = batching_utils.dynamically_batch(
      np_ds,
      **dynamic_batch_size_config,
  )

  def intermediate_graph_to_batch(graph):
    central_node_mask = graph.nodes['is_central_node']
    label = graph.nodes['label']
    node_indices = graph.nodes['index']
    absolute_indices = graph.nodes['absolute_index']

    ### Construct label as a feature for non-central nodes.
    # First do a lookup with node indices, with a np.minimum to ensure we do not
    # index out of bounds due to num_authors being larger than num_papers.
    is_same_as_central_node = graph.nodes['is_same_as_central_node']
    capped_indices = np.minimum(node_indices, node_labels.shape[0] - 1)
    label_as_feature = node_labels[capped_indices]
    # Nodes which are not in train set should get `num_classes` label.
    # Nodes in test set or non-arXiv nodes have -1 or nan labels.

    # Mask out invalid labels and non-papers.
    use_label_as_feature = np.logical_and(label_as_feature >= 0,
                                          graph.nodes['one_hot_type'][..., 0])
    if split == 'train' or not use_all_labels_when_not_training:
      # Mask out validation papers and non-arxiv papers who
      # got labels from fusing with arxiv papers.
      use_label_as_feature = np.logical_and(is_train_index[capped_indices],
                                            use_label_as_feature)
    label_as_feature = np.where(use_label_as_feature, label_as_feature,
                                NUM_CLASSES)
    # Mask out central node label in case it appears again.
    label_as_feature = np.where(is_same_as_central_node, NUM_CLASSES,
                                label_as_feature)
    # Nodes which are not papers get `NUM_CLASSES+1` label.
    label_as_feature = np.where(graph.nodes['one_hot_type'][..., 0],
                                label_as_feature, NUM_CLASSES+1)

    nodes = {
        'label_as_feature': label_as_feature,
        'year': graph.nodes['year'],
        'bitstring_year': _get_bitstring_year_representation(
            graph.nodes['year']),
        'one_hot_type': graph.nodes['one_hot_type'],
        'one_hot_depth': graph.nodes['one_hot_depth'],
    }

    graph = graph._replace(
        nodes=nodes,
        globals={},
    )
    is_train_or_valid_node = np.logical_and(
        is_train_or_valid_index[capped_indices],
        graph.nodes['one_hot_type'][..., 0])
    if is_training:
      label_mask = np.logical_and(central_node_mask, is_train_or_valid_node)
    else:
      # `label_mask` is used to index into valid central nodes by prediction
      # calculator. Since that computation is only done when not training, and
      # at that time we are guaranteed all central nodes have valid labels,
      # we just set label_mask = central_node_mask when not training.
      label_mask = central_node_mask
    batch = Batch(
        graph=graph,
        node_labels=label,
        central_node_mask=central_node_mask,
        label_mask=label_mask,
        node_indices=node_indices,
        absolute_node_indices=absolute_indices)

    # Transform integers into one-hots.
    batch = _add_one_hot_features_to_batch(batch)

    # Gather PCA features.
    return _add_embeddings_to_batch(batch, array_dict['bert_pca_129'])

  batch_list = []
  for batch in batched_np_ds:
    with jax.profiler.StepTraceAnnotation('batch_postprocessing'):
      batch = intermediate_graph_to_batch(batch)
    if is_training:
      batch_list.append(batch)
      if len(batch_list) == jax.local_device_count():
        yield jax.device_put_sharded(batch_list, jax.local_devices())
        batch_list = []
    else:
      yield batch


def _get_bitstring_year_representation(year: np.ndarray):
  """Return year as bitstring."""
  min_year = 1900
  max_training_year = 2018
  offseted_year = np.minimum(year, max_training_year) - min_year
  return np.unpackbits(offseted_year.astype(np.uint8), axis=-1)


def _np_one_hot(targets: np.ndarray, nb_classes: int):
  res = np.zeros(targets.shape + (nb_classes,), dtype=np.float16)
  np.put_along_axis(res, targets.astype(np.int32)[..., None], 1.0, axis=-1)
  return res


def _get_one_hot_year_representation(
    year: np.ndarray,
    one_hot_type: np.ndarray,
):
  """Returns good representation for year."""
  # Bucket edges found based on quantiles to bucket into 20 equal sized buckets.
  bucket_edges = np.array([
      1964, 1975, 1983, 1989, 1994, 1998, 2001, 2004,
      2006, 2008, 2009, 2011, 2012, 2013, 2014, 2016,
      2017,  # 2018, 2019, 2020 contain last-year-of-train, eval, test nodes
  ])
  year = np.squeeze(year, axis=-1)
  year_id = np.searchsorted(bucket_edges, year)
  is_paper = one_hot_type[..., 0]
  bucket_id_for_non_paper = len(bucket_edges) + 1
  bucket_id = np.where(is_paper, year_id, bucket_id_for_non_paper)
  one_hot_year = _np_one_hot(bucket_id, len(bucket_edges) + 2)
  return one_hot_year


def _add_one_hot_features_to_batch(batch: Batch) -> Batch:
  """Transforms integer features into one-hot features."""
  nodes = batch.graph.nodes.copy()
  nodes['one_hot_year'] = _get_one_hot_year_representation(
      nodes['year'], nodes['one_hot_type'])
  del nodes['year']

  # NUM_CLASSES plus one category for papers for which a class is not provided
  # and another for nodes that are not papers.
  nodes['one_hot_label_as_feature'] = _np_one_hot(
      nodes['label_as_feature'], NUM_CLASSES + 2)
  del nodes['label_as_feature']
  return batch._replace(graph=batch.graph._replace(nodes=nodes))


def _add_embeddings_to_batch(batch: Batch, embeddings: np.ndarray) -> Batch:
  nodes = batch.graph.nodes.copy()
  nodes['features'] = embeddings[batch.absolute_node_indices]
  graph = batch.graph._replace(nodes=nodes)
  return batch._replace(graph=graph)
