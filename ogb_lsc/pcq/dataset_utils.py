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

"""Dataset utilities."""

import functools
from typing import List, Optional

import jax
import jraph
from ml_collections import config_dict
import numpy as np
from ogb import utils
from ogb.utils import features
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tree

# pylint: disable=g-bad-import-order
# pytype: disable=import-error
import batching_utils
import conformer_utils
import datasets


curry = lambda f: functools.partial(functools.partial, f)


def build_dataset_iterator(
    data_root: str,
    split: str,
    dynamic_batch_size_config: config_dict.ConfigDict,
    sample_random: float,
    cached_conformers_file: str,
    debug: bool = False,
    is_training: bool = True,
    augment_with_random_mirror_symmetry: bool = False,
    positions_noise_std: Optional[float] = None,
    k_fold_split_id: Optional[int] = None,
    num_k_fold_splits: Optional[int] = None,
    filter_in_or_out_samples_with_nans_in_conformers: Optional[str] = None,
):
  """Returns an iterator over Batches from the dataset."""
  if debug:
    max_items_to_read_from_dataset = 10
    prefetch_buffer_size = 1
    shuffle_buffer_size = 1
  else:
    max_items_to_read_from_dataset = -1  # < 0 means no limit.
    prefetch_buffer_size = 64
    # It can take a while to fill the shuffle buffer with k fold splits.
    shuffle_buffer_size = 128 if k_fold_split_id is None else int(1e6)

  num_local_devices = jax.local_device_count()

  # Load all smile strings.
  indices, smiles, labels = _load_smiles(
      data_root,
      split,
      k_fold_split_id=k_fold_split_id,
      num_k_fold_splits=num_k_fold_splits)
  if debug:
    indices = indices[:100]
    smiles = smiles[:100]
    labels = labels[:100]
  # Generate all conformer features from smile strings ahead of time.
  # This gives us a boost from multi-parallelism as opposed to doing it
  # online.
  conformers = _load_conformers(indices, smiles, cached_conformers_file)

  data_generator = (
      lambda: _get_pcq_graph_generator(indices, smiles, labels, conformers))
  # Create a dataset yielding graphs from smile strings.
  example = next(data_generator())
  signature_from_example = tree.map_structure(_numpy_to_tensor_spec, example)
  ds = tf.data.Dataset.from_generator(
      data_generator, output_signature=signature_from_example)

  ds = ds.take(max_items_to_read_from_dataset)
  ds = ds.cache()
  if is_training:
    ds = ds.shuffle(shuffle_buffer_size)

  # Apply transformations.
  def map_fn(graph, conformer_positions):
    graph = _maybe_one_hot_atoms_with_noise(
        graph, is_training=is_training, sample_random=sample_random)
    # Add conformer features.
    graph = _add_conformer_features(
        graph,
        conformer_positions,
        augment_with_random_mirror_symmetry=augment_with_random_mirror_symmetry,
        noise_std=positions_noise_std,
        is_training=is_training,
    )
    return _downcast_ints(graph)

  ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
  if filter_in_or_out_samples_with_nans_in_conformers:
    if filter_in_or_out_samples_with_nans_in_conformers not in ("in", "out"):
      raise ValueError(
          "Unknown value specified for the argument "
          "`filter_in_or_out_samples_with_nans_in_conformers`: %s" %
          filter_in_or_out_samples_with_nans_in_conformers)

    filter_fn = _get_conformer_filter(
        with_nans=(filter_in_or_out_samples_with_nans_in_conformers == "in"))
    ds = ds.filter(filter_fn)

  if is_training:
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.repeat()

  ds = ds.prefetch(prefetch_buffer_size)
  it = tfds.as_numpy(ds)

  # Dynamic batching.
  batched_gen = batching_utils.dynamically_batch(
      it,
      n_node=dynamic_batch_size_config.n_node + 1,
      n_edge=dynamic_batch_size_config.n_edge,
      n_graph=dynamic_batch_size_config.n_graph + 1,
  )

  if is_training:
    # Stack `num_local_devices` of batches together for pmap updates.
    batch_size = num_local_devices

    def _batch(l):
      assert l
      return tree.map_structure(lambda *l: np.stack(l, axis=0), *l)

    def batcher_fn():
      batch = []
      for sample in batched_gen:
        batch.append(sample)
        if len(batch) == batch_size:
          yield _batch(batch)
          batch = []
      if batch:
        yield _batch(batch)

    for sample in batcher_fn():
      yield sample
  else:
    for sample in batched_gen:
      yield sample


def _get_conformer_filter(with_nans: bool):
  """Selects a conformer filter to apply.

  Args:
    with_nans: Filter only selects samples with NaNs in conformer features.
    Else, selects samples without any NaNs in conformer features.

  Returns:
    A function that can be used with tf.data.Dataset.filter().

  Raises:
    ValueError:
      If the input graph to the filter has no conformer features to filter.
  """

  def _filter(graph: jraph.GraphsTuple) -> tf.Tensor:

    if ("positions" not in graph.nodes) or (
        "positions_targets" not in graph.nodes) or (
            "positions_nan_mask" not in graph.globals):
      raise ValueError("Conformer features not available to filter.")

    any_nan = tf.logical_not(tf.squeeze(graph.globals["positions_nan_mask"]))
    return any_nan if with_nans else tf.logical_not(any_nan)

  return _filter


def _numpy_to_tensor_spec(arr: np.ndarray) -> tf.TensorSpec:
  if not isinstance(arr, np.ndarray):
    return tf.TensorSpec([],
                         dtype=tf.int32 if isinstance(arr, int) else tf.float32)
  elif arr.shape:
    return tf.TensorSpec((None,) + arr.shape[1:], arr.dtype)
  else:
    return tf.TensorSpec([], arr.dtype)


def _sample_uniform_categorical(num: int, size: int) -> tf.Tensor:
  return tf.random.categorical(tf.math.log([[1 / size] * size]), num)[0]


@curry(jax.tree_map)
def _downcast_ints(x):
  if x.dtype == tf.int64:
    return tf.cast(x, tf.int32)
  return x


def _one_hot_atoms(atoms: tf.Tensor) -> tf.Tensor:
  vocab_sizes = features.get_atom_feature_dims()
  one_hots = []
  for i in range(atoms.shape[1]):
    one_hots.append(tf.one_hot(atoms[:, i], vocab_sizes[i], dtype=tf.float32))
  return tf.concat(one_hots, axis=-1)


def _sample_one_hot_atoms(atoms: tf.Tensor) -> tf.Tensor:
  vocab_sizes = features.get_atom_feature_dims()
  one_hots = []
  num_atoms = tf.shape(atoms)[0]
  for i in range(atoms.shape[1]):
    sampled_category = _sample_uniform_categorical(num_atoms, vocab_sizes[i])
    one_hots.append(
        tf.one_hot(sampled_category, vocab_sizes[i], dtype=tf.float32))
  return tf.concat(one_hots, axis=-1)


def _one_hot_bonds(bonds: tf.Tensor) -> tf.Tensor:
  vocab_sizes = features.get_bond_feature_dims()
  one_hots = []
  for i in range(bonds.shape[1]):
    one_hots.append(tf.one_hot(bonds[:, i], vocab_sizes[i], dtype=tf.float32))
  return tf.concat(one_hots, axis=-1)


def _sample_one_hot_bonds(bonds: tf.Tensor) -> tf.Tensor:
  vocab_sizes = features.get_bond_feature_dims()
  one_hots = []
  num_bonds = tf.shape(bonds)[0]
  for i in range(bonds.shape[1]):
    sampled_category = _sample_uniform_categorical(num_bonds, vocab_sizes[i])
    one_hots.append(
        tf.one_hot(sampled_category, vocab_sizes[i], dtype=tf.float32))
  return tf.concat(one_hots, axis=-1)


def _maybe_one_hot_atoms_with_noise(
    x,
    is_training: bool,
    sample_random: float,
):
  """One hot atoms with noise."""
  gt_nodes = _one_hot_atoms(x.nodes)
  gt_edges = _one_hot_bonds(x.edges)
  if is_training:
    num_nodes = tf.shape(x.nodes)[0]
    sample_node_or_not = tf.random.uniform([num_nodes],
                                           maxval=1) < sample_random
    nodes = tf.where(
        tf.expand_dims(sample_node_or_not, axis=-1),
        _sample_one_hot_atoms(x.nodes), gt_nodes)
    num_edges = tf.shape(x.edges)[0]
    sample_edges_or_not = tf.random.uniform([num_edges],
                                            maxval=1) < sample_random
    edges = tf.where(
        tf.expand_dims(sample_edges_or_not, axis=-1),
        _sample_one_hot_bonds(x.edges), gt_edges)
  else:
    nodes = gt_nodes
    edges = gt_edges
  return x._replace(
      nodes={
          "atom_one_hots_targets": gt_nodes,
          "atom_one_hots": nodes,
      },
      edges={
          "bond_one_hots_targets": gt_edges,
          "bond_one_hots": edges
      })


def _load_smiles(
    data_root: str,
    split: str,
    k_fold_split_id: int,
    num_k_fold_splits: int,
):
  """Loads smiles trings for the input split."""

  if split == "test" or k_fold_split_id is None:
    indices = datasets.load_splits()[split]
  elif split == "train":
    indices = datasets.load_all_except_kth_fold_indices(
        data_root, k_fold_split_id, num_k_fold_splits)
    indices += datasets.load_splits()["train"]
  else:
    assert split == "valid"
    indices = datasets.load_kth_fold_indices(data_root, k_fold_split_id)

  smiles_and_labels = datasets.load_smile_strings(with_labels=True)
  smiles, labels = list(zip(*smiles_and_labels))
  return indices, [smiles[i] for i in indices], [labels[i] for i in indices]


def _convert_ogb_graph_to_graphs_tuple(ogb_graph):
  """Converts an OGB Graph to a GraphsTuple."""
  senders = ogb_graph["edge_index"][0]
  receivers = ogb_graph["edge_index"][1]
  edges = ogb_graph["edge_feat"]
  nodes = ogb_graph["node_feat"]
  n_node = np.array([ogb_graph["num_nodes"]])
  n_edge = np.array([len(senders)])
  graph = jraph.GraphsTuple(
      nodes=nodes,
      edges=edges,
      senders=senders,
      receivers=receivers,
      n_node=n_node,
      n_edge=n_edge,
      globals=None)
  return tree.map_structure(lambda x: x if x is not None else np.array(0.),
                            graph)


def _load_conformers(indices: List[int],
                     smiles: List[str],
                     cached_conformers_file: str):
  """Loads conformers."""
  smile_to_conformer = datasets.load_cached_conformers(cached_conformers_file)
  conformers = []
  for graph_idx, smile in zip(indices, smiles):
    del graph_idx  # Unused.
    if smile not in smile_to_conformer:
      raise KeyError("Cache did not have conformer entry for the smile %s" %
                     str(smile))
    conformers.append(dict(conformer=smile_to_conformer[smile]))
  return conformers


def _add_conformer_features(
    graph,
    conformer_features,
    augment_with_random_mirror_symmetry: bool,
    noise_std: float,
    is_training: bool,
):
  """Adds conformer features."""
  if not isinstance(graph.nodes, dict):
    raise ValueError("Expected a dict type for `graph.nodes`.")
  # Remove mean position to center around a canonical origin.
  positions = conformer_features["conformer"]
  # NaN's appear in ~0.13% of training, 0.104% of validation and 0.16% of test
  # nodes.
  # See this colab: http://shortn/_6UcuosxY7x.
  nan_mask = tf.reduce_any(tf.math.is_nan(positions))

  positions = tf.where(nan_mask, tf.constant(0., positions.dtype), positions)
  positions -= tf.reduce_mean(positions, axis=0, keepdims=True)

  # Optionally augment with a random rotation.
  if is_training:
    rot_mat = conformer_utils.get_random_rotation_matrix(
        augment_with_random_mirror_symmetry)
    positions = conformer_utils.rotate(positions, rot_mat)
  positions_targets = positions

  # Optionally add noise to the positions.
  if noise_std and is_training:
    positions = tf.random.normal(tf.shape(positions), positions, noise_std)

  return graph._replace(
      nodes=dict(
          positions=positions,
          positions_targets=positions_targets,
          **graph.nodes),
      globals={
          "positions_nan_mask":
              tf.expand_dims(tf.logical_not(nan_mask), axis=0),
          **(graph.globals if isinstance(graph.globals, dict) else {})
      })


def _get_pcq_graph_generator(indices, smiles, labels, conformers):
  """Returns a generator to yield graph."""
  for idx, smile, conformer_positions, label in zip(indices, smiles, conformers,
                                                    labels):
    graph = utils.smiles2graph(smile)
    graph = _convert_ogb_graph_to_graphs_tuple(graph)
    graph = graph._replace(
        globals={
            "target": np.array([label], dtype=np.float32),
            "graph_index": np.array([idx], dtype=np.int32),
            **(graph.globals if isinstance(graph.globals, dict) else {})
        })
    yield graph, conformer_positions
