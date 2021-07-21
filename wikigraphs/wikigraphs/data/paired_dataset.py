# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# WikiGraphs is licensed under the terms of the Creative Commons
# Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.
#
# WikiText-103 data (unchanged) is licensed by Salesforce.com, Inc. under the
# terms of the Creative Commons Attribution-ShareAlike 4.0 International
# (CC BY-SA 4.0) license. You can find details about CC BY-SA 4.0 at:
#
#     https://creativecommons.org/licenses/by-sa/4.0/legalcode
#
# Freebase data is licensed by Google LLC under the terms of the Creative
# Commons CC BY 4.0 license. You may obtain a copy of the License at:
#
#     https://creativecommons.org/licenses/by/4.0/legalcode
#
# ==============================================================================
"""Tools for accessing the graph-text paired datasets."""

import abc
import collections
from typing import List, Tuple, NamedTuple, Any, Dict, Optional, Union

from absl import logging
import jax.numpy as jnp
import jraph
import numpy as np

from wikigraphs.data import dataset
from wikigraphs.data import io_tools
from wikigraphs.data import tokenizers
from wikigraphs.data import tools


ArrayType = Union[np.ndarray, jnp.ndarray]
DATA_ROOT = '/tmp/data/wikigraphs'


class RawDataset(dataset.Dataset):
  """The untokenized raw dataset."""

  def __init__(self,
               subset: str = 'train',
               shuffle_data: bool = False,
               data_dir: str = None,
               version: str = 'max256'):
    """Constructor.

    Args:
      subset: which subset to load.
      shuffle_data: set to True to randomly shuffle the data.
      data_dir: if provided this will be used instead of the default location to
        look for data, it must contain files like `train.gz`, `valid.gz` and
        `test.gz`.
      version: which version of the data to load, this must be the name of a
        directory in `DATA_ROOT`.
    """
    super().__init__()
    self._subset = subset
    self._shuffle_data = shuffle_data
    self._data_dir = data_dir or DATA_ROOT
    self._dataset = None

    allowed_versions = ('max256', 'max512', 'max1024')
    if version not in allowed_versions:
      raise ValueError(f'Version {version} not one of the allowed versions:'
                       f' {allowed_versions}.')

    self._version = version

  def _load_data(self):
    """Load and prepare the data iterator."""
    if self._dataset is None:
      self._dataset = list(io_tools.read_pairs_from_gzip_txt_file(
          f'{self._data_dir}/{self._version}/{self._subset}.gz'))

    def source():
      n_pairs = len(self._dataset)
      if self._shuffle_data:
        idx = np.random.permutation(n_pairs)
      else:
        idx = np.arange(n_pairs)
      for i in range(n_pairs):
        yield self._dataset[idx[i]]

    return source()


class Graph:
  """A convenience class for representing graphs."""

  def __init__(self, nodes: List[str], edges: List[Tuple[int, int, str]]):
    """Construct a graph from a list of nodes and edges.

    Args:
      nodes: a list of node attributes, one for each node.
      edges: a list of (source_node_id, target_node_id, edge_attribute) for each
        edge.
    """
    self._nodes = nodes
    self._edges = edges
    self._node2id = {n: i for i, n in enumerate(nodes)}

  def nodes(self) -> List[str]:
    return self._nodes

  def edges(self) -> List[Tuple[int, int, str]]:
    return self._edges

  def node2id(self, node: str) -> int:
    return self._node2id[node]

  @classmethod
  def from_edges(cls, edges: List[str]) -> 'Graph':
    """Build a graph instance from a list of edges."""
    node2id = dict()
    parsed_edges = []
    next_node_id = 0

    for e in edges:
      src, edge, tgt = e.split('\t')[:3]
      src_id = node2id.get(src, next_node_id)
      if src_id == next_node_id:
        node2id[src] = src_id
        next_node_id += 1
      tgt_id = node2id.get(tgt, next_node_id)
      if tgt_id == next_node_id:
        node2id[tgt] = tgt_id
        next_node_id += 1
      parsed_edges.append((src_id, tgt_id, edge))

    id2node = {i: n for n, i in node2id.items()}
    return Graph(nodes=[id2node[i] for i in range(next_node_id)],
                 edges=parsed_edges)

  def to_edges(self) -> List[str]:
    r"""Convert graph to a list of edges.

    The converted list of edges should be compatible with the format specified
    in io_tools and compatible with the `from_edges` method above.

    Returns:
      edges: one edge per line, with the (source, target, edge_type) separated
        by `\t`.
    """
    edges = []
    for s, t, e in self._edges:
      edges.append(f'{self._nodes[s]}\t{e}\t{self._nodes[t]}')
    return edges

  @classmethod
  def subsample_nodes(
      cls, graph: 'Graph', subsample_rate: float = 1.0, center_node: str = None
      ) -> 'Graph':
    """Subsample the nodes of a graph."""
    graph_size = len(graph.nodes())
    if subsample_rate == 1.0 or graph_size <= 1:
      return graph
    subsampled_nodes_id = np.arange(graph_size)
    if subsample_rate < 1.0:
      subsample_graph_size = int(subsample_rate * graph_size)
      if center_node is not None:
        # We need to keep the center node during subsampling
        center_node_id = graph.node2id(center_node)
        subsampled_nodes_id = subsampled_nodes_id[
            subsampled_nodes_id != center_node_id]
        subsample_graph_size = max(1, subsample_graph_size - 1)
        subsampled_nodes_id = np.random.choice(
            subsampled_nodes_id, subsample_graph_size, replace=False)
        subsampled_nodes_id = np.append(subsampled_nodes_id, center_node_id)
      else:
        subsampled_nodes_id = np.random.choice(
            subsampled_nodes_id, subsample_graph_size, replace=False)
      subsampled_nodes_id = np.sort(subsampled_nodes_id)
      map_subsampled_nodes_id = {
          old_id: new_id for new_id, old_id in enumerate(subsampled_nodes_id)}
    nodes = []
    edges = []
    for node_id, n in enumerate(graph.nodes()):
      if node_id in subsampled_nodes_id:
        nodes.append(n)
    for out_node, in_node, e in graph.edges():
      if out_node in subsampled_nodes_id and in_node in subsampled_nodes_id:
        edges.append((map_subsampled_nodes_id[out_node],
                      map_subsampled_nodes_id[in_node], e))
    return Graph(nodes=nodes, edges=edges)


class ParsedGraphTextPair(NamedTuple):
  """Graph-text pair with graph parsed into a `Graph` instance."""
  center_node: str
  title: str
  text: str
  graph: Graph


class ParsedDataset(dataset.Dataset):
  """Raw dataset + parsing graphs into Graph instances."""

  def __init__(self,
               subset: str = 'train',
               shuffle_data: bool = False,
               data_dir: str = None,
               version: str = 'max256'):
    """Constructor.

    Args:
      subset: which subset to load.
      shuffle_data: set to True to randomly shuffle the data.
      data_dir: if provided this will be used instead of the default location to
        look for data, it must contain files like `train.gz`, `valid.gz` and
        `test.gz`.
      version: which version of the data to load, this must be the name of a
        directory in `DATA_ROOT`.
    """
    super().__init__()
    self._raw_data = RawDataset(subset=subset, shuffle_data=False,
                                data_dir=data_dir, version=version)
    self._shuffle_data = shuffle_data
    self._dataset = None

  def _load_data(self):
    if self._dataset is None:
      # pylint: disable=g-complex-comprehension
      self._dataset = [ParsedGraphTextPair(center_node=pair.center_node,
                                           title=pair.title,
                                           text=pair.text,
                                           graph=Graph.from_edges(pair.edges))
                       for pair in self._raw_data]

    def source():
      n_pairs = len(self._dataset)
      if self._shuffle_data:
        idx = np.random.permutation(n_pairs)
      else:
        idx = np.arange(n_pairs)
      for i in range(n_pairs):
        yield self._dataset[idx[i]]

    return source()


class BaseGraph2TextDataset(dataset.Dataset):
  """Base dataset class for graph-to-text tasks."""

  def __init__(self,
               tokenizer: tokenizers.Tokenizer,
               graph_tokenizer: Optional[tokenizers.GraphTokenizer] = None,
               batch_size: int = 1,
               timesteps: int = 128,
               subset: str = 'train',
               shuffle_data: bool = False,
               repeat: bool = False,
               version: str = 'max256',
               data_dir: str = None,
               subsample_nodes: float = 1.0,
               graph_retrieval_dataset: bool = False,
               debug: bool = False):
    """Constructor.

    Args:
      tokenizer: the tokenizer for text data.
      graph_tokenizer: the tokenizer for graph data.
      batch_size: number of sequences to put in a batch.
      timesteps: number of tokens to put in a sequence in a batch.
      subset: which subset to load.
      shuffle_data: whether to shuffle data.
      repeat: set to True to repeat the dataset infinitely, otherwise do only
        one pass through the dataset.
      version: which version of the data to load.
      data_dir: if set load data instead from this directory, and ignore
        `version`.
      subsample_nodes: the proportion of the nodes in a graph to keep.
      graph_retrieval_dataset: whether to construct the dataset for graph
        retrieval tasks.
      debug: set to True to use debug mode and only load a small number of
        examples.
    """
    super().__init__()
    self._parsed_data = ParsedDataset(subset=subset,
                                      shuffle_data=False,
                                      data_dir=data_dir,
                                      version=version)
    self._tokenizer = tokenizer
    self._graph_tokenizer = graph_tokenizer
    self._batch_size = batch_size
    self._timesteps = timesteps
    self._subset = subset
    self._shuffle_data = shuffle_data
    self._repeat = repeat
    self._subsample_nodes = subsample_nodes
    self._graph_retrieval_dataset = graph_retrieval_dataset
    self._debug = debug

    self._dataset = None

  @property
  def num_articles(self):
    return self._num_articles

  @abc.abstractmethod
  def _process_graph(self, center_node: str, graph: Graph):
    """Process the graph part of a `ParsedGraphTextPair` instance."""

  def _process_graph_text_pair(
      self, pair: ParsedGraphTextPair) -> Tuple[Any, np.ndarray]:
    """Process the given graph-text pair and prepare one example.

    Args:
      pair: the input `ParsedGraphTextPair` instance.

    Returns:
      graph: the processed graph content.
      text: the tokenized text, a sequence of token IDs.
    """
    return (self._process_graph(pair.center_node, pair.graph),
            self._tokenizer.encode(
                pair.text, prepend_bos=True, append_eos=True))

  def _load_data(self):
    """Prepare the data."""
    if self._dataset is None:
      if self._debug:
        data = [next(self._parsed_data) for _ in range(10)]
      else:
        data = list(self._parsed_data)
      self._dataset = [self._process_graph_text_pair(p) for p in data]
      self._num_articles = len(self._dataset)
      logging.info('Loaded a total of %d examples from %s set.',
                   self._num_articles, self._subset)
      if self._graph_retrieval_dataset:
        # For graph retrieval tasks we pair all texts and graphs in the dataset,
        # and indicate their (text_id, graph_id)
        retrieval_data = []
        for i, (g1, _) in enumerate(self._dataset):
          for j, (_, t2) in enumerate(self._dataset):
            retrieval_data.append(((g1, t2), (i, j)))
        self._dataset = retrieval_data
        logging.info('Constructed %d pairs.', len(self._dataset))

    def source():
      n_examples = len(self._dataset)
      if self._shuffle_data:
        idx = np.random.permutation(n_examples)
      else:
        idx = np.arange(n_examples)
      for i in range(n_examples):
        yield self._dataset[idx[i]]

    def maybe_repeated_source():
      if self._repeat:
        while True:
          yield from source()
      else:
        yield from source()

    data_iter = tools.batch_graph_text_pairs(
        maybe_repeated_source(),
        self._batch_size,
        self._timesteps + 1,
        pad_value=self._tokenizer.pad_token(),
        seq_and_graph_id=self._graph_retrieval_dataset)

    if self._graph_retrieval_dataset:
      data_iter = map(lambda x: dict(  # pylint: disable=g-long-lambda
          obs=x['obs'][:, :-1],
          target=x['obs'][:, 1:],
          should_reset=x['should_reset'][:, :-1],
          # If target is a <pad> token then that target should not be predicted.
          mask=(x['obs'][:, 1:] != self._tokenizer.pad_token()).astype(
              np.float32),
          seq_id=x['seq_id'],
          graph_id=x['graph_id'],
          graphs=self._process_graph_batch(x['graphs']),
          ), data_iter)
    else:
      data_iter = map(lambda x: dict(  # pylint: disable=g-long-lambda
          obs=x['obs'][:, :-1],
          target=x['obs'][:, 1:],
          should_reset=x['should_reset'][:, :-1],
          # If target is a <pad> token then that target should not be predicted.
          mask=(x['obs'][:, 1:] != self._tokenizer.pad_token()).astype(
              np.float32),
          graphs=self._process_graph_batch(x['graphs']),
          ), data_iter)

    # Filter out batches that does not have targets.
    # This may happen when an observation contains a single last token of the
    # sequence, which was predicted as target in the previous batch, and only
    # used as observation in this batch, without a matching target.  In this
    # case all the masks are 0, therefore this batch provides no training signal
    # and we can safely remove this batch.  This also avoids some potential
    # downstream issues.
    data_iter = filter(lambda x: x['mask'].sum() > 0, data_iter)
    return data_iter

  @abc.abstractmethod
  def _process_graph_batch(self, graphs: List[Any]):
    """Process a batch of graph data.

    Args:
      graphs: a list of graph data, each as returned by `_process_graph`.

    Returns:
      processed_graphs: processed tensor(s) that can be directly fed into a
        model.
    """

  @abc.abstractmethod
  def return_faux_batch(self) -> Dict[str, np.ndarray]:
    """Return a fake batch with the right shapes and dtypes."""


class TextOnlyDataset(BaseGraph2TextDataset):
  """Text-only version of the paired dataset."""

  def __init__(self,
               tokenizer: tokenizers.Tokenizer,
               graph_tokenizer: Optional[tokenizers.GraphTokenizer] = None,
               batch_size: int = 1,
               timesteps: int = 128,
               subset: str = 'train',
               shuffle_data: bool = False,
               repeat: bool = False,
               version: str = 'max256',
               data_dir: str = None,
               debug: bool = False,
               **kwargs):
    """Constructor.

    Args:
      tokenizer: the tokenizer for text data.
      graph_tokenizer: not used, keeping it here for compatibility with other
        graph2text datasets.
      batch_size: number of sequences to put in a batch.
      timesteps: number of tokens to put in a sequence in a batch.
      subset: which subset to load.
      shuffle_data: whether to shuffle data.
      repeat: set to True to repeat the dataset infinitely, otherwise do only
        one pass through the dataset.
      version: which version of the data to load.
      data_dir: if set load data instead from this directory, and ignore
        `version`.
      debug: set to True to use debug mode and only load a small number of
        examples.
      **kwargs: other arguments (for interface compatibility).
    """
    del graph_tokenizer
    super().__init__(tokenizer=tokenizer,
                     graph_tokenizer=None,
                     batch_size=batch_size,
                     timesteps=timesteps,
                     subset=subset,
                     shuffle_data=shuffle_data,
                     repeat=repeat,
                     version=version,
                     data_dir=data_dir,
                     debug=debug)

  def _process_graph_batch(self, graphs: List[Any]):
    del graphs
    return None

  def _process_graph(self, center_node: str, graph: Graph):
    del center_node
    del graph
    return None

  def __next__(self):
    batch = super().__next__()
    # Data should be text-only.
    del batch['graphs']
    return batch

  def return_faux_batch(self):
    """Return a fake batch with the right shapes and types."""
    obs = np.zeros((self._batch_size, self._timesteps), dtype=np.int32)
    target = np.zeros_like(obs)
    should_reset = np.zeros_like(obs, dtype=np.float32)
    mask = np.zeros_like(obs, dtype=np.float32)
    return dict(obs=obs, target=target, should_reset=should_reset, mask=mask)


class Bow2TextDataset(BaseGraph2TextDataset):
  """Dataset for bag-of-words to text."""

  def _process_graph(self, center_node: str, graph: Graph):
    """Process the graph part of a `ParsedGraphTextPair` instance."""
    # We don't use center node in a bag-of-words representation
    del center_node
    if self._subsample_nodes < 1.0:
      graph = Graph.subsample_nodes(graph, self._subsample_nodes)

    bow = np.zeros(self._graph_tokenizer.vocab_size, dtype=np.int32)
    for n in graph.nodes():
      for t in self._graph_tokenizer.encode_node(n):
        bow[t] += 1
    for _, _, e in graph.edges():
      for t in self._graph_tokenizer.encode_edge(e):
        bow[t] += 1
    return bow

  def _process_graph_batch(self, graphs: List[Any]):
    """Process a batch of graph data.

    Args:
      graphs: a list of graph data, each as returned by `_process_graph`.

    Returns:
      processed_graphs: processed tensor(s) that can be directly fed into a
        model.
    """
    empty_graph_bow = np.zeros(self._graph_tokenizer.vocab_size, dtype=np.int32)
    graphs = [g if g is not None else empty_graph_bow for g in graphs]
    # B x [V] -> [B, V]
    return np.stack(graphs, axis=0)

  def return_faux_batch(self):
    obs = np.zeros((self._batch_size, self._timesteps), dtype=np.int32)
    target = np.zeros_like(obs)
    should_reset = np.zeros_like(obs, dtype=np.float32)
    mask = np.zeros_like(obs, dtype=np.float32)
    graphs = np.zeros((self._batch_size, self._graph_tokenizer.vocab_size),
                      dtype=np.float32)
    return dict(obs=obs, target=target, should_reset=should_reset, mask=mask,
                graphs=graphs)


class Graph2TextDataset(BaseGraph2TextDataset):
  """Graph-to-text dataset.

  This dataset encodes the graph nodes and edges using a bag-of-words
  representation.
  """

  def __init__(self,
               tokenizer: tokenizers.Tokenizer,
               graph_tokenizer: tokenizers.GraphTokenizer,
               batch_size: int = 1,
               timesteps: int = 128,
               subset: str = 'train',
               shuffle_data: bool = False,
               repeat: bool = False,
               version: str = 'max256',
               data_dir: str = None,
               subsample_nodes: float = 1.0,
               graph_retrieval_dataset: bool = False,
               debug: bool = False):
    """Constructor.

    Args:
      tokenizer: the tokenizer for text data.
      graph_tokenizer: the tokenizer for graph data.
      batch_size: number of sequences to put in a batch.
      timesteps: number of tokens to put in a sequence in a batch.
      subset: which subset to load.
      shuffle_data: whether to shuffle data.
      repeat: set to True to repeat the dataset infinitely, otherwise do only
        one pass through the dataset.
      version: which version of the data to load.
      data_dir: if set load data instead from this directory, and ignore
        `version`.
      subsample_nodes: the proportion of the nodes in a graph to keep.
      graph_retrieval_dataset: whether to construct the dataset for graph
        retrieval tasks.
      debug: set to True to use debug mode and only load a small number of
        examples.
    """
    self._graph_feature_dim = graph_tokenizer.vocab_size
    super().__init__(tokenizer=tokenizer,
                     graph_tokenizer=graph_tokenizer,
                     batch_size=batch_size,
                     timesteps=timesteps,
                     subset=subset,
                     shuffle_data=shuffle_data,
                     repeat=repeat,
                     version=version,
                     data_dir=data_dir,
                     subsample_nodes=subsample_nodes,
                     graph_retrieval_dataset=graph_retrieval_dataset,
                     debug=debug)
    self._placeholder_graph = self._process_graph(
        center_node='<pad>',
        graph=Graph(nodes=['<pad>'], edges=[]))

  def _process_graph(self, center_node: str, graph: Graph):
    """Process the graph part of a `ParsedGraphTextPair` instance."""
    if self._subsample_nodes < 1.0:
      graph = Graph.subsample_nodes(graph, self._subsample_nodes, center_node)

    nodes = graph.nodes()
    edges = graph.edges()
    n_edges = len(edges)

    sender = np.zeros(n_edges, dtype=np.int32)
    receiver = np.zeros(n_edges, dtype=np.int32)

    nodes_bow = []
    edges_bow = []

    for n in nodes:
      bow = collections.defaultdict(int)
      for t in self._graph_tokenizer.encode_node(n):
        bow[t] += 1
      nodes_bow.append(bow)
    for i, (s, r, e) in enumerate(edges):
      bow = collections.defaultdict(int)
      for t in self._graph_tokenizer.encode_edge(e):
        bow[t] += 1
      edges_bow.append(bow)
      sender[i] = s
      receiver[i] = r

    return (nodes_bow, edges_bow, sender, receiver, graph.node2id(center_node))

  def _to_graph_with_features(
      self, nodes_bow, edges_bow, sender, receiver, center_node_id):
    """Convert the input to a `jraph.GraphsTuple` instance."""
    n_nodes = len(nodes_bow)
    n_edges = len(edges_bow)

    # +1 for the center node indicator
    nodes = np.zeros((n_nodes, self._graph_feature_dim + 1), dtype=np.float32)
    edges = np.zeros((n_edges, self._graph_feature_dim), dtype=np.float32)

    nodes[center_node_id][-1] = 1
    for i, bow in enumerate(nodes_bow):
      for t, c in bow.items():
        nodes[i][t] = c
    for i, bow in enumerate(edges_bow):
      for t, c in bow.items():
        edges[i][t] = c

    return jraph.GraphsTuple(
        nodes=nodes, edges=edges, senders=sender, receivers=receiver,
        globals=None, n_node=np.array([n_nodes], dtype=np.int32),
        n_edge=np.array([n_edges], dtype=np.int32))

  def _process_graph_batch(self, graphs: List[Any]):
    """Process a batch of graph data.

    Args:
      graphs: a list of graph data, each as returned by `_process_graph`.

    Returns:
      processed_graphs: a list of processed tensor(s).
    """
    graphs = [g if g is not None else self._placeholder_graph for g in graphs]
    return [self._to_graph_with_features(*g) for g in graphs]

  def return_faux_batch(self) -> Dict[str, np.ndarray]:
    """Return a fake batch with the right shapes and dimensions."""
    obs = np.zeros([self._batch_size, self._timesteps], dtype=np.int32)
    target = np.zeros([self._batch_size, self._timesteps], dtype=np.int32)
    should_reset = np.zeros_like(obs, np.float32)
    mask = np.zeros_like(obs, np.float32)
    # A batch should contain `batch_size` graphs.  Here we make sure each graph
    # has one node and one edge.
    graphs = self._batch_size * [jraph.GraphsTuple(
        nodes=np.zeros([1, self._graph_feature_dim + 1], dtype=np.float32),
        edges=np.zeros([1, self._graph_feature_dim], dtype=np.float32),
        senders=np.zeros([1], dtype=np.int32),
        receivers=np.zeros([1], dtype=np.int32),
        n_node=np.ones(1, dtype=np.int32),
        n_edge=np.ones(1, dtype=np.int32),
        globals=None)]
    return dict(obs=obs, target=target, mask=mask, should_reset=should_reset,
                graphs=graphs)
