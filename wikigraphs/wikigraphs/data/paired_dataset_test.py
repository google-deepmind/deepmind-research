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
"""Tests for wikigraphs.data.paired_dataset."""

from absl.testing import absltest
import jraph
from wikigraphs.data import io_tools
from wikigraphs.data import paired_dataset
from wikigraphs.data import tokenizers
from wikigraphs.data import wikitext


WIKITEXT_ROOT = '/tmp/data/wikitext-103'
WIKIGRAPHS_ROOT = '/tmp/data/wikigraphs'
WIKITEXT_VOCAB_FILE = '/tmp/data/wikitext-vocab.csv'
GRAPH_VOCAB_FILE = '/tmp/data/graph-vocab.csv'


class PairedDatasetTest(absltest.TestCase):

  def test_raw_paired_dataset_size(self):
    dataset = paired_dataset.RawDataset(
        subset='valid', shuffle_data=False, data_dir=WIKIGRAPHS_ROOT)
    pairs = list(dataset)
    self.assertLen(pairs, 48)

    self.assertEqual(pairs[0].title, 'Homarus_gammarus')
    self.assertEqual(pairs[-1].title, 'Rakie_Ayola')

    # Make sure the content of the articles match the original
    wikitext_set = wikitext.RawDataset(
        subset='valid', shuffle_data=False, version='raw',
        data_dir=WIKITEXT_ROOT)
    title2article = {wikitext.normalize_title(a.title).replace(' ', ''): a.text
                     for a in wikitext_set}
    for p in pairs:
      title = io_tools.normalize_freebase_string(p.title).replace(' ', '')
      article = title2article.get(title, None)
      self.assertIsNotNone(article)
      self.assertEqual(article, p.text)

  def test_graph_from_edges(self):
    edges = ['A\tE1\tB',
             'A\tE2\tC',
             'B\tE1\tC',
             'C\tE3\tD',
             'C\tE2\tE']
    graph = paired_dataset.Graph.from_edges(edges)
    self.assertEqual(graph.nodes(), ['A', 'B', 'C', 'D', 'E'])
    self.assertEqual(graph.edges(), [(0, 1, 'E1'),
                                     (0, 2, 'E2'),
                                     (1, 2, 'E1'),
                                     (2, 3, 'E3'),
                                     (2, 4, 'E2')])

  def test_graph_to_edges(self):
    edges = ['A\tE1\tB',
             'A\tE2\tC',
             'B\tE1\tC',
             'C\tE3\tD',
             'C\tE2\tE']
    graph = paired_dataset.Graph.from_edges(edges)
    self.assertEqual(graph.to_edges(), edges)

  def test_bow2text_dataset(self):
    tokenizer = tokenizers.WordTokenizer(vocab_file=WIKITEXT_VOCAB_FILE)
    graph_tokenizer = tokenizers.GraphTokenizer(vocab_file=GRAPH_VOCAB_FILE)

    batch_size = 4
    seq_len = 256
    dataset = paired_dataset.Bow2TextDataset(
        tokenizer,
        graph_tokenizer,
        batch_size=batch_size,
        timesteps=seq_len,
        subset='valid',
        subsample_nodes=0.7,
        repeat=False,
        data_dir=WIKIGRAPHS_ROOT)

    num_tokens = 0
    for batch in dataset:
      num_tokens += batch['mask'].sum()
      self.assertEqual(batch['graphs'].shape,
                       (batch_size, graph_tokenizer.vocab_size))

    raw_dataset = paired_dataset.RawDataset(subset='valid', shuffle_data=False)
    raw_num_tokens = 0
    n_pairs = 0
    for pair in raw_dataset:
      raw_num_tokens += len(tokenizer.encode(
          pair.text, prepend_bos=True, append_eos=True))
      n_pairs += 1

    # The first token of each example is not counted by `mask` as it masks the
    # targets, and the first token of each example never appears in the targets.
    self.assertEqual(raw_num_tokens, num_tokens + n_pairs)

  def test_graph2text_dataset(self):
    tokenizer = tokenizers.WordTokenizer(vocab_file=WIKITEXT_VOCAB_FILE)
    graph_tokenizer = tokenizers.GraphTokenizer(vocab_file=GRAPH_VOCAB_FILE)

    batch_size = 4
    seq_len = 256
    dataset = paired_dataset.Graph2TextDataset(
        tokenizer,
        graph_tokenizer,
        batch_size=batch_size,
        timesteps=seq_len,
        subsample_nodes=0.8,
        subset='valid',
        data_dir=WIKIGRAPHS_ROOT)
    data_iter = iter(dataset)
    batch = next(data_iter)
    self.assertEqual(batch['obs'].shape, (batch_size, seq_len))
    self.assertEqual(batch['target'].shape, (batch_size, seq_len))
    self.assertEqual(batch['should_reset'].shape, (batch_size, seq_len))
    self.assertEqual(batch['mask'].shape, (batch_size, seq_len))
    self.assertIsInstance(batch['graphs'], list)
    self.assertLen(batch['graphs'], batch_size)
    for i in range(batch_size):
      self.assertIsInstance(batch['graphs'][i], jraph.GraphsTuple)

      # +1 for the center_node mask
      self.assertEqual(
          batch['graphs'][i].nodes.shape[-1], graph_tokenizer.vocab_size + 1)
      self.assertEqual(
          batch['graphs'][i].edges.shape[-1], graph_tokenizer.vocab_size)
      n_edges = batch['graphs'][i].n_edge
      self.assertEqual(batch['graphs'][i].senders.shape, (n_edges,))
      self.assertEqual(batch['graphs'][i].receivers.shape, (n_edges,))

    # Make sure the token count matches across the tokenized data and the raw
    # data set.
    num_tokens = 0
    for batch in dataset:
      num_tokens += batch['mask'].sum()

    raw_dataset = paired_dataset.RawDataset(subset='valid', shuffle_data=False)
    raw_num_tokens = 0
    n_pairs = 0
    for pair in raw_dataset:
      raw_num_tokens += len(tokenizer.encode(
          pair.text, prepend_bos=True, append_eos=True))
      n_pairs += 1

    # The first token of each example is not counted by `mask` as it masks the
    # targets, and the first token of each example never appears in the targets.
    self.assertEqual(raw_num_tokens, num_tokens + n_pairs)

  def test_text_only_dataset(self):
    tokenizer = tokenizers.WordTokenizer(vocab_file=WIKITEXT_VOCAB_FILE)

    batch_size = 4
    seq_len = 256
    dataset = paired_dataset.TextOnlyDataset(
        tokenizer,
        batch_size=batch_size,
        timesteps=seq_len,
        subset='valid',
        data_dir=WIKIGRAPHS_ROOT)
    data_iter = iter(dataset)
    batch = next(data_iter)
    faux_batch = dataset.return_faux_batch()

    self.assertCountEqual(list(batch.keys()),
                          ['obs', 'target', 'should_reset', 'mask'])
    self.assertCountEqual(list(faux_batch.keys()),
                          ['obs', 'target', 'should_reset', 'mask'])
    for k, v in batch.items():
      faux_v = faux_batch[k]
      self.assertEqual(v.shape, faux_v.shape)
      self.assertEqual(v.dtype, faux_v.dtype)

    self.assertEqual(batch['obs'].shape, (batch_size, seq_len))
    self.assertEqual(batch['target'].shape, (batch_size, seq_len))
    self.assertEqual(batch['should_reset'].shape, (batch_size, seq_len))
    self.assertEqual(batch['mask'].shape, (batch_size, seq_len))

    num_tokens = 0
    for batch in dataset:
      num_tokens += batch['mask'].sum()

    raw_dataset = paired_dataset.RawDataset(subset='valid', shuffle_data=False)
    raw_num_tokens = 0
    n_pairs = 0
    for pair in raw_dataset:
      raw_num_tokens += len(tokenizer.encode(
          pair.text, prepend_bos=True, append_eos=True))
      n_pairs += 1
    self.assertEqual(num_tokens + n_pairs, raw_num_tokens)

  def test_bow_retrieval_dataset(self):
    tokenizer = tokenizers.WordTokenizer(vocab_file=WIKITEXT_VOCAB_FILE)
    graph_tokenizer = tokenizers.GraphTokenizer(vocab_file=GRAPH_VOCAB_FILE)

    batch_size = 4
    seq_len = 256
    dataset = paired_dataset.Bow2TextDataset(
        tokenizer,
        graph_tokenizer,
        batch_size=batch_size,
        timesteps=seq_len,
        subsample_nodes=0.8,
        graph_retrieval_dataset=True,
        subset='valid',
        data_dir=WIKIGRAPHS_ROOT)
    data_iter = iter(dataset)
    batch = next(data_iter)

    self.assertEqual(batch['obs'].shape, (batch_size, seq_len))
    self.assertEqual(batch['target'].shape, (batch_size, seq_len))
    self.assertEqual(batch['should_reset'].shape, (batch_size, seq_len))
    self.assertEqual(batch['mask'].shape, (batch_size, seq_len))
    self.assertEqual(batch['graph_id'].shape, (batch_size,))
    self.assertEqual(batch['seq_id'].shape, (batch_size,))

  def test_graph_retrieval_dataset(self):
    tokenizer = tokenizers.WordTokenizer(vocab_file=WIKITEXT_VOCAB_FILE)
    graph_tokenizer = tokenizers.GraphTokenizer(vocab_file=GRAPH_VOCAB_FILE)

    batch_size = 4
    seq_len = 256
    dataset = paired_dataset.Graph2TextDataset(
        tokenizer,
        graph_tokenizer,
        batch_size=batch_size,
        timesteps=seq_len,
        subsample_nodes=0.8,
        graph_retrieval_dataset=True,
        subset='valid',
        data_dir=WIKIGRAPHS_ROOT)
    data_iter = iter(dataset)
    batch = next(data_iter)

    self.assertEqual(batch['obs'].shape, (batch_size, seq_len))
    self.assertEqual(batch['target'].shape, (batch_size, seq_len))
    self.assertEqual(batch['should_reset'].shape, (batch_size, seq_len))
    self.assertEqual(batch['mask'].shape, (batch_size, seq_len))
    self.assertEqual(batch['graph_id'].shape, (batch_size,))
    self.assertEqual(batch['seq_id'].shape, (batch_size,))


if __name__ == '__main__':
  absltest.main()
