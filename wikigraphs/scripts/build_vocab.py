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
"""Script for building vocabulary files for datasets."""

import collections
import csv
import enum
import io
import os
from typing import List, Tuple

from absl import app
from absl import flags
from absl import logging

from wikigraphs.data import io_tools
from wikigraphs.data import paired_dataset
from wikigraphs.data import tokenizers
from wikigraphs.data import wikitext


class DatasetType(enum.Enum):
  text = 1
  graph = 2
  wikitext = 3


FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', 'Path to the directory that contains the'
                    ' unzipped wikitext-103 data.')
flags.DEFINE_string('vocab_file_path', '', 'Path to the output vocab file.')
flags.DEFINE_enum_class('data_type', DatasetType.wikitext, DatasetType,
                        'One of {`wikitext`, `graph`, `text`}.')
flags.DEFINE_integer('threshold', 1, 'Frequency threshold for a word to be'
                     ' included in the vocabulary.')
flags.DEFINE_string('version', 'max256', 'Which version of paired data to use.')


def get_vocab(dataset: wikitext.RawDataset) -> List[Tuple[str, int]]:
  """Build vocabulary, return (word, count) tuples sorted by count."""
  vocab = collections.defaultdict(int)

  for pair in dataset:
    for t in pair.text.split(' '):
      if t:
        vocab[t] += 1

  return sorted(vocab.items(), key=lambda t: -t[1])


def write_vocab(vocab: List[Tuple[str, int]], output_path: str):
  """Write a vocab list to a file."""
  output_dir = os.path.dirname(output_path)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  with open(output_path, mode='wb') as f_:
    with io.TextIOWrapper(f_, encoding='utf-8') as f:
      w = csv.writer(f)
      w.writerows(vocab)


def build_wikitext_vocab():
  logging.info('Loading the dataset.')
  dataset = wikitext.RawDataset(subset='train', data_dir=FLAGS.data_dir)
  logging.info('Building the vocab.')
  vocab = get_vocab(dataset)
  logging.info('Finished, vocab size %d, total number of tokens %d',
               len(vocab), sum([c for _, c in vocab]))
  logging.info('Writing the vocab to %s', FLAGS.vocab_file_path)
  write_vocab(vocab, FLAGS.vocab_file_path)


def build_graph_vocab():
  """Build vocabulary for graph data."""
  logging.info('Loading the dataset.')
  dataset = paired_dataset.ParsedDataset(
      subset='train', data_dir=FLAGS.data_dir, version=FLAGS.version)
  logging.info('Building graph vocab.')

  vocab = collections.defaultdict(int)
  for pair in dataset:
    graph = pair.graph
    for n in graph.nodes():
      for t in tokenizers.GraphTokenizer.split_node(n):
        if t:
          vocab[t] += 1
    for _, _, e in graph.edges():
      for t in tokenizers.GraphTokenizer.split_edge(e):
        if t:
          vocab[t] += 1

  vocab = sorted(vocab.items(), key=lambda t: -t[1])
  vocab = [k for k, v in vocab if v >= FLAGS.threshold]

  logging.info('Finished, vocab size %d.', len(vocab))
  logging.info('Writing the vocab to %s.', FLAGS.vocab_file_path)

  io_tools.write_txt_file(FLAGS.vocab_file_path, '\n'.join(vocab),
                          # Some unicode characters requires utf-16 to encode.
                          encoding='utf-16')


def build_text_vocab():
  """Build vocabulary for the text part of the graph-to-text data."""
  logging.info('Loading the dataset.')
  dataset = paired_dataset.ParsedDataset(
      subset='train', data_dir=FLAGS.data_dir, version=FLAGS.version)
  logging.info('Building text vocab.')

  vocab = collections.defaultdict(int)
  for pair in dataset:
    for t in pair.text.split(' '):
      if t:
        vocab[t] += 1

  vocab = sorted(vocab.items(), key=lambda t: -t[1])
  logging.info('Finished, vocab size %d, total number of tokens %d.',
               len(vocab), sum([v for _, v in vocab]))
  vocab = [(k, v) for k, v in vocab if v >= FLAGS.threshold]
  logging.info('After filtering, vocab size %d.', len(vocab))
  logging.info('Writing the vocab to %s.', FLAGS.vocab_file_path)

  write_vocab(vocab, FLAGS.vocab_file_path)


def main(_):
  if FLAGS.data_type == DatasetType.wikitext:
    build_wikitext_vocab()
  elif FLAGS.data_type == DatasetType.text:
    build_text_vocab()
  elif FLAGS.data_type == DatasetType.graph:
    build_graph_vocab()
  else:
    raise ValueError(f'Unknown data type {FLAGS.data_type}.')


if __name__ == '__main__':
  app.run(main)
