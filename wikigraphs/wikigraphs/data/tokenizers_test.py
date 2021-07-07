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
"""Tests for wikigraphs.data.tokenizers."""
from absl.testing import absltest
from wikigraphs.data import tokenizers


WIKITEXT_VOCAB_FILE = '/tmp/data/wikitext-vocab.csv'
GRAPH_VOCAB_FILE = '/tmp/data/graph-vocab.csv'


class TokenizerTest(absltest.TestCase):

  def test_tokenizer(self):
    tokenizer = tokenizers.WordTokenizer(vocab_file=WIKITEXT_VOCAB_FILE)
    # Vocab size must match published number.
    self.assertEqual(tokenizer.vocab_size, 267735 + 2)

    s = 'Hello world ! \n How are you ?'
    encoded = tokenizer.encode(s, prepend_bos=True)
    self.assertEqual(encoded.shape, (9,))
    decoded = tokenizer.decode(encoded)
    self.assertEqual(s, decoded)

  def test_graph_tokenizer_tokenize_nodes_edges(self):
    self.assertEqual(
        tokenizers.GraphTokenizer.split_node(
            '"Hello, how are you?"'),
        ['hello', ',', 'how', 'are', 'you', '?'])
    self.assertEqual(
        tokenizers.GraphTokenizer.split_node(
            '"This building was built in 1998."'),
        ['this', 'building', 'was', 'built', 'in', '<number>', '.'])
    self.assertEqual(
        tokenizers.GraphTokenizer.split_node('ns/m.030ssw'),
        ['<entity>'])

    self.assertEqual(
        tokenizers.GraphTokenizer.split_edge('ns/common.topic.description'),
        ['common', 'topic', 'description'])
    self.assertEqual(
        tokenizers.GraphTokenizer.split_edge('ns/type.object.name'),
        ['type', 'object', 'name'])

  def test_graph_tokenizer_vocab(self):
    tokenizer = tokenizers.GraphTokenizer(vocab_file=GRAPH_VOCAB_FILE)
    self.assertEqual(tokenizer.vocab_size, 31087 + 3)


if __name__ == '__main__':
  absltest.main()
