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
"""Tests for wikigraphs.data.wikitext."""

from absl.testing import absltest
from wikigraphs.data import tokenizers
from wikigraphs.data import wikitext


WIKITEXT_ROOT = '/tmp/data/wikitext-103'
WIKITEXT_VOCAB_FILE = '/tmp/data/wikitext-vocab.csv'


class WikitextTest(absltest.TestCase):

  def test_wikitext_size(self):
    valid_set = wikitext.RawDataset(
        subset='valid', shuffle_data=False, data_dir=WIKITEXT_ROOT)
    n_tokens = 0
    n_articles = 0
    for article in valid_set:
      n_tokens += len([t for t in article.text.split(' ') if t])
      n_articles += 1

    # Dataset size must match published values.
    self.assertEqual(n_tokens, 217646)
    self.assertEqual(n_articles, 60)

  def test_wikitext_dataset_size(self):
    tokenizer = tokenizers.WordTokenizer(vocab_file=WIKITEXT_VOCAB_FILE)
    batch_size = 4
    timesteps = 256
    valid_set = wikitext.WikitextDataset(
        tokenizer=tokenizer, batch_size=batch_size, timesteps=timesteps,
        subset='valid', shuffle_data=False, repeat=False,
        data_dir=WIKITEXT_ROOT)
    n_tokens = 0
    n_bos = 0
    for batch in valid_set:
      n_tokens += (batch['obs'] != tokenizer.pad_token()).sum()
      n_bos += (batch['obs'] == tokenizer.bos_token()).sum()
      self.assertEqual(
          batch['obs'].shape, (batch_size, timesteps))
      self.assertEqual(
          batch['target'].shape, (batch_size, timesteps))
      self.assertEqual(
          batch['should_reset'].shape, (batch_size, timesteps))
      self.assertEqual(
          batch['mask'].shape, (batch_size, timesteps))

    n_tokens -= n_bos
    self.assertEqual(n_tokens, 217646)
    self.assertEqual(n_bos, 60)


if __name__ == '__main__':
  absltest.main()
