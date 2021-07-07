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
"""Wikitext-103 datasets."""

import re
from typing import NamedTuple, List

from absl import logging
import numpy as np

from wikigraphs.data import dataset
from wikigraphs.data import tokenizers
from wikigraphs.data import tools


# The data directory that contains subdirectories `wikitext-103` and
# `wikitext-103-raw`.
DATA_ROOT = '/tmp/data/wikitext-103'


class WikitextArticle(NamedTuple):
  title: str
  text: str


def articles_from_file(file_path: str) -> List[WikitextArticle]:
  """Read wikitext articles from file.

  Args:
    file_path: path to the input `.tokens` file.

  Returns:
    A list of `WikitextArticle` tuples.
  """
  with open(file_path, mode='rb') as f:
    content = f.read()
  content = content.decode('utf-8')

  title_re = re.compile(r'(\n = ([^=].*) = \n \n)')
  parts = title_re.split(content)

  # Skip the first part which is empty
  return [WikitextArticle(title=parts[i+1], text=parts[i] + parts[i+2])
          for i in range(1, len(parts), 3)]


class RawDataset(dataset.Dataset):
  """Raw text dataset for wikitext-103."""

  def __init__(self,
               subset: str = 'train',
               shuffle_data: bool = False,
               data_dir: str = None,
               version: str = 'tokens'):
    """Constructor.

    Args:
      subset: which subset to load, one of {"train", "valid", "test"}.
      shuffle_data: if set to True the data will be randomly shuffled.
      data_dir: if provided will be used instead of the default `DATA_ROOT` as
        the directory that contains the data.
      version: one of {'tokens', 'raw'}
    """
    super().__init__()
    self._subset = subset
    self._shuffle_data = shuffle_data
    self._data_dir = data_dir or DATA_ROOT
    self._dataset = None

    allowed_versions = ('tokens', 'raw')
    if version not in allowed_versions:
      raise ValueError(f'Version must be one of {allowed_versions}.')
    self._version = version

  def _load_data(self):
    """Prepare data for another pass through the dataset."""
    if self._dataset is None:
      data_root = self._data_dir + ('-raw' if self._version == 'raw' else '')
      self._dataset = articles_from_file(
          f'{data_root}/wiki.{self._subset}.{self._version}')

    def source():
      n_articles = len(self._dataset)
      if self._shuffle_data:
        idx = np.random.permutation(n_articles)
      else:
        idx = np.arange(n_articles)
      for i in range(n_articles):
        yield self._dataset[idx[i]]

    return source()


def normalize_title(title: str) -> str:
  """Normalize the wikitext article title by handling special characters."""
  return title.replace(
      '@-@', '-').replace('@,@', ',').replace('@.@', '.').replace(' ', '')


class WikitextDataset(dataset.Dataset):
  """Tokenized dataset for wikitext-103."""

  def __init__(self,
               tokenizer: tokenizers.Tokenizer,
               batch_size: int = 1,
               timesteps: int = 128,
               subset: str = 'train',
               shuffle_data: bool = True,
               data_dir: str = None,
               repeat: bool = False,
               debug: bool = False,
               **kwargs):
    """Constructor.

    Args:
      tokenizer: a tokenizer for text data.
      batch_size: number of sequences to put into a batch.
      timesteps: length of the sequences.
      subset: which subset to load, one of {"train", "valid", "test"}.
      shuffle_data: if set to True the data will be randomly shuffled.
      data_dir: if provided will be used instead of the default `DATA_ROOT` as
        the directory that contains the data.
      repeat: set to False to go through the data only once, otherwise go
        through the data indefinitely.
      debug: set to True to only load a small amount of data for fast debugging.
      **kwargs: other arguments (for interface compatibility).
    """
    super().__init__()
    self._tokenizer = tokenizer
    self._batch_size = batch_size
    self._timesteps = timesteps
    self._subset = subset
    self._shuffle_data = shuffle_data
    self._data_dir = data_dir
    self._repeat = repeat
    self._debug = debug
    self._dataset = None

  def _load_data(self):
    """Prepare data for one pass through the dataset."""
    # Pre-tokenize everything in our dataset so we don't have to when going
    # through the data more than once.
    if not self._dataset:
      raw_dataset = RawDataset(
          subset=self._subset, shuffle_data=False, data_dir=self._data_dir)
      if self._debug:
        # Load a small number of examples for debugging.
        self._dataset = [
            self._tokenizer.encode(next(raw_dataset).text, prepend_bos=True)
            for _ in range(5)]
      else:
        self._dataset = [self._tokenizer.encode(item.text, prepend_bos=True)
                         for item in raw_dataset]
      logging.info('%s set loaded, total %d examples.',
                   self._subset, len(self._dataset))

    def source():
      idx = np.random.permutation(len(self._dataset))
      for i in idx:
        yield self._dataset[i]

    def repeated_source():
      if self._repeat:
        while True:
          yield from source()
      else:
        yield from source()

    data_iter = tools.dynamic_batch(
        repeated_source(),
        self._batch_size,
        self._timesteps + 1,  # Extra token to count for the overlap.
        return_incomplete_batch=True,
        pad=True,
        pad_value=self._tokenizer.pad_token())
    data_iter = map(lambda x: dict(  # pylint: disable=g-long-lambda
        obs=x['obs'][:, :-1],
        target=x['obs'][:, 1:],
        should_reset=x['should_reset'][:, :-1],
        mask=(x['obs'][:, 1:] != self._tokenizer.pad_token()).astype(
            np.float32),
        ), data_iter)
    return data_iter

  def return_faux_batch(self):
    """Return a fake batch with the right shapes and dtypes."""
    obs = np.zeros((self._batch_size, self._timesteps), dtype=np.int32)
    target = np.zeros_like(obs, dtype=np.int32)
    should_reset = np.zeros_like(obs, dtype=np.float32)
    mask = np.zeros_like(obs, dtype=np.float32)
    return dict(obs=obs, target=target, should_reset=should_reset, mask=mask)
