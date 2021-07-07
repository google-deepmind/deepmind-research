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
"""Tokenizers for text data."""

import abc
import csv
import io
import re
from typing import List

import nltk
import numpy as np

from wikigraphs.data import io_tools


class Tokenizer(abc.ABC):
  """Base class for tokenizers."""

  @abc.abstractmethod
  def encode(self,
             inputs: str,
             prepend_bos: bool = False,
             append_eos: bool = False) -> np.ndarray:
    """Encode input string into an array of token IDs.

    Args:
      inputs: a string.
      prepend_bos: set to True to add <bos> token at the beginning of the token
        sequence.
      append_eos: set to True to add <eos> token at the end of the token
        sequence.

    Returns:
      tokens: [n_tokens] int array.
    """

  @abc.abstractmethod
  def decode(self, inputs) -> str:
    """Decode a sequence of tokens back into a string.

    Args:
      inputs: array or list of ints.

    Returns:
      s: the decoded string using this tokenizer.
    """

  @property
  @abc.abstractmethod
  def vocab_size(self) -> int:
    """Size of the vocabulary."""

  @abc.abstractmethod
  def pad_token(self) -> int:
    """ID of the <pad> token."""

  @abc.abstractmethod
  def bos_token(self) -> int:
    """ID of the <bos> token."""


class WordTokenizer(Tokenizer):
  """Word-level tokenizer for white-space separated text data."""

  def __init__(self, vocab_file: str):
    """Constructor.

    Args:
      vocab_file: a csv vocab file.
    """
    content = io_tools.read_txt_file(vocab_file, encoding='utf-8')

    with io.StringIO(content) as f:
      r = csv.reader(f)
      vocab = [w for w, _ in r]

    # Add pad and bos tokens to the vocab
    to_add = ['<pad>', '<bos>']
    if '<unk>' not in vocab:
      to_add.append('<unk>')
    vocab = to_add + vocab

    # token-index mappings
    self._t2i = {t: i for i, t in enumerate(vocab)}
    self._i2t = {i: t for t, i in self._t2i.items()}

    self._unk_token = self._t2i['<unk>']
    self._bos_token = self._t2i['<bos>']
    self._pad_token = self._t2i['<pad>']

  @property
  def vocab_size(self):
    return len(self._t2i)

  def encode(self, inputs, prepend_bos=False, append_eos=False):
    tokens = [self._t2i.get(t, self._unk_token) for t in inputs.split(' ') if t]
    if prepend_bos:
      tokens = [self._bos_token] + tokens
    if append_eos:
      # Reuse <bos> as <eos>.
      tokens.append(self._bos_token)
    return np.array(tokens, dtype=np.int32)

  def decode(self, inputs):
    """Decode a sequence of token IDs back into a string."""
    # Remove the first <bos> token if there is any.
    if inputs[0] == self._bos_token:
      inputs = inputs[1:]
    tokens = []
    for i in inputs:
      # Use <bos> also as <eos> and stop there.
      if i == self._bos_token:
        break
      tokens.append(self._i2t[i])
    return ' '.join(tokens)

  def pad_token(self):
    return self._pad_token

  def bos_token(self):
    return self._bos_token


class GraphTokenizer:
  """Tokenizer for the content on the graphs."""

  def __init__(self, vocab_file: str):
    """Constructor.

    Args:
      vocab_file: path to a vocab file.
    """
    content = io_tools.read_txt_file(vocab_file, encoding='utf-16')

    vocab = content.split('\n')
    vocab = ['<pad>', '<bos>', '<unk>'] + vocab

    # token-index mappings
    self._t2i = {t: i for i, t in enumerate(vocab)}
    self._i2t = {i: t for t, i in self._t2i.items()}

    self._unk_token = self._t2i['<unk>']
    self._bos_token = self._t2i['<bos>']
    self._pad_token = self._t2i['<pad>']

  @property
  def vocab_size(self):
    return len(self._t2i)

  def encode_node(self, txt: str) -> np.ndarray:
    return np.array([self._t2i.get(t, self._unk_token)
                     for t in self.split_node(txt)])

  def encode_edge(self, txt: str) -> np.ndarray:
    return np.array([self._t2i.get(t, self._unk_token)
                     for t in self.split_edge(txt)])

  def encode(self, inputs, prepend_bos=False, append_eos=False):
    tokens = [self._t2i.get(t, self._unk_token) for t in inputs.split(' ') if t]
    if prepend_bos:
      tokens = [self._bos_token] + tokens
    if append_eos:
      # Reuse <bos> as <eos>.
      tokens.append(self._bos_token)
    return np.array(tokens, dtype=np.int32)

  def decode(self, inputs):
    """Decode a sequence of token IDs back into a string."""
    # Remove the first <bos> token if there is any.
    if inputs[0] == self._bos_token:
      inputs = inputs[1:]
    tokens = []
    for i in inputs:
      # Use <bos> also as <eos> and stop there.
      if i == self._bos_token:
        break
      tokens.append(self._i2t[i])
    return ' '.join(tokens)

  @classmethod
  def split_node(cls, txt: str) -> List[str]:
    """Split a node string into a sequence of tokens."""
    if txt[0] == '"' and txt[-1] == '"':  # Node is a string literal.
      tokens = nltk.wordpunct_tokenize(io_tools.normalize_freebase_string(
          txt[1:-1].lower()))
      for i, t in enumerate(tokens):
        if t.isnumeric():
          tokens[i] = '<number>'
      return tokens
    else:  # If node is not a string literal it is always an entity.
      return ['<entity>']

  @classmethod
  def split_edge(cls, txt: str) -> List[str]:
    """Split an edge string into a sequence of tokens."""
    return re.split('[._ ]+', txt.lower().split('/')[1])

  def pad_token(self):
    return self._pad_token

  def bos_token(self):
    return self._bos_token
