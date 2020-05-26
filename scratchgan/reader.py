# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for parsing text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

from absl import logging
import numpy as np
from tensorflow.compat.v1.io import gfile

# sequences: [N, MAX_TOKENS_SEQUENCE] array of int32
# lengths: [N, 2] array of int32, such that
#   lengths[i, 0] is the number of non-pad tokens in sequences[i, :]
FILENAMES = {
    "emnlp2017": ("train.json", "valid.json", "test.json"),
}

# EMNLP2017 sentences have max length 50, add one for a PAD token so that all
# sentences end with PAD.
MAX_TOKENS_SEQUENCE = {"emnlp2017": 52}

UNK = "<unk>"
PAD = " "

PAD_INT = 0


def tokenize(sentence):
  """Split a string into words."""
  return sentence.split(" ") + [PAD]


def _build_vocab(json_data):
  """Builds full vocab from json data."""
  vocab = collections.Counter()
  for sentence in json_data:
    tokens = tokenize(sentence["s"])
    vocab.update(tokens)
    for title in sentence["t"]:
      title_tokens = tokenize(title)
      vocab.update(title_tokens)
  # Most common words first.
  count_pairs = sorted(list(vocab.items()), key=lambda x: (-x[1], x[0]))
  words, _ = list(zip(*count_pairs))
  words = list(words)
  if UNK not in words:
    words = [UNK] + words
  word_to_id = dict(list(zip(words, list(range(len(words))))))

  # Tokens are now sorted by frequency. There's no guarantee that `PAD` will
  # end up at `PAD_INT` index. Enforce it by swapping whatever token is
  # currently at the `PAD_INT` index with the `PAD` token.
  word = list(word_to_id.keys())[list(word_to_id.values()).index(PAD_INT)]
  word_to_id[PAD], word_to_id[word] = word_to_id[word], word_to_id[PAD]
  assert word_to_id[PAD] == PAD_INT

  return word_to_id


def string_sequence_to_sequence(string_sequence, word_to_id):
  result = []
  for word in string_sequence:
    if word in word_to_id:
      result.append(word_to_id[word])
    else:
      result.append(word_to_id[UNK])
  return result


def _integerize(json_data, word_to_id, dataset):
  """Transform words into integers."""
  sequences = np.full((len(json_data), MAX_TOKENS_SEQUENCE[dataset]),
                      word_to_id[PAD], np.int32)
  sequence_lengths = np.zeros(shape=(len(json_data)), dtype=np.int32)
  for i, sentence in enumerate(json_data):
    sequence_i = string_sequence_to_sequence(
        tokenize(sentence["s"]), word_to_id)
    sequence_lengths[i] = len(sequence_i)
    sequences[i, :sequence_lengths[i]] = np.array(sequence_i)
  return {
      "sequences": sequences,
      "sequence_lengths": sequence_lengths,
  }


def get_raw_data(data_path, dataset, truncate_vocab=20000):
  """Load raw data from data directory "data_path".

  Reads text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  Args:
    data_path: string path to the directory where simple-examples.tgz has been
      extracted.
    dataset: one of ["emnlp2017"]
    truncate_vocab: int, number of words to keep in the vocabulary.

  Returns:
    tuple (train_data, valid_data, vocabulary) where each of the data
    objects can be passed to iterator.

  Raises:
    ValueError: dataset not in ["emnlp2017"].
  """
  if dataset not in FILENAMES:
    raise ValueError("Invalid dataset {}. Valid datasets: {}".format(
        dataset, list(FILENAMES.keys())))
  train_file, valid_file, _ = FILENAMES[dataset]

  train_path = os.path.join(data_path, train_file)
  valid_path = os.path.join(data_path, valid_file)

  with gfile.GFile(train_path, "r") as json_file:
    json_data_train = json.load(json_file)
  with gfile.GFile(valid_path, "r") as json_file:
    json_data_valid = json.load(json_file)

  word_to_id = _build_vocab(json_data_train)
  logging.info("Full vocab length: %d", len(word_to_id))
  # Assume the vocab is sorted by frequency.
  word_to_id_truncated = {
      k: v for k, v in word_to_id.items() if v < truncate_vocab
  }
  logging.info("Truncated vocab length: %d", len(word_to_id_truncated))

  train_data = _integerize(json_data_train, word_to_id_truncated, dataset)
  valid_data = _integerize(json_data_valid, word_to_id_truncated, dataset)
  return train_data, valid_data, word_to_id_truncated


def iterator(raw_data, batch_size, random=False):
  """Looping iterators on the raw data."""
  sequences = raw_data["sequences"]
  sequence_lengths = raw_data["sequence_lengths"]

  num_examples = sequences.shape[0]
  indice_range = np.arange(num_examples)

  if random:
    while True:
      indices = np.random.choice(indice_range, size=batch_size, replace=True)
      yield {
          "sequence": sequences[indices, :],
          "sequence_length": sequence_lengths[indices],
      }
  else:
    start = 0
    while True:
      sequence = sequences[start:(start + batch_size), :]
      sequence_length = sequence_lengths[start:(start + batch_size)]
      start += batch_size
      if start + batch_size > num_examples:
        start = (start + batch_size) % num_examples
      yield {
          "sequence": sequence,
          "sequence_length": sequence_length,
      }
