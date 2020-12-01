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
"""Utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.io import gfile
from scratchgan import reader

EVAL_FILENAME = "evaluated_checkpoints.csv"
GLOVE_DIM = 300
GLOVE_STD = 0.3836  # Standard dev. of GloVe embeddings.


def _get_embedding_initializer(vocab_file, embedding_source, vocab_size):
  """Loads pretrained embeddings from a file in GloVe format."""
  with gfile.GFile(embedding_source, "r") as f:
    embedding_lines = f.readlines()

  # First line contains embedding dim.
  _, embedding_dim = list(map(int, embedding_lines[0].split()))
  # Get the tokens as strings.
  tokens = [line.split()[0] for line in embedding_lines[1:]]
  # Get the actual embedding matrix.
  unsorted_emb = np.array(
      [[float(x) for x in line.split()[1:]] for line in embedding_lines[1:]])

  # Get the expected vocab order.
  with gfile.GFile(vocab_file, "r") as f:
    tokens_order = [l.strip() for l in f.readlines()]
  assert vocab_size == len(tokens_order)

  # Put the embeddings in the order.
  sorted_emb = np.zeros((vocab_size, embedding_dim))
  for i, token in enumerate(tokens_order):
    if token in tokens:
      sorted_emb[i, :] = unsorted_emb[tokens.index(token), :]
    else:  # If we don't have a pretrained embedding, initialize randomly.
      sorted_emb[i, :] = np.random.normal(
          loc=0.0, scale=GLOVE_STD, size=(GLOVE_DIM,))

  return sorted_emb.astype(np.float32)


def append_position_signal(embeddings, position_dim=8):
  """Append position signal. See get_position_signal."""
  batch_size, sequence_length, embedding_dim = embeddings.get_shape().as_list()
  positions = get_position_signal(sequence_length, position_dim)

  # Append to embeddings.
  position_inputs = tf.tile(positions[None, :, :], [batch_size, 1, 1])
  embeddings_pos = tf.concat([embeddings, position_inputs], axis=2)
  embeddings_pos.shape.assert_is_compatible_with(
      [batch_size, sequence_length, embedding_dim + position_dim])
  return embeddings_pos


def get_position_signal(sequence_length, position_dim=8):
  """Return fixed position signal as sine waves.

  Sine waves frequencies are linearly spaced so that shortest is 2 and
  longest is half the maximum length. That way the longest frequency
  is long enough to be monotonous over the whole sequence length.
  Sine waves are also shifted so that they don't all start with the same
  value.
  We don't use learned positional embeddings because these embeddings are
  projected linearly along with the original embeddings, and the projection is
  learned.

  Args:
    sequence_length: int, T, length of the sequence..
    position_dim: int, P, number of sine waves.

  Returns:
    A [T, P] tensor, position embeddings.
  """
  # Compute the frequencies.
  periods = tf.exp(
      tf.lin_space(
          tf.log(2.0), tf.log(tf.to_float(sequence_length)), position_dim))
  frequencies = 1.0 / periods  # Shape [T, P].

  # Compute the sine waves.
  xs = frequencies[None, :] * tf.to_float(tf.range(sequence_length)[:, None])
  shifts = tf.lin_space(0.0, 2.0, position_dim)[None, :]  # [1, P]
  positions = tf.math.cos(math.pi * (xs + shifts))  # [T, P]
  positions.shape.assert_is_compatible_with([sequence_length, position_dim])
  return positions


def get_mask_by_length(lengths, max_length):
  """Returns a mask where x[i , j] = (j < lengths[i]).

  Args:
    lengths: [B] tensor of int32 such that 0 <= lengths[i] <= max_length.
    max_length: scalar tensor of int32.

  Returns:
    [B, max_length] tensor of booleans such that x[i, j] is True
    if and only if j < lengths[i].
  """
  batch_size = lengths.get_shape().as_list()[0]
  indices = tf.range(start=0, limit=max_length)
  all_indices = tf.tile(indices[None, :], [batch_size, 1])
  all_lengths = tf.tile(lengths[:, None], [1, max_length])
  mask = (all_indices < all_lengths)
  mask_boolean = tf.cast(mask, tf.bool)
  return mask_boolean


def get_mask_past_symbol(reference, symbol, optimize_for_tpu=False):
  """For each row, mask is True before and at the first occurrence of symbol."""
  batch_size, max_length = reference.get_shape().as_list()
  symbol = tf.convert_to_tensor(symbol)
  symbol.shape.assert_is_compatible_with([])

  first_indices = get_first_occurrence_indices(reference, symbol,
                                               optimize_for_tpu)
  first_indices.shape.assert_is_compatible_with([batch_size])

  keep_lengths = tf.minimum(first_indices, max_length)
  mask = get_mask_by_length(keep_lengths, max_length)
  mask.shape.assert_is_compatible_with([batch_size, max_length])
  mask.set_shape([batch_size, max_length])
  return mask


def get_first_occurrence_indices(reference, symbol, optimize_for_tpu=False):
  """For each row in reference, get index after the first occurrence of symbol.

  If symbol is not present on a row, return reference.shape[1] instead.

  Args:
    reference: [B, T] tensor of elements of the same type as symbol.
    symbol: int or [] scalar tensor of the same dtype as symbol.
    optimize_for_tpu: bool, whether to use a TPU-capable variant.

  Returns:
    A [B] reference of tf.int32 where x[i] is such that
    reference[i, x[i]-1] == symbol, and reference[i, j] != symbol
    for j<i-1. If symbol is not present on row i then x[i] = T.
  """
  if optimize_for_tpu:
    # Run code which can be compiled on TPU.
    # Transpose refernce to [T, B]
    reference = tf.transpose(reference, [1, 0])
    range_tensor = tf.range(reference.shape.as_list()[0])
    indexes = tf.stack([range_tensor] * reference.shape.as_list()[1], 1)
    symbol = tf.stack([symbol] * reference.shape.as_list()[1], 0)

    initial_indices = tf.constant(
        reference.shape.as_list()[0],
        shape=[reference.shape.as_list()[1]],
        dtype=tf.int32)

    # We want a function which moves backwards.
    def fn(current_index, elems):
      ref, ind = elems
      return tf.where(tf.equal(ref, symbol), ind + 1, current_index)

    min_indexes = tf.scan(
        fn, (reference, indexes),
        initializer=initial_indices,
        parallel_iterations=1,
        reverse=True)
    return min_indexes[0]

  batch_size, max_length = reference.get_shape().as_list()
  symbol = tf.convert_to_tensor(symbol)
  symbol.shape.assert_is_compatible_with([])
  # Add symbol at the end of each row, to make sure tf.where works.
  tensor = tf.concat(
      [reference, tf.tile(symbol[None, None], [batch_size, 1])], axis=1)
  index_all_occurrences = tf.where(tf.equal(tensor, symbol))
  index_all_occurrences = tf.cast(index_all_occurrences, tf.int32)
  # `index_all_occurrences` is a [N, 2] tensor with coordinates of all positions
  # of `symbol` in `tensor`. So N will be >= batch size since there can be
  # several `symbol` in one row of tensor. We need to take only the position
  # of the first occurrence for each row. `segment_min` does that, taking the
  # lowest column index for each row index.
  index_first_occurrences = tf.segment_min(index_all_occurrences[:, 1],
                                           index_all_occurrences[:, 0])
  index_first_occurrences.set_shape([batch_size])
  index_first_occurrences = tf.minimum(index_first_occurrences + 1, max_length)
  return index_first_occurrences


def sequence_to_sentence(sequence, id_to_word):
  """Turn a sequence into a sentence , inverse of sentence_to_sequence."""
  words = []
  for token_index in sequence:
    if token_index in id_to_word:
      words.append(id_to_word[token_index])
    else:
      words.append(reader.UNK)
  return " ".join(words)


def batch_sequences_to_sentences(sequences, id_to_word):
  return [sequence_to_sentence(sequence, id_to_word) for sequence in sequences]


def write_eval_results(checkpoint_dir, all_gen_sentences, checkpoint_name,
                       mean_train_prob, mean_valid_prob, mean_gen_prob, fid):
  """Write evaluation results to disk."""
  to_write = ",".join(
      map(str, [
          checkpoint_name, mean_train_prob, mean_valid_prob, mean_gen_prob, fid
      ]))
  eval_filepath = os.path.join(checkpoint_dir, EVAL_FILENAME)
  previous_eval_content = ""
  if gfile.exists(eval_filepath):
    with gfile.GFile(eval_filepath, "r") as f:
      previous_eval_content = f.read()
  with gfile.GFile(eval_filepath, "w") as f:
    f.write(previous_eval_content + to_write + "\n")

  with gfile.GFile(
      os.path.join(checkpoint_dir, checkpoint_name + "_sentences.txt"),
      "w") as f:
    f.write("\n".join(all_gen_sentences))


def maybe_pick_models_to_evaluate(checkpoint_dir):
  """Pick a checkpoint to evaluate that has not been evaluated already."""
  logging.info("Picking checkpoint to evaluate from %s.", checkpoint_dir)

  filenames = gfile.listdir(checkpoint_dir)
  filenames = [f[:-5] for f in filenames if f[-5:] == ".meta"]
  logging.info("Found existing checkpoints: %s", filenames)

  evaluated_filenames = []
  if gfile.exists(os.path.join(checkpoint_dir, EVAL_FILENAME)):
    with gfile.GFile(os.path.join(checkpoint_dir, EVAL_FILENAME), "r") as f:
      evaluated_filenames = [l.strip().split(",")[0] for l in f.readlines()]
    logging.info("Found already evaluated checkpoints: %s", evaluated_filenames)

  checkpoints_to_evaluate = [
      f for f in filenames if f not in evaluated_filenames
  ]
  logging.info("Remaining potential checkpoints: %s", checkpoints_to_evaluate)

  if checkpoints_to_evaluate:
    return os.path.join(checkpoint_dir, checkpoints_to_evaluate[0])
  else:
    return None


def get_embedding_path(data_dir, dataset):
  """By convention, this is where we store the embedding."""
  return os.path.join(data_dir, "glove_%s.txt" % dataset)


def make_partially_trainable_embeddings(vocab_file, embedding_source,
                                        vocab_size, trainable_embedding_size):
  """Makes embedding matrix with pretrained GloVe [1] part and trainable part.

  [1] Pennington, J., Socher, R., & Manning, C. (2014, October). Glove: Global
  vectors for word representation. In Proceedings of the 2014 conference on
  empirical methods in natural language processing (EMNLP) (pp. 1532-1543).

  Args:
    vocab_file: vocabulary file.
    embedding_source: path to the actual embeddings.
    vocab_size: number of words in vocabulary.
    trainable_embedding_size: size of the trainable part of the embeddings.

  Returns:
    A matrix of partially pretrained embeddings.
  """

  # Our embeddings have 2 parts: a pre-trained, frozen, GloVe part,
  # and a trainable, randomly initialized part.
  # The standard deviation of the GloVe part is used to initialize
  # the trainable part, so that both part have roughly the same distribution.
  #
  # Let g_ij be the j-th coordinates of the GloVe embedding of the i-th word.
  # So that 0 < i < |vocab| and 0 < j < 300.
  # Then sum_ij (g_ij - sum_kl g_kl)^2 = (0.3836)^2
  #
  # In reality g_ij follows a truncated normal distribution
  # min(max(N(0, s), -4.2), 4.2) but we approximate it by N(0, 0.3836).
  embedding_initializer = _get_embedding_initializer(
      vocab_file=vocab_file,
      embedding_source=embedding_source,
      vocab_size=vocab_size)
  pretrained_embedding = tf.get_variable(
      "pretrained_embedding",
      initializer=embedding_initializer,
      dtype=tf.float32)
  trainable_embedding = tf.get_variable(
      "trainable_embedding",
      shape=[vocab_size, trainable_embedding_size],
      initializer=tf.initializers.random_normal(mean=0.0, stddev=GLOVE_STD))
  # We just concatenate embeddings, they will pass through a projection
  # matrix afterwards.
  embedding = tf.concat([pretrained_embedding, trainable_embedding], axis=1)
  return embedding
