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
"""Generators for text data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from scratchgan import utils


class LSTMGen(snt.AbstractModule):
  """A multi-layer LSTM language model.

  Uses tied input/output embedding weights.
  """

  def __init__(self,
               vocab_size,
               feature_sizes,
               max_sequence_length,
               batch_size,
               use_layer_norm,
               trainable_embedding_size,
               input_dropout,
               output_dropout,
               pad_token,
               embedding_source=None,
               vocab_file=None,
               name='lstm_gen'):
    super(LSTMGen, self).__init__(name=name)
    self._feature_sizes = feature_sizes
    self._max_sequence_length = max_sequence_length
    self._vocab_size = vocab_size
    self._batch_size = batch_size
    self._use_layer_norm = use_layer_norm
    self._trainable_embedding_size = trainable_embedding_size
    self._embedding_source = embedding_source
    self._vocab_file = vocab_file
    self._input_dropout = input_dropout
    self._output_dropout = output_dropout
    self._pad_token = pad_token
    if self._embedding_source:
      assert vocab_file

  def _build(self, is_training=True, temperature=1.0):
    input_keep_prob = (1. - self._input_dropout) if is_training else 1.0
    output_keep_prob = (1. - self._output_dropout) if is_training else 1.0

    batch_size = self._batch_size
    max_sequence_length = self._max_sequence_length
    if self._embedding_source:
      all_embeddings = utils.make_partially_trainable_embeddings(
          self._vocab_file, self._embedding_source, self._vocab_size,
          self._trainable_embedding_size)
    else:
      all_embeddings = tf.get_variable(
          'trainable_embeddings',
          shape=[self._vocab_size, self._trainable_embedding_size],
          trainable=True)
    _, self._embedding_size = all_embeddings.shape.as_list()
    input_embeddings = tf.nn.dropout(all_embeddings, keep_prob=input_keep_prob)
    output_embeddings = tf.nn.dropout(
        all_embeddings, keep_prob=output_keep_prob)

    out_bias = tf.get_variable(
        'out_bias', shape=[1, self._vocab_size], dtype=tf.float32)
    in_proj = tf.get_variable(
        'in_proj', shape=[self._embedding_size, self._feature_sizes[0]])
    # If more than 1 layer, then output has dim sum(self._feature_sizes),
    # which is different from input dim == self._feature_sizes[0]
    # So we need a different projection matrix for input and output.
    if len(self._feature_sizes) > 1:
      out_proj = tf.get_variable(
          'out_proj', shape=[self._embedding_size,
                             sum(self._feature_sizes)])
    else:
      out_proj = in_proj

    encoder_cells = []
    for feature_size in self._feature_sizes:
      encoder_cells += [
          snt.LSTM(feature_size, use_layer_norm=self._use_layer_norm)
      ]
    encoder_cell = snt.DeepRNN(encoder_cells)
    state = encoder_cell.initial_state(batch_size)

    # Manual unrolling.
    samples_list, logits_list, logprobs_list, embeddings_list = [], [], [], []
    sample = tf.tile(
        tf.constant(self._pad_token, dtype=tf.int32)[None], [batch_size])
    logging.info('Unrolling over %d steps.', max_sequence_length)
    for _ in range(max_sequence_length):
      # Input is sampled word at t-1.
      embedding = tf.nn.embedding_lookup(input_embeddings, sample)
      embedding.shape.assert_is_compatible_with(
          [batch_size, self._embedding_size])
      embedding_proj = tf.matmul(embedding, in_proj)
      embedding_proj.shape.assert_is_compatible_with(
          [batch_size, self._feature_sizes[0]])

      outputs, state = encoder_cell(embedding_proj, state)
      outputs_proj = tf.matmul(outputs, out_proj, transpose_b=True)
      logits = tf.matmul(
          outputs_proj, output_embeddings, transpose_b=True) + out_bias
      categorical = tfp.distributions.Categorical(logits=logits/temperature)
      sample = categorical.sample()
      logprobs = categorical.log_prob(sample)

      samples_list.append(sample)
      logits_list.append(logits)
      logprobs_list.append(logprobs)
      embeddings_list.append(embedding)

    # Create an op to retrieve embeddings for full sequence, useful for testing.
    embeddings = tf.stack(  # pylint: disable=unused-variable
        embeddings_list,
        axis=1,
        name='embeddings')
    sequence = tf.stack(samples_list, axis=1)
    logprobs = tf.stack(logprobs_list, axis=1)

    # The sequence stops after the first occurrence of a PAD token.
    sequence_length = utils.get_first_occurrence_indices(
        sequence, self._pad_token)
    mask = utils.get_mask_past_symbol(sequence, self._pad_token)
    masked_sequence = sequence * tf.cast(mask, tf.int32)
    masked_logprobs = logprobs * tf.cast(mask, tf.float32)
    return {
        'sequence': masked_sequence,
        'sequence_length': sequence_length,
        'logprobs': masked_logprobs
    }
