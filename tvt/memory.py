# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Memory Reader/Writer for RMA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import sonnet as snt
import tensorflow.compat.v1 as tf

ReadInformation = collections.namedtuple(
    'ReadInformation', ('weights', 'indices', 'keys', 'strengths'))


class MemoryWriter(snt.RNNCore):
  """Memory Writer Module."""

  def __init__(self, mem_shape, name='memory_writer'):
    """Initializes the `MemoryWriter`.

    Args:
      mem_shape: The shape of the memory `(num_rows, memory_width)`.
      name: The name to use for the Sonnet module.
    """
    super(MemoryWriter, self).__init__(name=name)
    self._mem_shape = mem_shape

  def _build(self, inputs, state):
    """Inserts z into the argmin row of usage markers and updates all rows.

    Returns an operation that, when executed, correctly updates the internal
    state and usage markers.

    Args:
      inputs: A tuple consisting of:
          * z, the value to write at this timestep
          * mem_state, the state of the memory at this timestep before writing
      state: The state is just the write_counter.

    Returns:
      A tuple of the new memory state and a tuple containing the next state.
    """
    z, mem_state = inputs

    # Stop gradient on writes to memory.
    z = tf.stop_gradient(z)

    prev_write_counter = state
    new_row_value = z

    # Find the index to insert the next row into.
    num_mem_rows = self._mem_shape[0]
    write_index = tf.cast(prev_write_counter, dtype=tf.int32) % num_mem_rows
    one_hot_row = tf.one_hot(write_index, num_mem_rows)
    write_counter = prev_write_counter + 1

    # Insert state variable to new row.
    # First you need to size it up to the full size.
    insert_new_row = lambda mem, o_hot, z: mem - (o_hot * mem) + (o_hot * z)
    new_mem = insert_new_row(mem_state,
                             tf.expand_dims(one_hot_row, axis=-1),
                             tf.expand_dims(new_row_value, axis=-2))

    new_state = write_counter

    return new_mem, new_state

  @property
  def state_size(self):
    """Returns a description of the state size, without batch dimension."""
    return tf.TensorShape([])

  @property
  def output_size(self):
    """Returns a description of the output size, without batch dimension."""
    return self._mem_shape


class MemoryReader(snt.AbstractModule):
  """Memory Reader Module."""

  def __init__(self,
               memory_word_size,
               num_read_heads,
               top_k=0,
               memory_size=None,
               name='memory_reader'):
    """Initializes the `MemoryReader`.

    Args:
      memory_word_size: The dimension of the 1-D read keys this memory reader
        should produce. Each row of the memory is of length `memory_word_size`.
      num_read_heads: The number of reads to perform.
      top_k: Softmax and summation when reading is only over top k most similar
        entries in memory. top_k=0 (default) means dense reads, i.e. no top_k.
      memory_size: Number of rows in memory.
      name: The name for this Sonnet module.
    """
    super(MemoryReader, self).__init__(name=name)
    self._memory_word_size = memory_word_size
    self._num_read_heads = num_read_heads
    self._top_k = top_k

    # This is not an RNNCore but it is useful to expose the output size.
    self._output_size = num_read_heads * memory_word_size

    num_read_weights = top_k if top_k > 0 else memory_size
    self._read_info_size = ReadInformation(
        weights=tf.TensorShape([num_read_heads, num_read_weights]),
        indices=tf.TensorShape([num_read_heads, num_read_weights]),
        keys=tf.TensorShape([num_read_heads, memory_word_size]),
        strengths=tf.TensorShape([num_read_heads]),
    )

    with self._enter_variable_scope():
      # Transforms to value-based read for each read head.
      output_dim = (memory_word_size + 1) * num_read_heads
      self._keys_and_read_strengths_generator = snt.Linear(output_dim)

  def _build(self, inputs):
    """Looks up rows in memory.

    In the args list, we have the following conventions:
      B: batch size
      M: number of slots in a row of the memory matrix
      R: number of rows in the memory matrix
      H: number of read heads in the memory controller

    Args:
      inputs: A tuple of
        *  read_inputs, a tensor of shape [B, ...] that will be flattened and
             passed through a linear layer to get read keys/read_strengths for
             each head.
        *  mem_state, the primary memory tensor. Of shape [B, R, M].

    Returns:
      The read from the memory (concatenated across read heads) and read
        information.
    """
    # Assert input shapes are compatible and separate inputs.
    _assert_compatible_memory_reader_input(inputs)
    read_inputs, mem_state = inputs

    # Determine the read weightings for each key.
    flat_outputs = self._keys_and_read_strengths_generator(
        snt.BatchFlatten()(read_inputs))

    # Separate the read_strengths from the rest of the weightings.
    h = self._num_read_heads
    flat_keys = flat_outputs[:, :-h]
    read_strengths = tf.nn.softplus(flat_outputs[:, -h:])

    # Reshape the weights.
    read_shape = (self._num_read_heads, self._memory_word_size)
    read_keys = snt.BatchReshape(read_shape)(flat_keys)

    # Read from memory.
    memory_reads, read_weights, read_indices, read_strengths = (
        read_from_memory(read_keys, read_strengths, mem_state, self._top_k))
    concatenated_reads = snt.BatchFlatten()(memory_reads)

    return concatenated_reads, ReadInformation(
        weights=read_weights,
        indices=read_indices,
        keys=read_keys,
        strengths=read_strengths)

  @property
  def output_size(self):
    """Returns a description of the output size, without batch dimension."""
    return self._output_size, self._read_info_size


def read_from_memory(read_keys, read_strengths, mem_state, top_k):
  """Function for cosine similarity content based reading from memory matrix.

  In the args list, we have the following conventions:
    B: batch size
    M: number of slots in a row of the memory matrix
    R: number of rows in the memory matrix
    H: number of read heads (of the controller or the policy)
    K: top_k if top_k>0

  Args:
    read_keys: the read keys of shape [B, H, M].
    read_strengths: the coefficients used to compute the normalised weighting
      vector of shape [B, H].
    mem_state: the primary memory tensor. Of shape [B, R, M].
    top_k: only use top k read matches, other reads do not go into softmax and
      are zeroed out in the output. top_k=0 (default) means use dense reads.

  Returns:
    The memory reads [B, H, M], read weights [B, H, top k], read indices
      [B, H, top k], and read strengths [B, H, 1].
  """
  _assert_compatible_read_from_memory_inputs(read_keys, read_strengths,
                                             mem_state)
  batch_size = read_keys.shape[0]
  num_read_heads = read_keys.shape[1]

  with tf.name_scope('memory_reading'):
    # Scale such that all rows are L2-unit vectors, for memory and read query.
    scaled_read_keys = tf.math.l2_normalize(read_keys, axis=-1)  # [B, H, M]
    scaled_mem = tf.math.l2_normalize(mem_state, axis=-1)  # [B, R, M]

    # The cosine distance is then their dot product.
    # Find the cosine distance between each read head and each row of memory.
    cosine_distances = tf.matmul(
        scaled_read_keys, scaled_mem, transpose_b=True)  # [B, H, R]

    # The rank must match cosine_distances for broadcasting to work.
    read_strengths = tf.expand_dims(read_strengths, axis=-1)  # [B, H, 1]
    weighted_distances = read_strengths * cosine_distances  # [B, H, R]

    if top_k:
      # Get top k indices (row indices with top k largest weighted distances).
      top_k_output = tf.nn.top_k(weighted_distances, top_k, sorted=False)
      read_indices = top_k_output.indices  # [B, H, K]

      # Create a sub-memory for each read head with only the top k rows.
      # Each batch_gather is [B, K, M] and the list stacks to [B, H, K, M].
      topk_mem_per_head = [tf.batch_gather(mem_state, ri_this_head)
                           for ri_this_head in tf.unstack(read_indices, axis=1)]
      topk_mem = tf.stack(topk_mem_per_head, axis=1)  # [B, H, K, M]
      topk_scaled_mem = tf.math.l2_normalize(topk_mem, axis=-1)  # [B, H, K, M]

      # Calculate read weights for each head's top k sub-memory.
      expanded_scaled_read_keys = tf.expand_dims(
          scaled_read_keys, axis=2)  # [B, H, 1, M]
      topk_cosine_distances = tf.reduce_sum(
          expanded_scaled_read_keys * topk_scaled_mem, axis=-1)  # [B, H, K]
      topk_weighted_distances = (
          read_strengths * topk_cosine_distances)  # [B, H, K]
      read_weights = tf.nn.softmax(
          topk_weighted_distances, axis=-1)  # [B, H, K]

      # For each head, read using the sub-memories and corresponding weights.
      expanded_weights = tf.expand_dims(read_weights, axis=-1)  # [B, H, K, 1]
      memory_reads = tf.reduce_sum(
          expanded_weights * topk_mem, axis=2)  # [B, H, M]
    else:
      read_weights = tf.nn.softmax(weighted_distances, axis=-1)

      num_rows_memory = mem_state.shape[1]
      all_indices = tf.range(num_rows_memory, dtype=tf.int32)
      all_indices = tf.reshape(all_indices, [1, 1, num_rows_memory])
      read_indices = tf.tile(all_indices, [batch_size, num_read_heads, 1])

      # This is the actual memory access.
      # Note that matmul automatically batch applies for us.
      memory_reads = tf.matmul(read_weights, mem_state)

    read_keys.shape.assert_is_compatible_with(memory_reads.shape)

    read_strengths = tf.squeeze(read_strengths, axis=-1)  # [B, H, 1] -> [B, H]

    return memory_reads, read_weights, read_indices, read_strengths


def _assert_compatible_read_from_memory_inputs(read_keys, read_strengths,
                                               mem_state):
  read_keys.shape.assert_has_rank(3)
  b_shape, h_shape, m_shape = read_keys.shape
  mem_state.shape.assert_has_rank(3)
  r_shape = mem_state.shape[1]

  read_strengths.shape.assert_is_compatible_with(
      tf.TensorShape([b_shape, h_shape]))
  mem_state.shape.assert_is_compatible_with(
      tf.TensorShape([b_shape, r_shape, m_shape]))


def _assert_compatible_memory_reader_input(input_tensors):
  """Asserts MemoryReader's _build has been given the correct shapes."""
  assert len(input_tensors) == 2
  _, mem_state = input_tensors
  mem_state.shape.assert_has_rank(3)
