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

"""Tests for hierarchical_attention.htm_attention."""

from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import numpy as np

from hierarchical_transformer_memory.hierarchical_attention import htm_attention


def _build_queries_and_memory(query_length, num_memories, mem_chunk_size,
                              batch_size=2, embedding_size=12):
  """Builds dummy queries + memory contents for tests."""
  queries = np.random.random([batch_size, query_length, embedding_size])
  memory_contents = np.random.random(
      [batch_size, num_memories, mem_chunk_size, embedding_size])
  # summary key = average across chunk
  memory_keys = np.mean(memory_contents, axis=2)
  # to accumulate newest memories before writing
  memory_accumulator = np.zeros_like(memory_contents[:, -1, :, :])
  memory = htm_attention.HierarchicalMemory(
      keys=memory_keys,
      contents=memory_contents,
      accumulator=memory_accumulator,
      steps_since_last_write=np.zeros([batch_size,], dtype=np.int32))
  return queries, memory


class HierarchicalAttentionTest(parameterized.TestCase):

  @parameterized.parameters([
      {
          'query_length': 1,
          'num_memories': 7,
          'mem_chunk_size': 5,
          'mem_k': 4,
      },
      {
          'query_length': 9,
          'num_memories': 7,
          'mem_chunk_size': 5,
          'mem_k': 4,
      },
  ])
  @hk.testing.transform_and_run
  def test_output_shapes(self, query_length, num_memories, mem_chunk_size,
                         mem_k):
    np.random.seed(0)
    batch_size = 2
    embedding_size = 12
    num_heads = 3
    queries, memory = _build_queries_and_memory(
        query_length=query_length, num_memories=num_memories,
        mem_chunk_size=mem_chunk_size, embedding_size=embedding_size)
    hm_att = htm_attention.HierarchicalMemoryAttention(
        feature_size=embedding_size,
        k=mem_k,
        num_heads=num_heads)
    results = hm_att(queries, memory)
    self.assertEqual(results.shape,
                     (batch_size, query_length, embedding_size))
    self.assertTrue(np.all(np.isfinite(results)))

  @hk.testing.transform_and_run
  def test_masking(self):
    np.random.seed(0)
    batch_size = 2
    embedding_size = 12
    num_heads = 3
    query_length = 5
    num_memories = 7
    mem_chunk_size = 6
    mem_k = 4
    queries, memory = _build_queries_and_memory(
        query_length=query_length, num_memories=num_memories,
        mem_chunk_size=mem_chunk_size, embedding_size=embedding_size)
    hm_att = htm_attention.HierarchicalMemoryAttention(
        feature_size=embedding_size,
        k=mem_k,
        num_heads=num_heads)
    # get a random boolean mask
    mask = np.random.binomial(
        1, 0.5, [batch_size, query_length, num_memories]).astype(bool)
    results = hm_att(queries, memory, hm_mask=mask)
    self.assertEqual(results.shape,
                     (batch_size, query_length, embedding_size))
    self.assertTrue(np.all(np.isfinite(results)))


if __name__ == '__main__':
  absltest.main()
