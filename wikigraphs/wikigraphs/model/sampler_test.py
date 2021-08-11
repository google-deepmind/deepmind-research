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
"""Tests for wikigraphs.model.sampler."""

from absl.testing import absltest
import jraph
import numpy as np

from wikigraphs.model import sampler
from wikigraphs.model import transformer as models


class SamplerTest(absltest.TestCase):

  def test_uncond_sampler_runs(self):
    prompt = np.array([[0, 1, 2, -1, -1],
                       [0, 1, 2, -1, -1]], dtype=np.int32)
    vocab_size = prompt.max() + 1
    bos_token = 0
    memory_size = 2
    params = None

    def model_fn(x):
      return models.TransformerXL(
          vocab_size=vocab_size,
          emb_dim=8,
          num_layers=2,
          num_heads=4,
          cutoffs=[])(x, is_training=False, cache_steps=memory_size)

    uncond_sampler = sampler.TransformerXLSampler(model_fn)
    sample = uncond_sampler.sample(params, prompt)
    self.assertTrue((sample[:, 0] == bos_token).all())
    self.assertTrue((sample != -1).all())
    self.assertEqual(sample.shape, prompt.shape)
    sample2 = uncond_sampler.sample(params, prompt)
    self.assertTrue((sample2[:, 0] == bos_token).all())
    self.assertTrue((sample2 != -1).all())
    self.assertEqual(sample2.shape, prompt.shape)
    self.assertTrue((sample != sample2).any())

  def test_bow2text_sampler_runs(self):
    bow = np.array([[0, 0, 1, 0, 2, 0, 0, 1],
                    [0, 1, 0, 0, 1, 0, 1, 0]], dtype=np.int32)
    prompt = np.array([[0, 1, 2, -1, -1, -1],
                       [0, 1, 2, -1, -1, -1]], dtype=np.int32)
    vocab_size = prompt.max() + 1
    bos_token = 0
    memory_size = 2
    params = None

    def model_fn(bow, x):
      return models.Bow2TextTransformer(
          vocab_size=vocab_size,
          emb_dim=16,
          num_layers=2,
          num_heads=4,
          cutoffs=[])(bow, x, is_training=False, cache_steps=memory_size)

    bow_sampler = sampler.Bow2TextTransformerSampler(model_fn)
    sample = bow_sampler.sample(params, prompt, bow)
    self.assertTrue((sample[:, 0] == bos_token).all())
    self.assertTrue((sample != -1).all())
    self.assertEqual(sample.shape, prompt.shape)
    sample2 = bow_sampler.sample(params, prompt, bow)
    self.assertTrue((sample2[:, 0] == bos_token).all())
    self.assertTrue((sample2 != -1).all())
    self.assertEqual(sample2.shape, prompt.shape)
    self.assertTrue((sample != sample2).any())

  def test_graph2text_sampler_runs(self):
    graphs = jraph.GraphsTuple(
        nodes=np.ones((4, 3), dtype=np.float32),
        edges=np.ones((3, 1), dtype=np.float32),
        senders=np.array([0, 2, 3], dtype=np.int32),
        receivers=np.array([1, 3, 2], dtype=np.int32),
        n_node=np.array([2, 2], dtype=np.int32),
        n_edge=np.array([1, 2], dtype=np.int32),
        globals=None,
        )
    prompt = np.array([[0, 1, 2, -1, -1, -1],
                       [0, 1, 2, -1, -1, -1]], dtype=np.int32)
    vocab_size = prompt.max() + 1
    bos_token = 0
    memory_size = 2
    params = None

    def model_fn(graphs, max_graph_size, x):
      return models.Graph2TextTransformer(
          vocab_size=vocab_size,
          emb_dim=8,
          num_layers=2,
          num_heads=4,
          cutoffs=[],
          gnn_embed_dim=8,
          gnn_num_layers=2)(
              graphs, max_graph_size, True, x,
              is_training=False, cache_steps=memory_size)

    graph_sampler = sampler.Graph2TextTransformerSampler(model_fn)
    sample = graph_sampler.sample(params, prompt, graphs)
    self.assertTrue((sample[:, 0] == bos_token).all())
    self.assertTrue((sample != -1).all())
    self.assertEqual(sample.shape, prompt.shape)
    sample2 = graph_sampler.sample(params, prompt, graphs)
    self.assertTrue((sample2[:, 0] == bos_token).all())
    self.assertTrue((sample2 != -1).all())
    self.assertEqual(sample2.shape, prompt.shape)
    self.assertTrue((sample != sample2).any())


if __name__ == '__main__':
  absltest.main()
