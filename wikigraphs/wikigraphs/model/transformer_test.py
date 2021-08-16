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
"""Tests for wikigraphs.model.transformer."""

from absl import logging
from absl.testing import absltest

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from wikigraphs.model import embedding
from wikigraphs.model import transformer as models


def tree_size(nest):
  return sum(x.size for x in jax.tree_util.tree_leaves(nest))


class TransformerXlTest(absltest.TestCase):

  def test_transformer_param_count(self):
    seqs = np.array([[1, 2, 3, 0, 0],
                     [3, 3, 5, 1, 2]], dtype=np.int32)
    x = seqs[:, :-1]
    y = seqs[:, 1:]
    vocab_size = 267_735

    def forward(inputs, labels):
      input_mask = (labels != 0).astype(jnp.float32)
      model = models.TransformerXL(
          vocab_size=vocab_size,
          emb_dim=210,
          num_layers=2,
          num_heads=10,
          dropout_prob=0.0,
          dropout_attn_prob=0.0,
          self_att_init_scale=0.02,
          dense_init_scale=0.02,
          dense_dim=2100,
          cutoffs=(20000, 40000, 200000),  # WikiText-103
          relative_pos_clamp_len=None,
      )
      return model.loss(inputs, labels, mask=input_mask, cache_steps=2)

    init_fn, apply_fn = hk.transform_with_state(forward)
    key = hk.PRNGSequence(8)
    params, state = init_fn(next(key), x, y)
    out, _ = apply_fn(params, state, next(key), x, y)
    loss, metrics = out

    logging.info('loss: %g', loss)
    logging.info('metrics: %r', metrics)

    param_count = tree_size(params)
    self.assertEqual(param_count, 58_704_438)

  def test_transformer_with_extra_runs(self):
    extra = np.array([[1, 1, 0, 0],
                      [2, 2, 2, 2],
                      [3, 3, 3, 0]], dtype=np.int32)
    seqs = np.array([[1, 2, 3, 0, 0],
                     [2, 4, 5, 6, 0],
                     [3, 3, 5, 1, 2]], dtype=np.int32)
    x = seqs[:, :-1]
    y = seqs[:, 1:]
    vocab_size = seqs.max() + 1
    extra_vocab_size = extra.max() + 1

    def forward(inputs, labels, extra):
      input_mask = (labels != 0).astype(jnp.float32)
      extra_mask = (extra != 0).astype(jnp.float32)
      extra = hk.Embed(vocab_size=extra_vocab_size, embed_dim=16)(extra)
      model = models.TransformerXL(
          vocab_size=vocab_size,
          emb_dim=16,
          num_layers=2,
          num_heads=4,
          cutoffs=[],
      )
      return model.loss(inputs, labels, mask=input_mask,
                        extra=extra, extra_mask=extra_mask)

    init_fn, apply_fn = hk.transform_with_state(forward)
    key = hk.PRNGSequence(8)
    params, state = init_fn(next(key), x, y, extra)
    out, _ = apply_fn(params, state, next(key), x, y, extra)
    loss, metrics = out

    logging.info('loss: %g', loss)
    logging.info('metrics: %r', metrics)

  def test_graph_embedding_model_runs(self):
    graph = jraph.GraphsTuple(
        nodes=np.array([[0, 1, 1],
                        [1, 2, 0],
                        [0, 3, 0],
                        [0, 4, 4]], dtype=np.float32),
        edges=np.array([[1, 1],
                        [2, 2],
                        [3, 3]], dtype=np.float32),
        senders=np.array([0, 1, 2], dtype=np.int32),
        receivers=np.array([1, 2, 3], dtype=np.int32),
        n_node=np.array([4], dtype=np.int32),
        n_edge=np.array([3], dtype=np.int32),
        globals=None)
    embed_dim = 3

    def forward(graph):
      return embedding.GraphEmbeddingModel(embed_dim=3, num_layers=2)(graph)

    init_fn, apply_fn = hk.without_apply_rng(hk.transform(forward))
    key = hk.PRNGSequence(8)
    params = init_fn(next(key), graph)
    out = apply_fn(params, graph)

    self.assertEqual(out.nodes.shape, (graph.nodes.shape[0], embed_dim))
    self.assertEqual(out.edges.shape, (graph.edges.shape[0], embed_dim))
    np.testing.assert_array_equal(out.senders, graph.senders)
    np.testing.assert_array_equal(out.receivers, graph.receivers)
    np.testing.assert_array_equal(out.n_node, graph.n_node)

  def test_unpack_and_pad(self):
    x = np.array([1, 1, 2, 2, 2, 3, 4, 4], dtype=np.float32)
    s = np.array([2, 3, 1, 2], dtype=np.int32)

    tensors, mask = models.unpack_and_pad(x, s, pad_size=s.max(), pad_value=0)

    np.testing.assert_array_equal(
        tensors,
        [[1, 1, 0],
         [2, 2, 2],
         [3, 0, 0],
         [4, 4, 0]])
    np.testing.assert_array_equal(
        mask,
        [[1, 1, 0],
         [1, 1, 1],
         [1, 0, 0],
         [1, 1, 0]])

    # [n, 1] tensor
    x = np.array([1, 1, 2, 2, 2, 3, 4, 4], dtype=np.float32)[:, None]
    s = np.array([2, 3, 1, 2], dtype=np.int32)

    tensors, mask = models.unpack_and_pad(x, s, pad_size=s.max(), pad_value=0)

    np.testing.assert_array_equal(
        tensors,
        np.array([[1, 1, 0],
                  [2, 2, 2],
                  [3, 0, 0],
                  [4, 4, 0]])[:, :, None])
    np.testing.assert_array_equal(
        mask,
        [[1, 1, 0],
         [1, 1, 1],
         [1, 0, 0],
         [1, 1, 0]])

  def test_graph_conditioned_transformer_runs(self):
    graphs = jraph.GraphsTuple(
        nodes=np.ones((4, 3), dtype=np.float32),
        edges=np.ones((3, 1), dtype=np.float32),
        senders=np.array([0, 2, 3], dtype=np.int32),
        receivers=np.array([1, 3, 2], dtype=np.int32),
        n_node=np.array([2, 2], dtype=np.int32),
        n_edge=np.array([1, 2], dtype=np.int32),
        globals=None,
        )
    seqs = np.array([[1, 1, 0],
                     [2, 2, 2]], dtype=np.int32)
    vocab_size = seqs.max() + 1
    embed_dim = 8

    x = seqs[:, :-1]
    y = seqs[:, 1:]

    def forward(graphs, inputs, labels):
      graphs = models.GraphEmbeddingModel(embed_dim=embed_dim,
                                          num_layers=2)(graphs)
      extra, extra_mask = models.unpack_and_pad(graphs.nodes,
                                                graphs.n_node,
                                                graphs.n_node.max())
      input_mask = (labels != 0).astype(jnp.float32)
      transformer = models.TransformerXL(vocab_size=vocab_size,
                                         emb_dim=embed_dim,
                                         num_layers=2,
                                         num_heads=4,
                                         cutoffs=[])
      return transformer.loss(inputs, labels, mask=input_mask, extra=extra,
                              extra_mask=extra_mask)

    init_fn, apply_fn = hk.transform_with_state(forward)
    key = hk.PRNGSequence(8)
    params, state = init_fn(next(key), graphs, x, y)
    out, _ = apply_fn(params, state, next(key), graphs, x, y)
    loss, metrics = out

    logging.info('loss: %g', loss)
    logging.info('metrics: %r', metrics)

  def test_graph_conditioned_transformer_learns(self):
    graphs = jraph.GraphsTuple(
        nodes=np.ones((4, 3), dtype=np.float32),
        edges=np.ones((3, 1), dtype=np.float32),
        senders=np.array([0, 2, 3], dtype=np.int32),
        receivers=np.array([1, 3, 2], dtype=np.int32),
        n_node=np.array([2, 2], dtype=np.int32),
        n_edge=np.array([1, 2], dtype=np.int32),
        globals=None,
        )
    seqs = np.array([[1, 2, 2, 0],
                     [1, 3, 3, 3]], dtype=np.int32)
    vocab_size = seqs.max() + 1
    embed_dim = 8
    max_graph_size = graphs.n_node.max()

    logging.info('Training seqs: %r', seqs)

    x = seqs[:, :-1]
    y = seqs[:, 1:]

    def model_fn(vocab_size, embed_dim):
      return models.Graph2TextTransformer(
          vocab_size=vocab_size,
          emb_dim=embed_dim,
          num_layers=2,
          num_heads=4,
          cutoffs=[],
          gnn_embed_dim=embed_dim,
          gnn_num_layers=2)

    def forward(graphs, inputs, labels, max_graph_size):
      input_mask = (labels != 0).astype(jnp.float32)
      return model_fn(vocab_size, embed_dim).loss(
          graphs, max_graph_size, False, inputs, labels, mask=input_mask)

    init_fn, apply_fn = hk.transform_with_state(forward)
    rng = hk.PRNGSequence(8)
    params, state = init_fn(next(rng), graphs, x, y, max_graph_size)

    def apply(*args, **kwargs):
      out, state = apply_fn(*args, **kwargs)
      return out[0], (out[1], state)
    apply = jax.jit(apply, static_argnums=6)

    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale(-1e-3))
    opt_state = optimizer.init(params)
    for i in range(500):
      (loss, model_state), grad = jax.value_and_grad(apply, has_aux=True)(
          params, state, next(rng), graphs, x, y, max_graph_size)
      metrics, state = model_state
      updates, opt_state = optimizer.update(grad, opt_state, params)
      params = optax.apply_updates(params, updates)
      if (i + 1) % 100 == 0:
        logging.info(
            'Step %d, %r', i + 1, {k: float(v) for k, v in metrics.items()})
    logging.info('Loss: %.8f', loss)
    self.assertLess(loss, 1.0)

  def test_bow_transformer_runs(self):
    bow = np.array([[0, 0, 1, 0, 2, 0, 0, 1],
                    [0, 1, 0, 0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.int32)
    seqs = np.array([[1, 2, 3, 0, 0],
                     [2, 4, 5, 6, 0],
                     [3, 3, 5, 1, 2]], dtype=np.int32)
    x = seqs[:, :-1]
    y = seqs[:, 1:]
    vocab_size = seqs.max() + 1

    def forward(bow, inputs, labels):
      model = models.Bow2TextTransformer(
          vocab_size=vocab_size,
          emb_dim=16,
          num_layers=2,
          num_heads=4,
          cutoffs=[])
      return model.loss(bow, inputs, labels)

    init_fn, apply_fn = hk.transform_with_state(forward)
    key = hk.PRNGSequence(8)
    params, state = init_fn(next(key), bow, x, y)
    out, _ = apply_fn(params, state, next(key), bow, x, y)
    loss, metrics = out

    logging.info('loss: %g', loss)
    logging.info('metrics: %r', metrics)

  def test_bow_transformer_learns(self):
    bow = np.array([[0, 0, 1, 0, 2, 0, 0, 1],
                    [0, 1, 0, 0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.int32)
    seqs = np.array([[1, 2, 2, 3, 0, 0],
                     [1, 2, 4, 5, 6, 0],
                     [1, 3, 3, 5, 4, 2]], dtype=np.int32)
    x = seqs[:, :-1]
    y = seqs[:, 1:]
    vocab_size = seqs.max() + 1

    def model_fn():
      return models.Bow2TextTransformer(
          vocab_size=vocab_size,
          emb_dim=16,
          num_layers=2,
          num_heads=4,
          cutoffs=[])

    def loss_fn(bow, inputs, labels):
      mask = (labels != 0).astype(jnp.float32)
      return model_fn().loss(bow, inputs, labels, mask=mask)

    init_fn, apply_fn = hk.transform_with_state(loss_fn)
    key = hk.PRNGSequence(8)
    params, state = init_fn(next(key), bow, x, y)

    def apply(*args, **kwargs):
      out, state = apply_fn(*args, **kwargs)
      return out[0], (out[1], state)
    value_and_grad = jax.jit(jax.value_and_grad(apply, has_aux=True))

    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale(-1e-3))
    opt_state = optimizer.init(params)
    for i in range(800):
      (loss, model_state), grad = value_and_grad(
          params, state, next(key), bow, x, y)
      metrics, state = model_state
      updates, opt_state = optimizer.update(grad, opt_state, params)
      params = optax.apply_updates(params, updates)
      if (i + 1) % 100 == 0:
        logging.info('Step %d, %r', i + 1,
                     {k: float(v) for k, v in metrics.items()})
    logging.info('Loss: %.8f', loss)
    self.assertLess(loss, 0.1)


if __name__ == '__main__':
  absltest.main()
