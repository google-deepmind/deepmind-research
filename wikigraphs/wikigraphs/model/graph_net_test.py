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
"""Tests for wikigraphs.model.graph_net."""

from absl import logging
from absl.testing import absltest
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from wikigraphs.model import graph_net as gn


class GraphNetTest(absltest.TestCase):

  def test_node_classification(self):
    # If node has more than 2 neighbors --> class 1, otherwise class 0.
    # Graph structure:
    # 1         4
    # | \     / |
    # |  0 - 3  |
    # | /     \ |
    # 2         5

    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 3],
        [3, 4],
        [4, 5],
        [5, 3],
    ], dtype=np.int32)

    n_node = edges.max() + 1
    n_edge = edges.shape[0]
    g = jraph.GraphsTuple(
        senders=edges[:, 0],
        receivers=edges[:, 1],
        edges=np.ones((edges.shape[0], 1), dtype=np.float32),
        nodes=np.ones((n_node, 1), dtype=np.float32),
        n_node=np.array([n_node], dtype=np.int32),
        n_edge=np.array([n_edge], dtype=np.int32),
        globals=None)
    g = gn.add_reverse_edges(g)
    targets = np.array([1, 0, 0, 1, 0, 0], dtype=np.int32)
    n_classes = 2

    def forward(graph, targets):
      model = gn.SimpleGraphNet(num_layers=5, layer_norm=False)
      graph = model(graph)
      nodes = graph.nodes
      logits = hk.Linear(n_classes)(nodes)
      pred = logits.argmax(axis=-1)
      accuracy = (pred == targets).mean()
      targets = jax.nn.one_hot(targets, n_classes, dtype=jnp.float32)
      return -jnp.mean(jnp.sum(
          jax.nn.log_softmax(logits, axis=-1) * targets, axis=-1)), accuracy

    init_fn, apply_fn = hk.without_apply_rng(hk.transform(forward))
    rng = hk.PRNGSequence(0)
    params = init_fn(next(rng), g, targets)

    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale(-1e-3))
    opt_state = optimizer.init(params)
    apply_fn = jax.jit(apply_fn)
    for i in range(500):
      (loss, acc), grad = jax.value_and_grad(apply_fn,
                                             has_aux=True)(params, g, targets)
      updates, opt_state = optimizer.update(grad, opt_state, params)
      params = optax.apply_updates(params, updates)
      if (i + 1) % 100 == 0:
        logging.info('Step %d, loss %.8f, accuracy %.4f', i + 1, loss, acc)
    self.assertLess(loss, 0.01)
    self.assertEqual(acc, 1.0)

  def test_pad_size(self):
    self.assertEqual(gn.pad_size(1), 1)
    self.assertEqual(gn.pad_size(5), 8)
    self.assertEqual(gn.pad_size(7), 8)
    self.assertEqual(gn.pad_size(101), 128)

  def test_pad_graphs(self):
    # No new edges to add
    graphs = jraph.GraphsTuple(
        nodes=np.arange(6)[:, None],
        edges=np.arange(4)[:, None],
        senders=np.array([0, 2, 3, 4]),
        receivers=np.array([1, 3, 4, 5]),
        n_node=np.array([2, 4]),
        n_edge=np.array([1, 3]),
        globals=None)
    padded = gn.pad_graphs(graphs)
    np.testing.assert_array_equal(
        padded.nodes,
        np.array([0, 1, 2, 3, 4, 5, 0, 0])[:, None])
    np.testing.assert_array_equal(padded.edges, graphs.edges)
    np.testing.assert_array_equal(padded.senders, graphs.senders)
    np.testing.assert_array_equal(padded.receivers, graphs.receivers)
    np.testing.assert_array_equal(padded.n_node, [2, 4, 2])
    np.testing.assert_array_equal(padded.n_edge, [1, 3, 0])

    # Add just a single default node
    graphs = jraph.GraphsTuple(
        nodes=np.arange(7)[:, None],
        edges=np.arange(5)[:, None],
        senders=np.array([0, 2, 3, 5, 6]),
        receivers=np.array([1, 3, 4, 6, 5]),
        n_node=np.array([2, 3, 2]),
        n_edge=np.array([1, 2, 2]),
        globals=None)
    padded = gn.pad_graphs(graphs)
    np.testing.assert_array_equal(
        padded.nodes,
        np.array([0, 1, 2, 3, 4, 5, 6, 0])[:, None])
    np.testing.assert_array_equal(
        padded.edges,
        np.array([0, 1, 2, 3, 4, 0, 0, 0])[:, None])
    np.testing.assert_array_equal(
        padded.senders,
        [0, 2, 3, 5, 6, 7, 7, 7])
    np.testing.assert_array_equal(
        padded.receivers,
        [1, 3, 4, 6, 5, 7, 7, 7])
    np.testing.assert_array_equal(
        padded.n_node, [2, 3, 2, 1])
    np.testing.assert_array_equal(
        padded.n_edge, [1, 2, 2, 3])

    # Num. nodes is a power of 2 but we still pad at least one extra node
    graphs = jraph.GraphsTuple(
        nodes=np.arange(8)[:, None],
        edges=np.arange(5)[:, None],
        senders=np.array([0, 2, 3, 5, 6]),
        receivers=np.array([1, 3, 4, 6, 7]),
        n_node=np.array([2, 3, 3]),
        n_edge=np.array([1, 2, 2]),
        globals=None)
    padded = gn.pad_graphs(graphs)
    np.testing.assert_array_equal(
        padded.nodes,
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0])[:, None])
    np.testing.assert_array_equal(
        padded.edges,
        np.array([0, 1, 2, 3, 4, 0, 0, 0])[:, None])
    np.testing.assert_array_equal(
        padded.senders,
        [0, 2, 3, 5, 6, 8, 8, 8])
    np.testing.assert_array_equal(
        padded.receivers,
        [1, 3, 4, 6, 7, 8, 8, 8])
    np.testing.assert_array_equal(
        padded.n_node, [2, 3, 3, 8])
    np.testing.assert_array_equal(
        padded.n_edge, [1, 2, 2, 3])

  def test_batch_graphs_by_device(self):
    # batch 4 graphs for 2 devices
    num_devices = 2
    graphs = [
        jraph.GraphsTuple(
            nodes=np.arange(2)[:, None],
            edges=np.arange(2)[:, None],
            senders=np.array([0, 1]),
            receivers=np.array([1, 0]),
            n_node=np.array([2]),
            n_edge=np.array([2]),
            globals=None),
        jraph.GraphsTuple(
            nodes=np.arange(3)[:, None],
            edges=np.arange(1)[:, None],
            senders=np.array([2]),
            receivers=np.array([0]),
            n_node=np.array([3]),
            n_edge=np.array([1]),
            globals=None),
        jraph.GraphsTuple(
            nodes=np.arange(4)[:, None],
            edges=np.arange(2)[:, None],
            senders=np.array([1, 0]),
            receivers=np.array([2, 3]),
            n_node=np.array([4]),
            n_edge=np.array([2]),
            globals=None),
        jraph.GraphsTuple(
            nodes=np.arange(5)[:, None],
            edges=np.arange(3)[:, None],
            senders=np.array([2, 1, 3]),
            receivers=np.array([1, 4, 0]),
            n_node=np.array([5]),
            n_edge=np.array([3]),
            globals=None),
    ]
    batched = gn.batch_graphs_by_device(graphs, num_devices)
    self.assertLen(batched, num_devices)
    np.testing.assert_array_equal(
        batched[0].nodes,
        np.array([0, 1, 0, 1, 2])[:, None])
    np.testing.assert_array_equal(
        batched[0].edges,
        np.array([0, 1, 0])[:, None])
    np.testing.assert_array_equal(
        batched[0].senders,
        np.array([0, 1, 4]))
    np.testing.assert_array_equal(
        batched[0].receivers,
        np.array([1, 0, 2]))
    np.testing.assert_array_equal(
        batched[0].n_node,
        np.array([2, 3]))
    np.testing.assert_array_equal(
        batched[0].n_edge,
        np.array([2, 1]))
    np.testing.assert_array_equal(
        batched[1].nodes,
        np.array([0, 1, 2, 3, 0, 1, 2, 3, 4])[:, None])
    np.testing.assert_array_equal(
        batched[1].edges,
        np.array([0, 1, 0, 1, 2])[:, None])
    np.testing.assert_array_equal(
        batched[1].senders,
        np.array([1, 0, 6, 5, 7]))
    np.testing.assert_array_equal(
        batched[1].receivers,
        np.array([2, 3, 5, 8, 4]))
    np.testing.assert_array_equal(
        batched[1].n_node,
        np.array([4, 5]))
    np.testing.assert_array_equal(
        batched[1].n_edge,
        np.array([2, 3]))

  def test_pad_graphs_by_device(self):
    graphs = [
        jraph.GraphsTuple(
            nodes=np.arange(5)[:, None],     # pad to 8
            edges=np.arange(3)[:, None],     # pad to 4
            senders=np.array([0, 1, 4]),     # pad to 4
            receivers=np.array([1, 0, 2]),   # pad to 4
            n_node=np.array([2, 3]),         # pad to 3
            n_edge=np.array([2, 1]),         # pad to 3
            globals=None),
        jraph.GraphsTuple(
            nodes=np.arange(4)[:, None],     # pad to 8
            edges=np.arange(1)[:, None],     # pad to 4
            senders=np.array([1]),           # pad to 4
            receivers=np.array([0]),         # pad to 4
            n_node=np.array([2, 2]),         # pad to 3
            n_edge=np.array([1, 0]),         # pad to 3
            globals=None),
    ]
    padded = gn.pad_graphs_by_device(graphs)
    np.testing.assert_array_equal(
        padded.nodes,
        np.array([0, 1, 2, 3, 4, 0, 0, 0,
                  0, 1, 2, 3, 0, 0, 0, 0])[:, None])
    np.testing.assert_array_equal(
        padded.edges,
        np.array([0, 1, 2, 0, 0, 0, 0, 0])[:, None])
    np.testing.assert_array_equal(
        padded.senders,
        np.array([0, 1, 4, 5, 1, 4, 4, 4]))
    np.testing.assert_array_equal(
        padded.receivers,
        np.array([1, 0, 2, 5, 0, 4, 4, 4]))
    np.testing.assert_array_equal(
        padded.n_node,
        np.array([2, 3, 3, 2, 2, 4]))
    np.testing.assert_array_equal(
        padded.n_edge,
        np.array([2, 1, 1, 1, 0, 3]))


if __name__ == '__main__':
  absltest.main()
