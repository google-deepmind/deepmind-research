# Copyright 2019 Deepmind Technologies Limited.
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

"""Tests for graph_model."""

import itertools
from absl.testing import parameterized

from graph_nets import graphs
import numpy as np
import tensorflow.compat.v1 as tf

from glassy_dynamics import graph_model


class GraphModelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes a small tractable test (particle) system."""
    super(GraphModelTest, self).setUp()
    # Fixes random seed to ensure deterministic outputs.
    tf.random.set_random_seed(1234)

    # In this test we use a small tractable set of particles covering all corner
    # cases:
    # a) eight particles with different types,
    # b) periodic box is not cubic,
    # c) three disjoint cluster of particles separated by a threshold > 2,
    # d) first two clusters overlap with the periodic boundary,
    # e) first cluster is not fully connected,
    # f) second cluster is fully connected,
    # g) and third cluster is a single isolated particle.
    #
    # The formatting of the code below separates the three clusters by
    # adding linebreaks after each cluster.
    self._positions = np.array(
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 9.0],
         [0.0, 5.0, 0.0], [0.0, 5.0, 1.0], [3.0, 5.0, 0.0],
         [2.0, 3.0, 3.0]])
    self._types = np.array([0.0, 0.0, 1.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0])
    self._box = np.array([4.0, 10.0, 10.0])

    # Creates the corresponding graph elements, assuming a threshold of 2 and
    # the conventions described in `graph_nets.graphs`.
    self._edge_threshold = 2
    self._nodes = np.array(
        [[0.0], [0.0], [1.0], [0.0],
         [0.0], [1.0], [0.0],
         [0.0]])
    self._edges = np.array(
        [[0.0, 0.0, 0.0], [-1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, -1.0],
         [1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, -1.0],
         [0.0, -1.5, 0.0], [0.0, 0.0, 0.0], [0.0, -1.5, -1.0],
         [0.0, 0.0, 1.0], [-1.5, 0.0, 1.0], [0.0, 1.5, 1.0], [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, -1.0],
         [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]])

    self._receivers = np.array(
        [0, 1, 2, 3, 0, 1, 3, 0, 2, 3, 0, 1, 2, 3,
         4, 5, 6, 4, 5, 6, 4, 5, 6,
         7])
    self._senders = np.array(
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3,
         4, 4, 4, 5, 5, 5, 6, 6, 6,
         7])

  def _get_graphs_tuple(self):
    """Returns a GraphsTuple containing a graph based on the test system."""
    return graphs.GraphsTuple(
        nodes=tf.constant(self._nodes, dtype=tf.float32),
        edges=tf.constant(self._edges, dtype=tf.float32),
        globals=tf.constant(np.array([[0.0]]), dtype=tf.float32),
        receivers=tf.constant(self._receivers, dtype=tf.int32),
        senders=tf.constant(self._senders, dtype=tf.int32),
        n_node=tf.constant([len(self._nodes)], dtype=tf.int32),
        n_edge=tf.constant([len(self._edges)], dtype=tf.int32))

  def test_make_graph_from_static_structure(self):
    graphs_tuple_op = graph_model.make_graph_from_static_structure(
        tf.constant(self._positions, dtype=tf.float32),
        tf.constant(self._types, dtype=tf.int32),
        tf.constant(self._box, dtype=tf.float32),
        self._edge_threshold)
    graphs_tuple = self.evaluate(graphs_tuple_op)
    self.assertLen(self._nodes, graphs_tuple.n_node)
    self.assertLen(self._edges, graphs_tuple.n_edge)
    np.testing.assert_almost_equal(graphs_tuple.nodes, self._nodes)
    np.testing.assert_equal(graphs_tuple.senders, self._senders)
    np.testing.assert_equal(graphs_tuple.receivers, self._receivers)
    np.testing.assert_almost_equal(graphs_tuple.globals, np.array([[0.0]]))
    np.testing.assert_almost_equal(graphs_tuple.edges, self._edges)

  def _is_equal_up_to_rotation(self, x, y):
    for axes in itertools.permutations([0, 1, 2]):
      for mirrors in itertools.product([1, -1], repeat=3):
        if np.allclose(x, y[:, axes] * mirrors):
          return True
    return False

  def test_apply_random_rotation(self):
    graphs_tuple = self._get_graphs_tuple()
    rotated_graphs_tuple_op = graph_model.apply_random_rotation(graphs_tuple)
    rotated_graphs_tuple = self.evaluate(rotated_graphs_tuple_op)
    np.testing.assert_almost_equal(rotated_graphs_tuple.nodes, self._nodes)
    np.testing.assert_almost_equal(rotated_graphs_tuple.senders, self._senders)
    np.testing.assert_almost_equal(
        rotated_graphs_tuple.receivers, self._receivers)
    np.testing.assert_almost_equal(
        rotated_graphs_tuple.globals, np.array([[0.0]]))
    self.assertTrue(self._is_equal_up_to_rotation(rotated_graphs_tuple.edges,
                                                  self._edges))

  @parameterized.named_parameters(('no_propagation', 0, (30,)),
                                  ('multi_propagation', 5, (15,)),
                                  ('multi_layer', 1, (20, 30)))
  def test_GraphModel(self, n_recurrences, mlp_sizes):
    graphs_tuple = self._get_graphs_tuple()
    output_op = graph_model.GraphBasedModel(n_recurrences=n_recurrences,
                                            mlp_sizes=mlp_sizes)(graphs_tuple)
    self.assertListEqual(output_op.shape.as_list(), [len(self._types)])
    # Tests if the model runs without crashing.
    with self.session():
      tf.global_variables_initializer().run()
      output_op.eval()


if __name__ == '__main__':
  tf.test.main()
