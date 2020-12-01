# Copyright 2020 DeepMind Technologies Limited.
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

"""Tests for the Geometric Manifold Component Estimator (GEOMANCER)."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from geomancer import geomancer


class GeomancerTest(parameterized.TestCase):

  @parameterized.parameters(
      {'zero_trace': False},
      {'zero_trace': True})
  def test_sym_op(self, zero_trace):
    """sym_op on tril(X) gives same result as QXQ' for symmetric X?"""
    n = 5
    x = np.random.randn(n, n)
    x += x.T
    if zero_trace:
      np.fill_diagonal(x, np.diag(x)-np.trace(x)/n)
    q, _ = np.linalg.qr(np.random.randn(n, n))
    sym_q = geomancer.sym_op(q, zero_trace=zero_trace)
    tril_x = x[np.tril_indices(n)]
    if zero_trace:
      tril_x = tril_x[:-1]
    vec_y = sym_q @ tril_x
    y = q @ x @ q.T
    y_ = geomancer.vec_to_sym(vec_y, n, zero_trace=zero_trace)
    np.testing.assert_allclose(y_, y)

  def test_ffdiag(self):
    k = 2
    n = 5
    w, _ = np.linalg.qr(np.random.randn(n, n))
    psi = np.random.randn(k, n)
    a = np.zeros((k, n, n))
    for i in range(k):
      a[i] = w @ np.diag(psi[i]) @ w.T
    w_ = geomancer.ffdiag(a)
    for i in range(k):
      x = w_ @ a[i] @ w_.T
      diag = np.diag(x).copy()
      np.fill_diagonal(x, 1.0)
      # check that x is diagonal
      np.testing.assert_allclose(x, np.eye(n), rtol=1e-10, atol=1e-10)
      self.assertTrue(np.all(np.min(
          np.abs(diag[None, :] - psi[i][:, None]), axis=0) < 1e-10))

  def test_make_nearest_neighbor_graph(self):
    n = 100
    # make points on a circle
    data = np.zeros((n, 2))
    for i in range(n):
      data[i, 0] = np.sin(i*2*np.pi/n)
      data[i, 1] = np.cos(i*2*np.pi/n)
    graph = geomancer.make_nearest_neighbors_graph(data, 4, n=10)
    for i in range(n):
      self.assertLen(graph.rows[i], 4)
      self.assertIn((i+1) % n, graph.rows[i])
      self.assertIn((i+2) % n, graph.rows[i])
      self.assertIn((i-1) % n, graph.rows[i])
      self.assertIn((i-2) % n, graph.rows[i])


if __name__ == '__main__':
  absltest.main()
