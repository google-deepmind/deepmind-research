# Copyright 2020 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.random as jnr

from kfac_ferminet_alpha import layers_and_loss_tags
from kfac_ferminet_alpha import loss_functions
from kfac_ferminet_alpha import tag_graph_matcher
from kfac_ferminet_alpha.tests import common


def tagged_autoencoder(all_params, x_in):
  h_in = x_in
  layers_values = []
  for i, params in enumerate(all_params):
    h_out = common.fully_connected_layer(params, h_in)
    h_out = layers_and_loss_tags.register_dense(h_out, h_in, params[0],
                                                params[1],)
    layers_values.append((h_out, h_in))
    # Last layer does not have a nonlinearity
    if i % 4 != 3:
      h_in = jnp.tanh(h_out)
    else:
      h_in = h_out
  h1, _ = loss_functions.register_normal_predictive_distribution(
      h_in, targets=x_in, weight=1.0)
  h2, t2 = loss_functions.register_normal_predictive_distribution(
      h_in, targets=x_in, weight=0.1)
  return [[h1, t2], [h2, t2]]


class TestGraphMatcher(unittest.TestCase):
  """Class for running all of the tests for integrating the systems."""

  def _test_jaxpr(self, init_func, model_func, tagged_model, data_shape):
    data_shape = tuple(data_shape)
    rng_key = jnr.PRNGKey(12345)
    init_key, data_key = jnr.split(rng_key)
    params = init_func(init_key, data_shape)
    data = jnr.normal(data_key, (11,) + data_shape)
    func = tag_graph_matcher.auto_register_tags(model_func, (params, data))
    jaxpr = jax.make_jaxpr(func)(params, data).jaxpr
    tagged_jaxpr = jax.make_jaxpr(tagged_model)(params, data).jaxpr
    self.assertEqual(len(jaxpr.invars), len(tagged_jaxpr.invars))
    self.assertEqual(len(jaxpr.constvars), len(tagged_jaxpr.constvars))
    self.assertEqual(len(jaxpr.outvars), len(tagged_jaxpr.outvars))
    for eq, tagged_eq in zip(jaxpr.eqns, tagged_jaxpr.eqns):
      eq_in_vars = [v for v in eq.invars]
      tagged_in_vars = [v for v in tagged_eq.invars]
      self.assertEqual(len(eq_in_vars), len(tagged_in_vars))
      self.assertEqual(len(eq.outvars), len(tagged_eq.outvars))
      self.assertEqual(eq.primitive, tagged_eq.primitive)
      for variable, t_variable in zip(eq_in_vars + eq.outvars,
                                      tagged_in_vars + tagged_eq.outvars):
        if isinstance(variable, jax.core.Literal):
          self.assertEqual(variable.aval, t_variable.aval)
        else:
          if variable.count != t_variable.count:
            print("0")
          self.assertEqual(variable.count, t_variable.count)

  def test_autoencoder(self):
    self._test_jaxpr(common.init_autoencoder, common.autoencoder,
                     tagged_autoencoder, [784])


if __name__ == "__main__":
  absltest.main()
