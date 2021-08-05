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
from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.random as jnr
import jax.test_util as jtu

from kfac_ferminet_alpha import loss_functions
from kfac_ferminet_alpha import tag_graph_matcher as tgm
from kfac_ferminet_alpha import tracer
from kfac_ferminet_alpha import utils
from kfac_ferminet_alpha.tests import common


def autoencoder_aux(all_aux, all_params, x_in):
  h_in = x_in
  layers_values = []
  for i, (params, aux) in enumerate(zip(all_params, all_aux)):
    h_out = common.fully_connected_layer(params, h_in + aux[1]) + aux[0]
    layers_values.append((h_out, h_in))
    # Last layer does not have a nonlinearity
    if i % 4 != 3:
      h_in = jnp.tanh(h_out)
    else:
      h_in = h_out
  h1, _ = loss_functions.register_normal_predictive_distribution(h_in, x_in)
  h2, _ = loss_functions.register_normal_predictive_distribution(
      h_in, targets=x_in, weight=0.1)
  l1 = (h1 - x_in)**2 + jnp.log(jnp.pi) / 2
  l2 = (h2 - x_in)**2 + jnp.log(jnp.pi) / 2
  return [l1, l2 * 0.1], layers_values


class TestTracer(jtu.JaxTestCase):
  """Class for running all of the tests for integrating the systems."""

  @staticmethod
  def generate_data(init_func, func, data_shape, rng_key):
    n = 3

    rng_key, key = jnr.split(rng_key)
    params = init_func(key, data_shape)
    rng_key, key = jnr.split(rng_key)
    p_tangents = init_func(key, data_shape)
    rng_key, key = jnr.split(rng_key)
    data = jnr.normal(key, [n] + data_shape)

    loss_vals, layer_vals = func(params, data)
    h = layer_vals[-1][0]
    keys = jnr.split(key, len(loss_vals))
    h_tangents = tuple(jnr.normal(key, shape=h.shape) for key in keys)

    return params, data, p_tangents, h_tangents

  def assertStructureAllClose(self, x, y, **kwargs):
    x_v, x_tree = jax.tree_flatten(x)
    y_v, y_tree = jax.tree_flatten(y)
    self.assertEqual(x_tree, y_tree)
    for xi, yi in zip(x_v, y_v):
      self.assertEqual(xi.shape, yi.shape)
      self.assertAllClose(xi, yi, check_dtypes=True, **kwargs)

  def test_tacer_jvp(self):
    init_func = common.init_autoencoder
    func = common.autoencoder
    data_shape = [784]
    rng_key = jnr.PRNGKey(12345)
    params, data, p_tangents, _ = self.generate_data(init_func, func,
                                                     data_shape, rng_key)

    def no_data_func(args):
      outputs = func(args, data)
      return outputs[0], outputs[1][-1][0]

    # True computation
    (primals_out, tangents_out) = jax.jvp(no_data_func, [params], [p_tangents])
    loss_vals, _ = primals_out
    _, h_tangents = tangents_out
    loss_tangents = ((h_tangents,),) * len(loss_vals)
    # Tracer computation
    tracer_jvp = tracer.trace_losses_matrix_vector_jvp(func)
    tracer_losses, tracer_loss_tangents = tracer_jvp((params, data), p_tangents)
    tracer_losses = [loss.evaluate(None) for loss in tracer_losses]

    self.assertStructureAllClose(loss_vals, tracer_losses)
    self.assertStructureAllClose(loss_tangents, tracer_loss_tangents)

  def test_tracer_vjp(self):
    init_func = common.init_autoencoder
    func = common.autoencoder
    data_shape = [784]
    rng_key = jnr.PRNGKey(12345)
    params, data, _, h_tangents = self.generate_data(init_func, func,
                                                     data_shape, rng_key)

    def no_data_func(args):
      outputs = func(args, data)
      return outputs[0], outputs[1][-1][0]

    # True computation
    (loss_vals, _), vjp_func = jax.vjp(no_data_func, params)
    loss_tangents = jax.tree_map(jnp.zeros_like, loss_vals)
    summed_h_tangents = sum(jax.tree_flatten(h_tangents)[0])
    p_tangents = vjp_func((loss_tangents, summed_h_tangents))
    # Tracer computation
    trace_vjp = tracer.trace_losses_matrix_vector_vjp(func)
    tracer_losses, tracer_vjp_func = trace_vjp(params, data)
    tracer_losses = [loss.evaluate(None) for loss in tracer_losses]
    tracer_p_tangents = tracer_vjp_func(h_tangents)

    self.assertStructureAllClose(loss_vals, tracer_losses)
    self.assertStructureAllClose(p_tangents, tracer_p_tangents, atol=3e-6)

  def test_tracer_hvp(self):
    init_func = common.init_autoencoder
    func = common.autoencoder
    data_shape = [784]
    rng_key = jnr.PRNGKey(12345)
    params, data, p_tangents, _ = self.generate_data(init_func, func,
                                                     data_shape, rng_key)

    def no_data_func(args):
      outputs = func(args, data)
      return sum(jax.tree_map(jnp.sum, outputs[0]))

    # True computation
    grad_func = jax.grad(no_data_func)

    def grad_time_tangents(args):
      return utils.inner_product(grad_func(args), p_tangents)

    hvp = jax.grad(grad_time_tangents)
    hvp_vectors = hvp(params)
    # Tracer computation
    tracer_hvp = tracer.trace_losses_matrix_vector_hvp(func)
    tracer_hvp_vectors = tracer_hvp((params, data), p_tangents)

    self.assertStructureAllClose(hvp_vectors, tracer_hvp_vectors, atol=1e-4)

  def test_trace_estimator(self):
    init_func = common.init_autoencoder
    func = common.autoencoder
    aux_func = autoencoder_aux
    data_shape = [784]
    rng_key = jnr.PRNGKey(12345)
    params, data, _, h_tangents = self.generate_data(init_func, func,
                                                     data_shape, rng_key)

    def aux_last_layer(aux, args):
      outs = aux_func(aux, args, data)
      return outs[1][-1][0]

    # True computation
    loss_vals, layer_vals = func(params, data)
    aux_vals = jax.tree_map(jnp.zeros_like, layer_vals)
    _, vjp = jax.vjp(aux_last_layer, aux_vals, params)
    summed_h_tangents = sum(jax.tree_flatten(h_tangents)[0])
    aux_tangents, p_tangents = vjp(summed_h_tangents)
    layers_info = []
    for aux_p, p_p in zip(layer_vals, params):
      info = dict()
      info["outputs"] = (aux_p[0],)
      info["inputs"] = (aux_p[1],)
      info["params"] = (p_p[0], p_p[1])
      layers_info.append(info)
    for i, (aux_t, p_t) in enumerate(zip(aux_tangents, p_tangents)):
      info = dict()
      info["outputs_tangent"] = (aux_t[0],)
      info["inputs_tangent"] = (aux_t[1],)
      info["params_tangent"] = (p_t[0], p_t[1])
      layers_info[i].update(info)
    layers_info = tuple(layers_info)

    func = tgm.auto_register_tags(func, (params, data))
    tracer_vjp = tracer.trace_estimator_vjp(func)
    tracer_losses, tracer_vjp_func = tracer_vjp((params, data))
    tracer_losses = [loss.evaluate(None) for loss in tracer_losses]
    tracer_outputs = tracer_vjp_func((h_tangents[:1], h_tangents[1:]))

    self.assertStructureAllClose(loss_vals, tracer_losses)
    self.assertStructureAllClose(tracer_outputs, layers_info, atol=3e-6)


if __name__ == "__main__":
  absltest.main()
