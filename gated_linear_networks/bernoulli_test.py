# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited.
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
"""Tests for `bernoulli.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tree

from gated_linear_networks import bernoulli


def _get_dataset(input_size, batch_size=None):
  """Get mock dataset."""
  if batch_size:
    inputs = jnp.ones([batch_size, input_size])
    side_info = jnp.ones([batch_size, input_size])
    targets = jnp.ones([batch_size])
  else:
    inputs = jnp.ones([input_size])
    side_info = jnp.ones([input_size])
    targets = jnp.ones([])

  return inputs, side_info, targets


class GatedLinearNetworkTest(parameterized.TestCase):

  # TODO(b/170843789): Factor out common test utilities.
  def setUp(self):
    super(GatedLinearNetworkTest, self).setUp()
    self._name = "test_network"
    self._rng = hk.PRNGSequence(jax.random.PRNGKey(42))

    self._output_sizes = (4, 5, 6)
    self._context_dim = 2

    def gln_factory():
      return bernoulli.GatedLinearNetwork(
          output_sizes=self._output_sizes,
          context_dim=self._context_dim,
          name=self._name)

    def inference_fn(inputs, side_info):
      return gln_factory().inference(inputs, side_info)

    def batch_inference_fn(inputs, side_info):
      return jax.vmap(inference_fn, in_axes=(0, 0))(inputs, side_info)

    def update_fn(inputs, side_info, label, learning_rate):
      params, predictions, unused_loss = gln_factory().update(
          inputs, side_info, label, learning_rate)
      return predictions, params

    def batch_update_fn(inputs, side_info, label, learning_rate):
      predictions, params = jax.vmap(
          update_fn, in_axes=(0, 0, 0, None))(inputs, side_info, label,
                                              learning_rate)
      avg_params = tree.map_structure(lambda x: jnp.mean(x, axis=0), params)
      return predictions, avg_params

    # Haiku transform functions.
    self._init_fn, inference_fn_ = hk.without_apply_rng(
        hk.transform_with_state(inference_fn))
    self._batch_init_fn, batch_inference_fn_ = hk.without_apply_rng(
        hk.transform_with_state(batch_inference_fn))
    _, update_fn_ = hk.without_apply_rng(hk.transform_with_state(update_fn))
    _, batch_update_fn_ = hk.without_apply_rng(
        hk.transform_with_state(batch_update_fn))

    self._inference_fn = jax.jit(inference_fn_)
    self._batch_inference_fn = jax.jit(batch_inference_fn_)
    self._update_fn = jax.jit(update_fn_)
    self._batch_update_fn = jax.jit(batch_update_fn_)

  @parameterized.named_parameters(("Online mode", None), ("Batch mode", 3))
  def test_shapes(self, batch_size):
    """Test shapes in online and batch regimes."""
    if batch_size is None:
      init_fn = self._init_fn
      inference_fn = self._inference_fn
    else:
      init_fn = self._batch_init_fn
      inference_fn = self._batch_inference_fn

    input_size = 10
    inputs, side_info, _ = _get_dataset(input_size, batch_size)
    input_size = inputs.shape[-1]

    # Initialize network.
    gln_params, gln_state = init_fn(next(self._rng), inputs, side_info)

    # Test shapes of parameters layer-wise.
    layer_input_size = input_size
    for layer_idx, output_size in enumerate(self._output_sizes):
      name = "{}/~/{}_layer_{}".format(self._name, self._name, layer_idx)
      weights = gln_params[name]["weights"]
      expected_shape = (output_size, 2**self._context_dim, layer_input_size + 1)
      self.assertEqual(weights.shape, expected_shape)

      layer_input_size = output_size

    # Test shape of output.
    output_size = sum(self._output_sizes)
    predictions, _ = inference_fn(gln_params, gln_state, inputs, side_info)
    expected_shape = (batch_size, output_size) if batch_size else (output_size,)
    self.assertEqual(predictions.shape, expected_shape)

  @parameterized.named_parameters(("Online mode", None), ("Batch mode", 3))
  def test_update(self, batch_size):
    """Test network updates in online and batch regimes."""
    if batch_size is None:
      init_fn = self._init_fn
      inference_fn = self._inference_fn
      update_fn = self._update_fn
    else:
      init_fn = self._batch_init_fn
      inference_fn = self._batch_inference_fn
      update_fn = self._batch_update_fn

    input_size = 10
    inputs, side_info, targets = _get_dataset(input_size, batch_size)

    # Initialize network.
    initial_params, gln_state = init_fn(next(self._rng), inputs, side_info)

    # Initial predictions.
    initial_predictions, _ = inference_fn(initial_params, gln_state, inputs,
                                          side_info)

    # Test that params remain valid after consecutive updates.
    gln_params = initial_params

    for _ in range(3):
      (_, gln_params), gln_state = update_fn(
          gln_params, gln_state, inputs, side_info, targets, learning_rate=1e-4)

      # Check updated weights layer-wise.
      for layer_idx in range(len(self._output_sizes)):
        name = "{}/~/{}_layer_{}".format(self._name, self._name, layer_idx)

        initial_weights = initial_params[name]["weights"]
        new_weights = gln_params[name]["weights"]

        # Shape consistency.
        self.assertEqual(new_weights.shape, initial_weights.shape)

      # Check that different weights yield different predictions.
      new_predictions, _ = inference_fn(gln_params, gln_state, inputs,
                                        side_info)
      self.assertFalse(np.array_equal(new_predictions, initial_predictions))

  def test_batch_consistency(self):
    """Test consistency between online and batch updates."""

    input_size = 10
    batch_size = 3
    inputs, side_info, targets = _get_dataset(input_size, batch_size)

    # Initialize network.
    gln_params, gln_state = self._batch_init_fn(
        next(self._rng), inputs, side_info)
    test_layer = "{}/~/{}_layer_0".format(self._name, self._name)

    for _ in range(10):

      # Update on full batch.
      (expected_predictions, expected_params), _ = self._batch_update_fn(
          gln_params, gln_state, inputs, side_info, targets, learning_rate=1e-3)

      # Average updates across batch and check equivalence.
      accum_predictions = []
      accum_weights = []
      for inputs_, side_info_, targets_ in zip(inputs, side_info, targets):
        (predictions, params), _ = self._update_fn(
            gln_params,
            gln_state,
            inputs_,
            side_info_,
            targets_,
            learning_rate=1e-3)
        accum_predictions.append(predictions)
        accum_weights.append(params[test_layer]["weights"])

      # Check prediction equivalence.
      actual_predictions = np.stack(accum_predictions, axis=0)
      np.testing.assert_array_almost_equal(actual_predictions,
                                           expected_predictions)

      # Check weight equivalence.
      actual_weights = np.mean(np.stack(accum_weights, axis=0), axis=0)
      expected_weights = expected_params[test_layer]["weights"]
      np.testing.assert_array_almost_equal(actual_weights, expected_weights)

      gln_params = expected_params


if __name__ == "__main__":
  absltest.main()
