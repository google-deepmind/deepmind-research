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
"""Base classes for Gated Linear Networks."""

import abc
import collections
import functools
import inspect
from typing import Any, Callable, Optional, Sequence, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp


Array = chex.Array
DType = Any
Initializer = hk.initializers.Initializer
Shape = Sequence[int]

EPS = 1e-12
MIN_ALPHA = 1e-5


def _l2_normalize(x: Array, axis: int) -> Array:
  return x / jnp.sqrt(jnp.maximum(jnp.sum(x**2, axis, keepdims=True), EPS))


def _wrapped_fn_argnames(fun):
  """Returns list of argnames of a (possibly wrapped) function."""
  return tuple(inspect.signature(fun).parameters)


def _vmap(fun, in_axes=0, out_axes=0, parameters=None):
  """JAX vmap with human-friendly axes."""

  def _axes(fun, d):
    """Maps dict {kwarg_i, : val_i} to [None, ..., val_i, ..., None]."""
    argnames = _wrapped_fn_argnames(fun) if not parameters else parameters
    for key in d:
      if key not in argnames:
        raise ValueError(f"{key} is not a valid axis.")
    return tuple(d.get(key, None) for key in argnames)

  in_axes = _axes(fun, in_axes) if isinstance(in_axes, dict) else in_axes
  return jax.vmap(fun, in_axes, out_axes)

# Map a neuron-level function across a layer.
_layer_vmap = functools.partial(
    _vmap,
    in_axes=({
        "weights": 0,
        "hyperplanes": 0,
        "hyperplane_bias": 0,
    }))


class NormalizedRandomNormal(hk.initializers.RandomNormal):
  """Random normal initializer with l2-normalization."""

  def __init__(self,
               stddev: float = 1.,
               mean: float = 0.,
               normalize_axis: int = 0):
    super(NormalizedRandomNormal, self).__init__(stddev, mean)
    self._normalize_axis = normalize_axis

  def __call__(self, shape: Shape, dtype: DType) -> Array:
    if self._normalize_axis >= len(shape):
      raise ValueError("Cannot normalize axis {} for ndim = {}.".format(
          self._normalize_axis, len(shape)))
    weights = super(NormalizedRandomNormal, self).__call__(shape, dtype)
    return _l2_normalize(weights, axis=self._normalize_axis)


class ShapeScaledConstant(hk.initializers.Initializer):
  """Initializes with a constant dependent on last dimension of input shape."""

  def __call__(self, shape: Shape, dtype: DType) -> jnp.ndarray:
    constant = 1. / shape[-1]
    return jnp.broadcast_to(constant, shape).astype(dtype)


class LocalUpdateModule(hk.Module):
  """Abstract base class for GLN variants and utils."""

  def __init__(self, name: Optional[str] = None):
    if hasattr(self, "__call__"):
      raise ValueError("Do not implement `__call__` for a LocalUpdateModule." +
                       " Implement `inference` and `update` instead.")
    super(LocalUpdateModule, self).__init__(name)

  @abc.abstractmethod
  def inference(self, *args, **kwargs):
    """Module inference step."""

  @abc.abstractmethod
  def update(self, *args, **kwargs):
    """Module update step."""

  @property
  @abc.abstractmethod
  def output_sizes(self) -> Shape:
    """Returns network output sizes."""


class GatedLinearNetwork(LocalUpdateModule):
  """Abstract base class for a multi-layer Gated Linear Network."""

  def __init__(self,
               output_sizes: Shape,
               context_dim: int,
               inference_fn: Callable[..., Array],
               update_fn: Callable[..., Array],
               init: Initializer,
               hyp_w_init: Optional[Initializer] = None,
               hyp_b_init: Optional[Initializer] = None,
               dtype: DType = jnp.float32,
               name: str = "gated_linear_network"):
    """Initialize a GatedLinearNetwork as a sequence of GatedLinearLayers."""
    super(GatedLinearNetwork, self).__init__(name=name)

    self._layers = []
    self._output_sizes = output_sizes
    for i, output_size in enumerate(self._output_sizes):
      layer = _GatedLinearLayer(
          output_size=output_size,
          context_dim=context_dim,
          update_fn=update_fn,
          inference_fn=inference_fn,
          init=init,
          hyp_w_init=hyp_w_init,
          hyp_b_init=hyp_b_init,
          dtype=dtype,
          name=name + "_layer_{}".format(i))
      self._layers.append(layer)
      self._name = name

  @abc.abstractmethod
  def _add_bias(self, inputs):
    pass

  def inference(self, inputs: Array, side_info: Array, *args,
                **kwargs) -> Array:
    """GatedLinearNetwork inference."""
    predictions_per_layer = []
    predictions = inputs
    for layer in self._layers:
      predictions = self._add_bias(predictions)
      predictions = layer.inference(predictions, side_info, *args, **kwargs)
      predictions_per_layer.append(predictions)

    return jnp.concatenate(predictions_per_layer, axis=0)

  def update(self, inputs, side_info, target, learning_rate, *args, **kwargs):
    """GatedLinearNetwork update."""
    all_params = []
    all_predictions = []
    all_losses = []
    predictions = inputs
    for layer in self._layers:
      predictions = self._add_bias(predictions)

      # Note: This is correct because returned predictions are pre-update.
      params, predictions, log_loss = layer.update(predictions, side_info,
                                                   target, learning_rate, *args,
                                                   **kwargs)
      all_params.append(params)
      all_predictions.append(predictions)
      all_losses.append(log_loss)

    new_params = dict(collections.ChainMap(*all_params))
    predictions = jnp.concatenate(all_predictions, axis=0)
    log_loss = jnp.concatenate(all_losses, axis=0)

    return new_params, predictions, log_loss

  @property
  def output_sizes(self):
    return self._output_sizes

  @staticmethod
  def _compute_context(
      side_info: Array,        # [side_info_size]
      hyperplanes: Array,      # [context_dim, side_info_size]
      hyperplane_bias: Array,  # [context_dim]
  ) -> Array:
    # Index weights by side information.
    context_dim = hyperplane_bias.shape[0]
    proj = jnp.dot(hyperplanes, side_info)
    bits = (proj > hyperplane_bias).astype(jnp.int32)
    weight_index = jnp.sum(
        bits *
        jnp.array([2**i for i in range(context_dim)])) if context_dim else 0
    return weight_index


class _GatedLinearLayer(LocalUpdateModule):
  """A single layer of a Gated Linear Network."""

  def __init__(self,
               output_size: int,
               context_dim: int,
               inference_fn: Callable[..., Array],
               update_fn: Callable[..., Array],
               init: Initializer,
               hyp_w_init: Optional[Initializer] = None,
               hyp_b_init: Optional[Initializer] = None,
               dtype: DType = jnp.float32,
               name: str = "gated_linear_layer"):
    """Initialize a GatedLinearLayer."""
    super(_GatedLinearLayer, self).__init__(name=name)
    self._output_size = output_size
    self._context_dim = context_dim
    self._inference_fn = inference_fn
    self._update_fn = update_fn
    self._init = init
    self._hyp_w_init = hyp_w_init
    self._hyp_b_init = hyp_b_init
    self._dtype = dtype
    self._name = name

  def _get_weights(self, input_size):
    """Get (or initialize) weight parameters."""
    weights = hk.get_parameter(
        "weights",
        shape=(self._output_size, 2**self._context_dim, input_size),
        dtype=self._dtype,
        init=self._init,
    )

    return weights

  def _get_hyperplanes(self, side_info_size):
    """Get (or initialize) hyperplane weights and bias."""

    hyp_w_init = self._hyp_w_init or NormalizedRandomNormal(
        stddev=1., normalize_axis=1)
    hyperplanes = hk.get_state(
        "hyperplanes",
        shape=(self._output_size, self._context_dim, side_info_size),
        init=hyp_w_init)

    hyp_b_init = self._hyp_b_init or hk.initializers.RandomNormal(stddev=0.05)
    hyperplane_bias = hk.get_state(
        "hyperplane_bias",
        shape=(self._output_size, self._context_dim),
        init=hyp_b_init)

    return hyperplanes, hyperplane_bias

  def inference(self, inputs: Array, side_info: Array, *args,
                **kwargs) -> Array:
    """GatedLinearLayer inference."""
    # Initialize layer weights.
    weights = self._get_weights(inputs.shape[0])

    # Initialize fixed random hyperplanes.
    side_info_size = side_info.shape[0]
    hyperplanes, hyperplane_bias = self._get_hyperplanes(side_info_size)

    # Perform layer-wise inference by mapping along output_size (num_neurons).
    layer_inference = _layer_vmap(self._inference_fn)
    predictions = layer_inference(inputs, side_info, weights, hyperplanes,
                                  hyperplane_bias, *args, **kwargs)

    return predictions

  def update(self, inputs: Array, side_info: Array, target: Array,
             learning_rate: float, *args,
             **kwargs) -> Tuple[Array, Array, Array]:
    """GatedLinearLayer update."""
    # Fetch layer weights.
    weights = self._get_weights(inputs.shape[0])

    # Fetch fixed random hyperplanes.
    side_info_size = side_info.shape[0]
    hyperplanes, hyperplane_bias = self._get_hyperplanes(side_info_size)

    # Perform layer-wise update by mapping along output_size (num_neurons).
    layer_update = _layer_vmap(self._update_fn)
    new_weights, predictions, log_loss = layer_update(inputs, side_info,
                                                      weights, hyperplanes,
                                                      hyperplane_bias, target,
                                                      learning_rate, *args,
                                                      **kwargs)

    assert new_weights.shape == weights.shape
    params = {self.module_name: {"weights": new_weights}}
    return params, predictions, log_loss

  @property
  def output_sizes(self):
    return self._output_size


class Mutator(LocalUpdateModule):
  """Abstract base class for GLN Mutators."""

  def __init__(
      self,
      network_factory: Callable[..., LocalUpdateModule],
      name: str,
  ):
    super(Mutator, self).__init__(name=name)
    self._network = network_factory()
    self._name = name

  @property
  def output_sizes(self):
    return self._network.output_sizes


class LastNeuronAggregator(Mutator):
  """Last neuron aggregator: network output is read from the last neuron."""

  def __init__(
      self,
      network_factory: Callable[..., LocalUpdateModule],
      name: str = "last_neuron",
  ):
    super(LastNeuronAggregator, self).__init__(network_factory, name)
    if self._network.output_sizes[-1] != 1:
      raise ValueError(
          "LastNeuronAggregator requires the last GLN layer to have"
          " output_size = 1.")

  def inference(self, *args, **kwargs) -> Array:
    predictions = self._network.inference(*args, **kwargs)
    return predictions[-1]

  def update(self, *args, **kwargs) -> Tuple[Array, Array, Array]:
    params_t, predictions_tm1, loss_tm1 = self._network.update(*args, **kwargs)
    return params_t, predictions_tm1[-1], loss_tm1[-1]
