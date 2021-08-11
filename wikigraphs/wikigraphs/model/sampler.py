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
"""Samplers for the graph2text transformers."""

import abc
from typing import Any, Optional, Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from wikigraphs.model import graph_net as gn


class BaseSampler:
  """Base class for transformer samplers."""

  def __init__(self,
               model_fn,
               temperature: float = 1.0,
               device: Optional[Any] = None,
               rng: Optional[np.ndarray] = None):
    """Constructor.

    Args:
      model_fn: a transformer language model defined in model.transformer.
      temperature: sampling temperature.
      device: the sampler will run on this device if provided.
      rng: random number generator.
    """
    self._temperature = temperature
    self._device = device or jax.local_devices()[0]
    init_fn, apply_fn = hk.transform_with_state(model_fn)

    if rng is None:
      rng = jax.random.PRNGKey(np.random.randint(2**32))
    rng = jax.random.fold_in(rng, jax.host_id())
    self._rng = rng
    self._init_state = None
    self._jit_model(init_fn, apply_fn)

  def _jit_model(self, init_fn, apply_fn):
    """Jit the `init_fn` and `apply_fn`."""
    pass

  @abc.abstractmethod
  def _sample(self,
              params: Mapping[str, Any],
              state: Mapping[str, Any],
              rng: jnp.ndarray,
              x: jnp.ndarray,
              **kwargs) -> np.ndarray:
    """Generate samples.

    Args:
      params: parameters of the transformer.
      state: state of the transformer.
      rng: random number generator.
      x: a prompt of shape [batch_size, sample_len], in which an entry of -1
        indicates it will be generate at that place. Otherwise it acts as the
        prompt.
      **kwargs: additional inputs.

    Returns:
      output: [batch_size, sample_len] tensor, the generated sequence.
    """

  @abc.abstractmethod
  def sample(self,
             params: Mapping[str, Any],
             x: jnp.ndarray,
             **kwargs) -> jnp.ndarray:
    """Generate samples based on the given parameters and prompts.

    Args:
      params: parameters of the transformer.
      x: a prompt of shape [batch_size, sample_len], in which an entry of -1
        indicates it will be generate at that place. Otherwise it acts as the
        prompt.
      **kwargs: additional inputs.

    Returns:
      output: the generated sequence.
    """


class TransformerXLSampler(BaseSampler):
  """Sampling from the TransformerXL model."""

  def _jit_model(self, init_fn, apply_fn):
    """Jit `init_fn` and `apply_fn`, the latter is used in `self._sample`."""
    self._init_fn = jax.jit(init_fn, device=self._device)
    self._apply_fn = apply_fn
    self._sample_fn = jax.jit(self._sample, device=self._device)

  def _sample(self,
              params: Mapping[str, Any],
              state: Mapping[str, Any],
              rng: jnp.ndarray,
              x: jnp.ndarray) -> np.ndarray:
    """Generate unconditional samples.

    Args:
      params: parameters of the transformer.
      state: state of the transformer.
      rng: random number generator.
      x: a prompt of shape [batch_size, sample_len], in which an entry of -1
        indicates it will be generate at that place. Otherwise it acts as the
        prompt.

    Returns:
      output: [batch_size, sample_len] tensor, the generated sequence.
    """
    batch_size, sample_len = x.shape

    def one_step(params, state, rng, i, x):
      step_sample = jax.lax.dynamic_slice(x, [0, i], [batch_size, 1])
      rng, rng_ = jax.random.split(rng)
      # step_sample shape is [batch_size, 1].
      logits, state = self._apply_fn(params, state, rng_, step_sample)
      rng, rng_ = jax.random.split(rng)
      step_sample = jax.random.categorical(rng_, logits / self._temperature)
      update = jnp.where(x[:, i + 1] < 0, step_sample[:, 0], x[:, i + 1])[:,
                                                                          None]
      x = jax.lax.dynamic_update_slice(x, update, [0, i + 1])
      return state, rng, x

    def loop_body(i, data):
      state, rng, x = data
      return one_step(params, state, rng, i, x)

    _, _, x = jax.lax.fori_loop(0, sample_len - 1, loop_body,
                                (state, rng, x))

    return x

  def sample(self,
             params: Mapping[str, Any],
             x: jnp.ndarray) -> jnp.ndarray:
    """Generate samples based on the given graphs and parameters.

    Args:
      params: parameters of the transformer.
      x: a prompt of shape [batch_size, sample_len], in which an entry of -1
        indicates it will be generate at that place. Otherwise it acts as the
        prompt.

    Returns:
      output: the generated sequence.
    """
    if self._init_state is None:
      self._rng, rng = jax.random.split(self._rng)
      self._init_params, self._init_state = self._init_fn(rng, x[:, :1])
    if params is None:
      params = self._init_params

    self._rng, rng = jax.random.split(self._rng)
    sample = self._sample_fn(params, self._init_state, rng, x)
    return sample


class Bow2TextTransformerSampler(BaseSampler):
  """Sampling from the TransformerXL model."""

  def _jit_model(self, init_fn, apply_fn):
    """Jit `init_fn` and `apply_fn`, the latter is used in `self._sample`."""
    self._init_fn = jax.jit(init_fn, device=self._device)
    self._apply_fn = apply_fn
    self._sample_fn = jax.jit(self._sample, device=self._device)

  def _sample(self,
              params: Mapping[str, Any],
              state: Mapping[str, Any],
              rng: jnp.ndarray,
              bow: jnp.ndarray,
              x: jnp.ndarray) -> np.ndarray:
    """Generate samples conditioned on the bag-of-words of the graph.

    Args:
      params: parameters of the transformer.
      state: state of the transformer.
      rng: random number generator.
      bow: a [batch_size, bow_vocab_size] tensor, each row is a bow vector.
      x: a prompt of shape [batch_size, sample_len], in which an entry of -1
        indicates it will be generate at that place. Otherwise it acts as the
        prompt.

    Returns:
      output: [batch_size, sample_len] tensor, the generated sequence.
    """
    batch_size, sample_len = x.shape

    def one_step(params, state, rng, i, x):
      step_sample = jax.lax.dynamic_slice(x, [0, i], [batch_size, 1])
      rng, rng_ = jax.random.split(rng)
      # step_sample shape is [batch_size, 1].
      logits, state = self._apply_fn(params, state, rng_, bow, step_sample)
      rng, rng_ = jax.random.split(rng)
      step_sample = jax.random.categorical(rng_, logits / self._temperature)
      update = jnp.where(x[:, i + 1] < 0, step_sample[:, 0], x[:, i + 1])[:,
                                                                          None]
      x = jax.lax.dynamic_update_slice(x, update, [0, i + 1])
      return state, rng, x

    def loop_body(i, data):
      state, rng, x = data
      return one_step(params, state, rng, i, x)

    _, _, x = jax.lax.fori_loop(0, sample_len - 1, loop_body,
                                (state, rng, x))

    return x

  def sample(self,
             params: Mapping[str, Any],
             x: jnp.ndarray,
             bow: jnp.ndarray) -> jnp.ndarray:
    """Generate samples based on the given graphs and parameters.

    Args:
      params: parameters of the transformer.
      x: a prompt of shape [batch_size, sample_len], in which an entry of -1
        indicates it will be generate at that place. Otherwise it acts as the
        prompt.
      bow: a [batch_size, bow_vocab_size] tensor, each row is a bow vector.

    Returns:
      output: the generated sequence.
    """
    if self._init_state is None:
      self._rng, rng = jax.random.split(self._rng)
      self._init_params, self._init_state = self._init_fn(rng, bow, x[:, :1])
    if params is None:
      params = self._init_params

    self._rng, rng = jax.random.split(self._rng)
    sample = self._sample_fn(params, self._init_state, rng, bow, x)
    return sample


class Graph2TextTransformerSampler(BaseSampler):
  """Sampling from the Graph2Text TransformerXL model."""

  def _jit_model(self, init_fn, apply_fn):
    """Jit `init_fn` and `apply_fn`, the latter is used in `self._sample`."""
    # `pad_n_nodes` is set as a static argument.
    self._init_fn = jax.jit(init_fn, device=self._device, static_argnums=2)
    self._apply_fn = apply_fn
    self._sample_fn = jax.jit(self._sample, device=self._device,
                              static_argnums=4)

  def _sample(self,
              params: Mapping[str, Any],
              state: Mapping[str, Any],
              rng: jnp.ndarray,
              graphs: jraph.GraphsTuple,
              pad_n_nodes: int,
              x: jnp.ndarray) -> np.ndarray:
    """Generate samples conditioned on the bag-of-words reprensation of graph.

    Args:
      params: parameters of the transformer.
      state: state of the transformer.
      rng: random number generator.
      graphs: a graph structured using graph_net.Graph.
      pad_n_nodes: size for each node to pad to.
      x: a prompt of shape [batch_size, sample_len], in which an entry of -1
        indicates it will be generate at that place. Otherwise it acts as the
        prompt.

    Returns:
      output: [batch_size, sample_len] tensor, the generated sequence.
    """
    batch_size, sample_len = x.shape

    def one_step(params, state, rng, i, x):
      step_sample = jax.lax.dynamic_slice(x, [0, i], [batch_size, 1])
      rng, rng_ = jax.random.split(rng)
      # step_sample shape is [batch_size, 1].
      logits, state = self._apply_fn(
          params, state, rng_, graphs, pad_n_nodes, step_sample)
      rng, rng_ = jax.random.split(rng)
      step_sample = jax.random.categorical(rng_, logits / self._temperature)
      update = jnp.where(x[:, i + 1] < 0, step_sample[:, 0], x[:, i + 1])[:,
                                                                          None]
      x = jax.lax.dynamic_update_slice(x, update, [0, i + 1])
      return state, rng, x

    def loop_body(i, data):
      state, rng, x = data
      return one_step(params, state, rng, i, x)

    _, _, x = jax.lax.fori_loop(0, sample_len - 1, loop_body,
                                (state, rng, x))

    return x

  def sample(self,
             params: Mapping[str, Any],
             x: jnp.ndarray,
             graphs: jraph.GraphsTuple,
             pad: bool = True) -> jnp.ndarray:
    """Generate samples based on the given graphs and parameters.

    Args:
      params: parameters of the transformer.
      x: a prompt of shape [batch_size, sample_len], in which an entry of -1
        indicates it will be generate at that place. Otherwise it acts as the
        prompt.
      graphs: a graph structured using graph_net.Graph.
      pad: whether to pad the graph nodes and edges or not.

    Returns:
      output: the generated sequence.
    """
    if pad:
      graphs = gn.pad_graphs(graphs)
      max_graph_size = gn.pad_size(graphs.n_node.max())
    else:
      max_graph_size = graphs.n_node.max()

    if self._init_state is None:
      self._rng, rng = jax.random.split(self._rng)
      self._init_params, self._init_state = self._init_fn(
          rng, graphs, max_graph_size, x[:, :1])
    if params is None:
      params = self._init_params

    self._rng, rng = jax.random.split(self._rng)
    sample = self._sample_fn(
        params, self._init_state, rng, graphs, max_graph_size, x)
    return sample
