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
"""Data Parallel Updater for Graph2text data."""

import functools
import os
import pickle

from absl import logging

import haiku as hk
import jax
from jax.tree_util import tree_multimap
import numpy as np
import optax


def call_fn_with_state_keys(jit_fn, state, other_inputs, keys):
  """Executes `jit_fn`, filtering out all keys except some subset."""
  state = state.copy()
  extra_state = {}
  for k in list(state.keys()):
    if k not in keys:
      extra_state[k] = state.pop(k)
  return jit_fn(state, *other_inputs), extra_state


class Updater:
  """Graph2text model updater with multi-GPU support."""

  def __init__(self, loss_fn, optimizer, devices=None, has_graph=False):
    self._net_init_fn, self._apply_fn = hk.transform_with_state(
        functools.partial(loss_fn, is_training=True))
    _, self._eval_apply_fn = hk.transform_with_state(
        functools.partial(loss_fn, is_training=False))

    if optimizer is None:
      optimizer = optax.identity()
    self._optimizer = optimizer

    self._num_devices = jax.local_device_count()
    if devices is None:
      devices = []
      for host_id in range(jax.process_count()):
        for device_id in jax.local_devices(host_id):
          devices.append(device_id)
    else:
      self._num_devices = min(self._num_devices, len(devices))

    def _pmap(f, static_broadcasted_argnums=()):
      return jax.pmap(f, axis_name='i', devices=devices,
                      static_broadcasted_argnums=static_broadcasted_argnums)

    def handle_graph_size(fn):
      def _fn(*args):
        batch = args[-1].copy()
        max_graph_size = batch['max_graph_size']
        del batch['max_graph_size']
        args = args[:-1] + (batch, max_graph_size)
        return fn(*args)
      return _fn

    # Try to jit.
    if has_graph:
      # If the model contains full graphs, we need to set the max_graph_size
      # as a statically broadcasted argument.
      self._init_fn = handle_graph_size(_pmap(self._init, 4))
      self._update_fn = handle_graph_size(_pmap(self._update, 2))
      self._eval_fn = handle_graph_size(_pmap(self._eval, 2))
    else:
      self._init_fn = _pmap(self._init)
      self._update_fn = _pmap(self._update)
      self._eval_fn = _pmap(self._eval)

  def _init(self, master_rng, params, network_state, data, max_graph_size=None):
    """Initializes state of the updater."""
    out_rng, init_rng = jax.random.split(master_rng)
    if max_graph_size is not None:
      new_params, new_network_state = self._net_init_fn(
          init_rng, data, max_graph_size)
    else:
      new_params, new_network_state = self._net_init_fn(init_rng, data)
    if params is None:
      params = new_params
    if network_state is None:
      network_state = new_network_state
    opt_state = self._optimizer.init(params)
    return dict(
        replicated_step=0,
        rng=out_rng,
        state=network_state,
        opt_state=opt_state,
        params=params,
    )

  def init(self, master_rng, data, params=None, network_state=None,
           replicated_params=False):
    """Initializes state of the updater."""
    data = self._preprocess(data)
    rngs = np.array([master_rng] * self._num_devices)
    if not replicated_params and params is not None:
      params = jax.tree_map(
          lambda x: np.array([x] * self._num_devices), params)
    state = self._init_fn(rngs, params, network_state, data)
    state['step'] = np.array(0, dtype=np.int64)
    # Wait for initialization to finish before starting training to keep
    # memory usage low.
    flat_params = jax.tree_leaves(state['params'])
    if flat_params:
      jax.tree_leaves(state['params'])[0].block_until_ready()
    return state

  def _update(self, state, data, max_graph_size=None):
    """Updates parameters."""
    replicated_step = state['replicated_step']
    rng = state['rng']
    opt_state = state['opt_state']
    params = state['params']
    net_state = state['state']

    rng, new_rng = jax.random.split(rng)
    rng = jax.random.fold_in(rng, jax.lax.axis_index('i'))

    def _loss(params, state, batch, rng):
      if max_graph_size is not None:
        (loss, metrics), state = self._apply_fn(params, state, rng, batch,
                                                max_graph_size)
      else:
        (loss, metrics), state = self._apply_fn(params, state, rng, batch)
      return loss, (metrics, state)

    (loss, (metrics, new_net_state)), g = jax.value_and_grad(
        _loss, has_aux=True)(params, net_state, data, rng)
    g = jax.lax.pmean(g, axis_name='i')
    loss = jax.lax.pmean(loss, axis_name='i')
    metrics = jax.lax.pmean(metrics, axis_name='i')

    updates, new_opt_state = self._optimizer.update(g, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    new_state = dict(
        replicated_step=replicated_step + 1,
        rng=new_rng,
        state=new_net_state,
        opt_state=new_opt_state,
        params=new_params,
    )

    metrics['loss'] = loss
    metrics['step'] = replicated_step
    return new_state, metrics

  def update(self, state, data):
    """Updates the state using some data and returns metrics."""
    data = self._preprocess(data)
    (state, out), extra_state = call_fn_with_state_keys(
        self._update_fn, state, [data], keys=set([
            'state', 'params', 'rng', 'replicated_step', 'opt_state']))
    state.update(extra_state)
    state['step'] += 1
    return state, tree_multimap(lambda x: x[0], out)

  def _eval(self, state, data, max_graph_size=None):
    """Evaluates the current state on the given data."""
    if max_graph_size is not None:
      (loss, metrics), new_state = self._eval_apply_fn(
          state['params'], state['state'], state['rng'], data, max_graph_size)
    else:
      (loss, metrics), new_state = self._eval_apply_fn(
          state['params'], state['state'], state['rng'], data)
    state['state'] = new_state
    loss = jax.lax.pmean(loss, axis_name='i')
    metrics = jax.lax.pmean(metrics, axis_name='i')
    metrics['loss'] = loss
    metrics['step'] = state['replicated_step']
    return state, metrics

  def eval_return_state(self, state, data):
    """Returns metrics without updating the model."""
    data = self._preprocess(data)
    (state, out), extra_state = call_fn_with_state_keys(
        self._eval_fn, state, [data], keys=set([
            'state', 'params', 'rng', 'replicated_step']))
    state.update(extra_state)
    return state, tree_multimap(lambda x: x[0], out)

  def eval(self, state, data):
    """Returns metrics without updating the model."""
    _, out = self.eval_return_state(state, data)
    return out

  def _preprocess(self, data):
    """Reshapes input so that it can be distributed across multiple cores."""
    multi_inputs = data.copy()

    def add_core_dimension(x):
      if np.isscalar(x):
        return x
      if x.shape[0] % self._num_devices != 0:
        raise ValueError(f'The batch size must be a multiple of the number of'
                         f' devices. Got batch size = {x.shape[0]} and number'
                         f' of devices = {self._num_devices}.')
      prefix = (self._num_devices, x.shape[0] // self._num_devices)
      return np.reshape(x, prefix + x.shape[1:])

    multi_inputs = tree_multimap(add_core_dimension, multi_inputs)
    return multi_inputs

  def params(self, state):
    """Returns model parameters."""
    return tree_multimap(lambda x: x[0], state['params'])

  def opt_state(self, state):
    """Returns the state of the optimiser."""
    return tree_multimap(lambda x: x[0], state['opt_state'])

  def network_state(self, state):
    """Returns the model's state."""
    return tree_multimap(lambda x: x[0], state['state'])

  def to_checkpoint_state(self, state):
    """Transforms the updater state into a checkpointable state."""
    checkpoint_state = state.copy()
    # Wrapper around checkpoint_state['step'] so we can get [0].
    checkpoint_state['step'] = checkpoint_state['step'][np.newaxis]
    # Unstack the replicated contents.
    checkpoint_state = tree_multimap(lambda x: x[0], checkpoint_state)
    return checkpoint_state

  def from_checkpoint_state(self, checkpoint_state):
    """Initializes the updater state from the checkpointed state."""
    # Expand the checkpoint so we have a copy for each device.
    state = tree_multimap(lambda x: np.stack(jax.local_device_count() * [x]),
                          checkpoint_state)
    state['step'] = state['step'][0]  # Undo stacking for step.
    return state


class CheckpointingUpdater:
  """A checkpointing wrapper around an Updater."""

  def __init__(self,
               inner: Updater,
               checkpoint_dir: str):
    self._inner = inner
    self._checkpoint_dir = checkpoint_dir

  def _checkpoint_paths(self):
    return [p for p in os.listdir(self._checkpoint_dir) if 'checkpoint' in p]

  def init(self, rng, data, params=None, network_state=None):
    """Initialize experiment state."""
    if not os.path.exists(self._checkpoint_dir) or not self._checkpoint_paths():
      os.makedirs(self._checkpoint_dir, exist_ok=True)
      return self._inner.init(rng, data, params, network_state)
    return self.load_checkpoint()

  def init_from_checkpoint(self, rng, data, checkpoint_state):
    params = self._inner.params(checkpoint_state)
    network_state = None
    return self._inner.init(rng, data, params, network_state)

  def eval_return_state(self, state, data):
    return self._inner.eval_return_state(state, data)

  def save_checkpoint(self, state):
    path = os.path.join(self._checkpoint_dir, 'checkpoint.pkl')
    logging.info('Serializing experiment state to %s', path)
    checkpoint_state = self._inner.to_checkpoint_state(jax.device_get(state))
    with open(path, 'wb') as f:
      pickle.dump(checkpoint_state, f)

  def load_checkpoint(self):
    checkpoint = os.path.join(self._checkpoint_dir,
                              self._checkpoint_paths()[-1])
    logging.info('Loading checkpoint from %s', checkpoint)
    with open(checkpoint, 'rb') as f:
      state = pickle.load(f)
    return self._inner.from_checkpoint_state(state)

  def update(self, state, data):
    """Update experiment state."""
    state, out = self._inner.update(state, data)
    return state, out
