# Copyright 2021 DeepMind Technologies Limited.
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

"""PCQM4M-LSC Jaxline experiment."""

import datetime
import functools
import os
import signal
import threading
from typing import Iterable, Mapping, NamedTuple, Tuple

from absl import app
from absl import flags
from absl import logging
import chex
import dill
import haiku as hk
import jax
from jax.config import config as jax_config
import jax.numpy as jnp
from jaxline import experiment
from jaxline import platform
from jaxline import utils
import jraph
import numpy as np
import optax
import tensorflow as tf
import tree

# pylint: disable=g-bad-import-order
import dataset_utils
import datasets
import model


FLAGS = flags.FLAGS


def _get_step_date_label(global_step: int):
  # Date removing microseconds.
  date_str = datetime.datetime.now().isoformat().split('.')[0]
  return f'step_{global_step}_{date_str}'


class _Predictions(NamedTuple):
  predictions: np.ndarray
  indices: np.ndarray


def tf1_ema(ema_value, current_value, decay, step):
  """Implements EMA with TF1-style decay warmup."""
  decay = jnp.minimum(decay, (1.0 + step) / (10.0 + step))
  return ema_value * decay + current_value * (1 - decay)


def _sort_predictions_by_indices(predictions: _Predictions):
  sorted_order = np.argsort(predictions.indices)
  return _Predictions(
      predictions=predictions.predictions[sorted_order],
      indices=predictions.indices[sorted_order])


class Experiment(experiment.AbstractExperiment):
  """OGB Graph Property Prediction GraphNet experiment."""

  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_opt_state': 'opt_state',
      '_network_state': 'network_state',
      '_ema_network_state': 'ema_network_state',
      '_ema_params': 'ema_params',
  }

  def __init__(self, mode, init_rng, config):
    """Initializes experiment."""
    super(Experiment, self).__init__(mode=mode, init_rng=init_rng)
    if mode not in ('train', 'eval', 'train_eval_multithreaded'):
      raise ValueError(f'Invalid mode {mode}.')

    # Do not use accelerators in data pipeline.
    tf.config.experimental.set_visible_devices([], device_type='GPU')
    tf.config.experimental.set_visible_devices([], device_type='TPU')

    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    self.loss = None
    self.forward = None

    # Needed for checkpoint restore.
    self._params = None
    self._network_state = None
    self._opt_state = None
    self._ema_network_state = None
    self._ema_params = None

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| "__/ _` | | "_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step: jnp.ndarray, rng: jnp.ndarray, **unused_args):
    """See Jaxline base class."""
    if self.loss is None:
      self._train_init()

    graph = next(self._train_input)
    out = self.update_parameters(
        self._params,
        self._ema_params,
        self._network_state,
        self._ema_network_state,
        self._opt_state,
        global_step,
        rng,
        graph._asdict())
    (self._params, self._ema_params, self._network_state,
     self._ema_network_state, self._opt_state, scalars) = out
    return utils.get_first(scalars)

  def _construct_loss_config(self):
    loss_config = getattr(model, self.config.model.loss_config_name)
    if self.config.model.loss_config_name == 'RegressionLossConfig':
      return loss_config(
          mean=datasets.NORMALIZE_TARGET_MEAN,
          std=datasets.NORMALIZE_TARGET_STD,
          kwargs=self.config.model.loss_kwargs)
    else:
      raise ValueError('Unknown Loss Config')

  def _train_init(self):
    self.loss = hk.transform_with_state(self._loss)
    self._train_input = utils.py_prefetch(
        lambda: self._build_numpy_dataset_iterator('train', is_training=True))
    init_stacked_graphs = next(self._train_input)
    init_key = utils.bcast_local_devices(self.init_rng)
    p_init = jax.pmap(self.loss.init)
    self._params, self._network_state = p_init(init_key,
                                               **init_stacked_graphs._asdict())

    # Learning rate scheduling.
    lr_schedule = optax.warmup_cosine_decay_schedule(
        **self.config.optimizer.lr_schedule)

    self.optimizer = getattr(optax, self.config.optimizer.name)(
        learning_rate=lr_schedule, **self.config.optimizer.optimizer_kwargs)

    self._opt_state = jax.pmap(self.optimizer.init)(self._params)
    self.update_parameters = jax.pmap(self._update_parameters, axis_name='i')
    if self.config.ema:
      self._ema_params = self._params
      self._ema_network_state = self._network_state

  def _loss(
      self, **graph: Mapping[str, chex.ArrayTree]) -> chex.ArrayTree:

    graph = jraph.GraphsTuple(**graph)
    model_instance = model.GraphPropertyEncodeProcessDecode(
        loss_config=self._construct_loss_config(), **self.config.model)
    loss, scalars = model_instance.get_loss(graph)
    return loss, scalars

  def _maybe_save_predictions(
      self,
      predictions: jnp.ndarray,
      split: str,
      global_step: jnp.ndarray,
  ):
    if not self.config.predictions_dir:
      return
    output_dir = os.path.join(self.config.predictions_dir,
                              _get_step_date_label(global_step))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, split + '.dill')

    with open(output_path, 'wb') as f:
      dill.dump(predictions, f)
    logging.info('Saved %s predictions at: %s', split, output_path)

  def _build_numpy_dataset_iterator(self, split: str, is_training: bool):
    dynamic_batch_size_config = (
        self.config.training.dynamic_batch_size
        if is_training else self.config.evaluation.dynamic_batch_size)

    return dataset_utils.build_dataset_iterator(
        split=split,
        dynamic_batch_size_config=dynamic_batch_size_config,
        sample_random=self.config.sample_random,
        debug=self.config.debug,
        is_training=is_training,
        **self.config.dataset_config)

  def _update_parameters(
      self,
      params: hk.Params,
      ema_params: hk.Params,
      network_state: hk.State,
      ema_network_state: hk.State,
      opt_state: optax.OptState,
      global_step: jnp.ndarray,
      rng: jnp.ndarray,
      graph: jraph.GraphsTuple,
  ) -> Tuple[hk.Params, hk.Params, hk.State, hk.State, optax.OptState,
             chex.ArrayTree]:
    """Updates parameters."""
    def get_loss(*x, **graph):
      (loss, scalars), network_state = self.loss.apply(*x, **graph)
      return loss, (scalars, network_state)
    grad_loss_fn = jax.grad(get_loss, has_aux=True)
    scaled_grads, (scalars, network_state) = grad_loss_fn(
        params, network_state, rng, **graph)
    grads = jax.lax.psum(scaled_grads, axis_name='i')
    updates, opt_state = self.optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    if ema_params is not None:
      ema = lambda x, y: tf1_ema(x, y, self.config.ema_decay, global_step)
      ema_params = jax.tree_multimap(ema, ema_params, params)
      ema_network_state = jax.tree_multimap(ema, ema_network_state,
                                            network_state)
    return params, ema_params, network_state, ema_network_state, opt_state, scalars

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step: jnp.ndarray, rng: jnp.ndarray,
               **unused_kwargs) -> chex.ArrayTree:
    """See Jaxline base class."""
    if self.forward is None:
      self._eval_init()

    if self.config.ema:
      params = utils.get_first(self._ema_params)
      state = utils.get_first(self._ema_network_state)
    else:
      params = utils.get_first(self._params)
      state = utils.get_first(self._network_state)
    rng = utils.get_first(rng)

    split = self.config.evaluation.split
    predictions, scalars = self._get_predictions(
        params, state, rng,
        utils.py_prefetch(
            functools.partial(
                self._build_numpy_dataset_iterator, split, is_training=False)))
    self._maybe_save_predictions(predictions, split, global_step[0])
    return scalars

  def _sum_regression_scalars(self, preds: jnp.ndarray,
                              graph: jraph.GraphsTuple) -> chex.ArrayTree:
    """Creates unnormalised values for accumulation."""
    targets = graph.globals['target']
    graph_mask = jraph.get_graph_padding_mask(graph)
    # Sum for accumulation, normalise later since there are a
    # variable number of graphs per batch.
    mae = model.sum_with_mask(jnp.abs(targets - preds), graph_mask)
    mse = model.sum_with_mask((targets - preds)**2, graph_mask)
    count = jnp.sum(graph_mask)
    return {'values': {'mae': mae.item(), 'mse': mse.item()},
            'counts': {'mae': count.item(), 'mse': count.item()}}

  def _get_prediction(
      self,
      params: hk.Params,
      state: hk.State,
      rng: jnp.ndarray,
      graph: jraph.GraphsTuple,
  ) -> np.ndarray:
    """Returns predictions for all the graphs in the dataset split."""
    model_out, _ = self.eval_apply(params, state, rng, **graph._asdict())
    prediction = np.squeeze(model_out['globals'], axis=1)
    return prediction

  def _get_predictions(
      self,
      params: hk.Params,
      state: hk.State,
      rng: jnp.ndarray,
      graph_iterator: Iterable[jraph.GraphsTuple],
  ) -> Tuple[_Predictions, chex.ArrayTree]:
    all_scalars = []
    predictions = []
    graph_indices = []
    for i, graph in enumerate(graph_iterator):
      prediction = self._get_prediction(params, state, rng, graph)
      if 'target' in graph.globals and not jnp.isnan(
          graph.globals['target']).any():
        scalars = self._sum_regression_scalars(prediction, graph)
        all_scalars.append(scalars)
      num_padding_graphs = jraph.get_number_of_padding_with_graphs_graphs(graph)
      num_valid_graphs = len(graph.n_node) - num_padding_graphs
      depadded_prediction = prediction[:num_valid_graphs]
      predictions.append(depadded_prediction)
      graph_indices.append(graph.globals['graph_index'][:num_valid_graphs])

      if i % 1000 == 0:
        logging.info('Generated predictions for %d batches so far', i + 1)

    predictions = _sort_predictions_by_indices(
        _Predictions(
            predictions=np.concatenate(predictions),
            indices=np.concatenate(graph_indices)))

    if all_scalars:
      sum_all_args = lambda *l: sum(l)
      # Sum over graphs in the dataset.
      accum_scalars = tree.map_structure(sum_all_args, *all_scalars)
      scalars = tree.map_structure(lambda x, y: x / y, accum_scalars['values'],
                                   accum_scalars['counts'])
    else:
      scalars = {}
    return predictions, scalars

  def _eval_init(self):
    self.forward = hk.transform_with_state(self._forward)
    self.eval_apply = jax.jit(self.forward.apply)

  def _forward(self, **graph: Mapping[str, chex.ArrayTree]) -> chex.ArrayTree:

    graph = jraph.GraphsTuple(**graph)
    model_instance = model.GraphPropertyEncodeProcessDecode(
        loss_config=self._construct_loss_config(), **self.config.model)
    return model_instance(graph)


def _restore_state_to_in_memory_checkpointer(restore_path):
  """Initializes experiment state from a checkpoint."""

  # Load pretrained experiment state.
  python_state_path = os.path.join(restore_path, 'checkpoint.dill')
  with open(python_state_path, 'rb') as f:
    pretrained_state = dill.load(f)
  logging.info('Restored checkpoint from %s', python_state_path)

  # Assign state to a dummy experiment instance for the in-memory checkpointer,
  # broadcasting to devices.
  dummy_experiment = Experiment(
      mode='train', init_rng=0, config=FLAGS.config.experiment_kwargs.config)
  for attribute, key in Experiment.CHECKPOINT_ATTRS.items():
    setattr(dummy_experiment, attribute,
            utils.bcast_local_devices(pretrained_state[key]))

  jaxline_state = dict(
      global_step=pretrained_state['global_step'],
      experiment_module=dummy_experiment)
  snapshot = utils.SnapshotNT(0, jaxline_state)

  # Finally, seed the jaxline `utils.InMemoryCheckpointer` global dict.
  utils.GLOBAL_CHECKPOINT_DICT['latest'] = utils.CheckpointNT(
      threading.local(), [snapshot])


def _save_state_from_in_memory_checkpointer(
    save_path, experiment_class: experiment.AbstractExperiment):
  """Saves experiment state to a checkpoint."""
  logging.info('Saving model.')
  for checkpoint_name, checkpoint in utils.GLOBAL_CHECKPOINT_DICT.items():
    if not checkpoint.history:
      logging.info('Nothing to save in "%s"', checkpoint_name)
      continue

    pickle_nest = checkpoint.history[-1].pickle_nest
    global_step = pickle_nest['global_step']

    state_dict = {'global_step': global_step}
    for attribute, key in experiment_class.CHECKPOINT_ATTRS.items():
      state_dict[key] = utils.get_first(
          getattr(pickle_nest['experiment_module'], attribute))
    save_dir = os.path.join(
        save_path, checkpoint_name, _get_step_date_label(global_step))
    python_state_path = os.path.join(save_dir, 'checkpoint.dill')
    os.makedirs(save_dir, exist_ok=True)
    with open(python_state_path, 'wb') as f:
      dill.dump(state_dict, f)
    logging.info(
        'Saved "%s" checkpoint to %s', checkpoint_name, python_state_path)


def _setup_signals(save_model_fn):
  """Sets up a signal for model saving."""
  # Save a model on Ctrl+C.
  def sigint_handler(unused_sig, unused_frame):
    # Ideally, rather than saving immediately, we would then "wait" for a good
    # time to save. In practice this reads from an in-memory checkpoint that
    # only saves every 30 seconds or so, so chances of race conditions are very
    # small.
    save_model_fn()
    logging.info(r'Use `Ctrl+\` to save and exit.')

  # Exit on `Ctrl+\`, saving a model.
  prev_sigquit_handler = signal.getsignal(signal.SIGQUIT)
  def sigquit_handler(unused_sig, unused_frame):
    # Restore previous handler early, just in case something goes wrong in the
    # next lines, so it is possible to press again and exit.
    signal.signal(signal.SIGQUIT, prev_sigquit_handler)
    save_model_fn()
    logging.info(r'Exiting on `Ctrl+\`')

    # Re-raise for clean exit.
    os.kill(os.getpid(), signal.SIGQUIT)

  signal.signal(signal.SIGINT, sigint_handler)
  signal.signal(signal.SIGQUIT, sigquit_handler)


def main(argv, experiment_class: experiment.AbstractExperiment):

  # Maybe restore a model.
  restore_path = FLAGS.config.restore_path
  if restore_path:
    _restore_state_to_in_memory_checkpointer(restore_path)

  # Maybe save a model.
  save_dir = os.path.join(FLAGS.config.checkpoint_dir, 'models')
  if FLAGS.config.one_off_evaluate:
    save_model_fn = lambda: None  # No need to save checkpoint in this case.
  else:
    save_model_fn = functools.partial(
        _save_state_from_in_memory_checkpointer, save_dir, experiment_class)
  _setup_signals(save_model_fn)  # Save on Ctrl+C (continue) or Ctrl+\ (exit).

  try:
    platform.main(experiment_class, argv)
  finally:
    save_model_fn()  # Save at the end of training or in case of exception.


if __name__ == '__main__':
  jax_config.update('jax_debug_nans', False)
  flags.mark_flag_as_required('config')
  app.run(lambda argv: main(argv, Experiment))
