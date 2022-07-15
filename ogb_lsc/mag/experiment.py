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

# pylint: disable=line-too-long

r"""MAG240M-LSC Jaxline experiment.

Usage:


```
# A path pointing to the data root.
DATA_ROOT=/tmp/mag/data
# A path for checkpoints.
CHECKPOINT_DIR=/tmp/checkpoint/
# A path for output predictions.
OUTPUT_DIR=/tmp/predictions/
# Whether we are training a model of a k_fold of models (None for no k-fold)
K_FOLD_INDEX=0

```


Some reusable arguments:

```
SHARED_ARGUMENTS="--config=ogb_lsc/mag/config.py \
                  --config.experiment_kwargs.config.dataset_kwargs.data_root=${DATA_ROOT} \
                  --config.experiment_kwargs.config.dataset_kwargs.k_fold_split_id=${K_FOLD_INDEX} \
                  --config.checkpoint_dir=${CHECKPOINT_DIR}"
```

Train only:
  ```
  python -m ogb_lsc.mag.experiment \
      ${SHARED_ARGUMENTS} --jaxline_mode="train"
  RESTORE_PATH=${CHECKPOINT_DIR}/models/latest/step_${STEP}_${TIMESTAMP}
  ```

Train with early stopping on a separate eval thread:
  ```
  python -m ogb_lsc.mag.experiment \
      ${SHARED_ARGUMENTS} --jaxline_mode="train_eval_multithreaded"
  RESTORE_PATH=${CHECKPOINT_DIR}/models/best/step_${STEP}_${TIMESTAMP}
  ```

Produce predictions with a pretrained model:
  ```
  SPLIT="valid"  # Or "test"
  EPOCHS_TO_ENSEMBLE=50  # We used this in the submission.
  python -m ogb_lsc.mag.experiment  \
      ${SHARED_ARGUMENTS} --jaxline_mode="eval" \
      --config.one_off_evaluate=True \
      --config.experiment_kwargs.config.num_eval_iterations_to_ensemble=${EPOCHS_TO_ENSEMBLE} \
      --config.restore_path=${RESTORE_PATH} \
      --config.experiment_kwargs.config.predictions_dir=${OUTPUT_DIR} \
      --config.experiment_kwargs.config.eval.split=${SPLIT}
  ```

Note it is also possible to pass a `restore_path` with `--jaxline_mode="train"`
and training will continue where it left off. In the case of
`--jaxline_mode="train_eval_multithreaded"` this will also work, but early
stopping will not take into account any past best performance up to that
restored model.


Other useful options:

To reduce the training batch size in case of OOM, for example for a batch size
of approximately 48 on average.

```
  SHARED_ARGUMENTS="${SHARED_ARGUMENTS} \
      --config.experiment_kwargs.config.training.dynamic_batch_size_config.n_node=16320 \
      --config.experiment_kwargs.config.training.dynamic_batch_size_config.n_edge=34560 \
      --config.experiment_kwargs.config.training.dynamic_batch_size_config.n_graph=48"
```

To reduce lead time by using dummy adjacency matrices, instead of loading the
the full ones into memory.

```
  SHARED_ARGUMENTS="${SHARED_ARGUMENTS} \
      --config.experiment_kwargs.config.dataset_kwargs.use_dummy_adjacencies=True"
```


"""
# pylint: enable=line-too-long


import datetime
import functools
import os
import signal
import threading
from typing import Tuple

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
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow.compat.v2 as tf

# pylint: disable=g-bad-import-order
import datasets
import losses
import models
import schedules


FLAGS = flags.FLAGS


class Experiment(experiment.AbstractExperiment):
  """MAG240M-LSC Jaxline experiment."""

  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_opt_state': 'opt_state',
      '_network_state': 'network_state',
      '_ema_network_state': 'ema_network_state',
      '_ema_params': 'ema_params',
  }

  def __init__(
      self,
      mode: str,
      init_rng: jnp.ndarray,
      config: config_dict.ConfigDict,
  ):
    """Initializes experiment."""
    super(Experiment, self).__init__(mode=mode, init_rng=init_rng)
    tf.config.experimental.set_visible_devices([], device_type='GPU')
    tf.config.experimental.set_visible_devices([], device_type='TPU')

    if mode not in ('train', 'eval', 'train_eval_multithreaded'):
      raise ValueError(f'Invalid mode {mode}.')

    self.mode = mode
    self.config = config
    self.init_rng = init_rng
    self.forward = hk.transform_with_state(self._forward_fn)

    self._predictions = None

    # Needed for checkpoint restore.
    self._params = None
    self._ema_params = None
    self._network_state = None
    self._ema_network_state = None
    self._opt_state = None

    # Track what has started.
    self._training = False
    self._evaluating = False

  def _train_init(self):
    iterator = self._build_numpy_dataset_iterator('train', is_training=True)
    self._train_input = utils.py_prefetch(lambda: iterator)
    dummy_batch = next(self._train_input)

    if self._params is None:
      self._initialize_experiment_state(self.init_rng, dummy_batch)
    self._update_func = jax.pmap(
        self._update_func,
        axis_name='i',
        donate_argnums=3,
    )
    self._training = True

  def _eval_init(self):

    split = self.config.eval.split
    # Will build the iterator at each evaluation.
    self._make_eval_dataset_iterator = functools.partial(
        utils.py_prefetch,
        lambda: self._build_numpy_dataset_iterator(split, is_training=False))
    self.eval_forward = jax.jit(
        functools.partial(self.forward.apply, is_training=False))
    self._evaluating = True

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(
      self,
      global_step: jnp.ndarray,
      rng: jnp.ndarray,
      **unused_args,
  ) -> losses.LogsDict:
    """See Jaxline base class."""
    if not self._training:
      self._train_init()

    with jax.profiler.StepTraceAnnotation('next_train_input'):
      batch = next(self._train_input)

    with jax.profiler.StepTraceAnnotation('update_step'):
      (self._params, self._ema_params, self._network_state,
       self._ema_network_state, self._opt_state, stats) = self._update_func(
           self._params,
           self._ema_params,
           self._network_state,
           self._ema_network_state,
           self._opt_state,
           global_step,
           rng,
           batch,
       )
      del batch  # Buffers donated to _update_func.

    with jax.profiler.StepTraceAnnotation('get_stats'):
      stats = utils.get_first(stats)
    return stats

  def _build_numpy_dataset_iterator(self, split: str, is_training: bool):
    if is_training:
      dynamic_batch_size_config = self.config.training.dynamic_batch_size_config
    else:
      dynamic_batch_size_config = self.config.eval.dynamic_batch_size_config
    return datasets.build_dataset_iterator(
        split=split,
        dynamic_batch_size_config=dynamic_batch_size_config,
        debug=self.config.debug,
        is_training=is_training,
        **self.config.dataset_kwargs)

  def _initialize_experiment_state(
      self,
      init_rng: jnp.ndarray,
      dummy_batch: datasets.Batch,
  ):
    """Initialize parameters and opt state if not restoring from checkpoint."""
    dummy_graph = dummy_batch.graph

    # Cast features to float32 so that parameters are as appropriate.
    dummy_graph = dummy_graph._replace(
        nodes=jax.tree_map(lambda x: x.astype(np.float32), dummy_graph.nodes),
        edges=jax.tree_map(lambda x: x.astype(np.float32), dummy_graph.edges),
    )
    init_key = utils.bcast_local_devices(init_rng)
    p_init = jax.pmap(functools.partial(self.forward.init, is_training=True))
    params, network_state = p_init(init_key, dummy_graph)
    opt_init, _ = self._optimizer(
        utils.bcast_local_devices(jnp.zeros([], jnp.int32)))
    opt_state = jax.pmap(opt_init)(params)

    # For EMA decay to work correctly, params/state must be floats.
    chex.assert_type(jax.tree_leaves(params), jnp.floating)
    chex.assert_type(jax.tree_leaves(network_state), jnp.floating)

    self._params = params
    self._ema_params = params
    self._network_state = network_state
    self._ema_network_state = network_state
    self._opt_state = opt_state

  def _get_learning_rate(self, global_step: jnp.ndarray) -> jnp.ndarray:
    return schedules.learning_schedule(
        global_step,
        **self.config.optimizer.learning_rate_schedule,
    )

  def _optimizer(
      self,
      learning_rate: jnp.ndarray,
  ) -> optax.GradientTransformation:
    optimizer_fn = getattr(optax, self.config.optimizer.name)
    return optimizer_fn(
        learning_rate=learning_rate,
        **self.config.optimizer.kwargs,
    )

  def _forward_fn(
      self,
      input_graph: jraph.GraphsTuple,
      is_training: bool,
      stop_gradient_embedding_to_logits: bool = False,
  ):
    model = models.NodePropertyEncodeProcessDecode(
        num_classes=datasets.NUM_CLASSES,
        **self.config.model_config,
    )
    return model(input_graph, is_training, stop_gradient_embedding_to_logits)

  def _bgrl_loss(
      self,
      params: hk.Params,
      ema_params: hk.Params,
      network_state: hk.State,
      ema_network_state: hk.State,
      rng: jnp.ndarray,
      batch: datasets.Batch,
  ) -> Tuple[jnp.ndarray, Tuple[losses.LogsDict, hk.State]]:
    """Computes fully supervised loss."""

    # First compute 2 graph corrupted views.
    first_corruption_key, second_corruption_key, rng = jax.random.split(rng, 3)
    (first_model_key, first_model_key_ema, second_model_key,
     second_model_key_ema, rng) = jax.random.split(rng, 5)
    first_corrupted_graph = losses.get_corrupted_view(
        batch.graph,
        rng_key=first_corruption_key,
        **self.config.training.loss_config.bgrl_loss_config.first_graph_corruption_config,  # pylint:disable=line-too-long
    )
    second_corrupted_graph = losses.get_corrupted_view(
        batch.graph,
        rng_key=second_corruption_key,
        **self.config.training.loss_config.bgrl_loss_config.second_graph_corruption_config,  # pylint:disable=line-too-long
    )

    # Then run the model on both.
    first_corrupted_output, _ = self.forward.apply(
        params,
        network_state,
        first_model_key,
        first_corrupted_graph,
        is_training=True,
        stop_gradient_embedding_to_logits=True,
    )
    second_corrupted_output, _ = self.forward.apply(
        params,
        network_state,
        second_model_key,
        second_corrupted_graph,
        is_training=True,
        stop_gradient_embedding_to_logits=True,
    )
    first_corrupted_output_ema, _ = self.forward.apply(
        ema_params,
        ema_network_state,
        first_model_key_ema,
        first_corrupted_graph,
        is_training=True,
        stop_gradient_embedding_to_logits=True,
    )
    second_corrupted_output_ema, _ = self.forward.apply(
        ema_params,
        ema_network_state,
        second_model_key_ema,
        second_corrupted_graph,
        is_training=True,
        stop_gradient_embedding_to_logits=True,
    )

    # These also contain projections for non-central nodes; remove them.
    num_nodes_per_graph = batch.graph.n_node
    node_central_indices = jnp.concatenate(
        [jnp.array([0]), jnp.cumsum(num_nodes_per_graph[:-1])])
    bgrl_loss, bgrl_stats = losses.bgrl_loss(
        first_online_predictions=first_corrupted_output
        .node_projection_predictions[node_central_indices],
        second_target_projections=second_corrupted_output_ema
        .node_embedding_projections[node_central_indices],
        second_online_predictions=second_corrupted_output
        .node_projection_predictions[node_central_indices],
        first_target_projections=first_corrupted_output_ema
        .node_embedding_projections[node_central_indices],
        symmetrize=self.config.training.loss_config.bgrl_loss_config.symmetrize,
        valid_mask=batch.central_node_mask[node_central_indices],
    )

    # Finally train decoder on original graph with optional stop gradient.
    stop_gradient = (
        self.config.training.loss_config.bgrl_loss_config
        .stop_gradient_for_supervised_loss)
    model_output, new_network_state = self.forward.apply(
        params,
        network_state,
        rng,
        batch.graph,
        is_training=True,
        stop_gradient_embedding_to_logits=stop_gradient,
    )
    supervised_loss, supervised_stats = losses.node_classification_loss(
        model_output.node_logits,
        batch,
    )
    stats = dict(**supervised_stats, **bgrl_stats)
    total_loss = (
        supervised_loss +
        self.config.training.loss_config.bgrl_loss_config.bgrl_loss_scale *
        bgrl_loss)
    return total_loss, (stats, new_network_state)

  def _loss(
      self,
      params: hk.Params,
      ema_params: hk.Params,
      network_state: hk.State,
      ema_network_state: hk.State,
      rng: jnp.ndarray,
      batch: datasets.Batch,
  ) -> Tuple[jnp.ndarray, Tuple[losses.LogsDict, hk.State]]:
    """Compute loss from params and batch."""

    # Cast to float32 since some losses are unstable with float16.
    graph = batch.graph._replace(
        nodes=jax.tree_map(lambda x: x.astype(jnp.float32), batch.graph.nodes),
        edges=jax.tree_map(lambda x: x.astype(jnp.float32), batch.graph.edges),
    )
    batch = batch._replace(graph=graph)
    return self._bgrl_loss(params, ema_params, network_state, ema_network_state,
                           rng, batch)

  def _update_func(
      self,
      params: hk.Params,
      ema_params: hk.Params,
      network_state: hk.State,
      ema_network_state: hk.State,
      opt_state: optax.OptState,
      global_step: jnp.ndarray,
      rng: jnp.ndarray,
      batch: datasets.Batch,
  ) -> Tuple[hk.Params, hk.Params, hk.State, hk.State, optax.OptState,
             losses.LogsDict]:
    """Updates parameters."""

    grad_fn = jax.value_and_grad(self._loss, has_aux=True)
    (_, (stats, new_network_state)), grads = grad_fn(
        params,
        ema_params,
        network_state,
        ema_network_state,
        rng,
        batch)
    learning_rate = self._get_learning_rate(global_step)
    _, opt_apply = self._optimizer(learning_rate)
    grad = jax.lax.pmean(grads, axis_name='i')
    updates, opt_state = opt_apply(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Stats and logging.
    param_norm = optax.global_norm(params)
    grad_norm = optax.global_norm(grad)
    ema_rate = schedules.ema_decay_schedule(
        step=global_step, **self.config.eval.ema_annealing_schedule)
    num_non_padded_nodes = (
        batch.graph.n_node.sum() -
        jraph.get_number_of_padding_with_graphs_nodes(batch.graph))
    num_non_padded_edges = (
        batch.graph.n_edge.sum() -
        jraph.get_number_of_padding_with_graphs_edges(batch.graph))
    num_non_padded_graphs = (
        batch.graph.n_node.shape[0] -
        jraph.get_number_of_padding_with_graphs_graphs(batch.graph))
    avg_num_nodes = num_non_padded_nodes / num_non_padded_graphs
    avg_num_edges = num_non_padded_edges / num_non_padded_graphs
    stats.update(
        dict(
            global_step=global_step,
            grad_norm=grad_norm,
            param_norm=param_norm,
            learning_rate=learning_rate,
            ema_rate=ema_rate,
            avg_num_nodes=avg_num_nodes,
            avg_num_edges=avg_num_edges,
        ))
    ema_fn = (lambda x, y:  # pylint:disable=g-long-lambda
              schedules.apply_ema_decay(x, y, ema_rate))
    ema_params = jax.tree_map(ema_fn, ema_params, params)
    ema_network_state = jax.tree_map(
        ema_fn,
        ema_network_state,
        network_state,
    )
    return (params, ema_params, new_network_state, ema_network_state, opt_state,
            stats)

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, **unused_kwargs):
    """See base class."""
    if not self._evaluating:
      self._eval_init()

    global_step = np.array(utils.get_first(global_step))
    ema_params = utils.get_first(self._ema_params)
    ema_network_state = utils.get_first(self._ema_network_state)
    rng = utils.get_first(rng)

    # Evaluate using the ema params.
    results, predictions = self._evaluate_with_ensemble(ema_params,
                                                        ema_network_state, rng)
    results['global_step'] = global_step

    # Store predictions if we got a path.
    self._maybe_save_predictions(predictions, global_step)

    return results

  def _evaluate_with_ensemble(
      self,
      params: hk.Params,
      state: hk.State,
      rng: jnp.ndarray,
  ):
    predictions_for_ensemble = []
    num_iterations = self.config.num_eval_iterations_to_ensemble
    for iteration in range(num_iterations):
      results, predictions = self._evaluate_params(params, state, rng)
      self._log_results(f'Eval iteration {iteration}/{num_iterations}', results)
      predictions_for_ensemble.append(predictions)

    if len(predictions_for_ensemble) > 1:
      predictions = losses.ensemble_predictions_by_probability_average(
          predictions_for_ensemble)
      results = losses.get_accuracy_dict(predictions)
      self._log_results(f'Ensembled {num_iterations} iterations', results)
    return results, predictions

  def _maybe_save_predictions(self, predictions, global_step):
    if not self.config.predictions_dir:
      return
    split = self.config.eval.split
    output_dir = os.path.join(
        self.config.predictions_dir, _get_step_date_label(global_step))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, split + '.dill')

    with open(output_path, 'wb') as f:
      dill.dump(predictions, f)
    logging.info('Saved %s predictions at: %s', split, output_path)

  def _evaluate_params(
      self,
      params: hk.Params,
      state: hk.State,
      rng: jnp.ndarray,
  ):
    """Evaluate given set of parameters."""
    num_valid = 0
    predictions_list = []
    labels_list = []
    logits_list = []
    indices_list = []
    for i, batch in enumerate(self._make_eval_dataset_iterator()):
      model_output, _ = self.eval_forward(
          params,
          state,
          rng,
          batch.graph,
      )

      (masked_indices,
       masked_predictions,
       masked_labels,
       masked_logits) = losses.get_predictions_labels_and_logits(
           model_output.node_logits, batch)
      predictions_list.append(masked_predictions)
      indices_list.append(masked_indices)
      labels_list.append(masked_labels)
      logits_list.append(masked_logits)

      num_valid += jnp.sum(batch.label_mask)

      if i % 10 == 0:
        logging.info('Generate predictons for %d batches so far', i + 1)

    predictions = losses.Predictions(
        np.concatenate(indices_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(predictions_list, axis=0),
        np.concatenate(logits_list, axis=0))

    if self.config.eval.split == 'test':
      results = dict(num_valid=num_valid, accuracy=np.nan)
    else:
      results = losses.get_accuracy_dict(predictions)

    return results, predictions

  def _log_results(self, prefix, results):
    logging_str = ', '.join(
        ['{}={:.4f}'.format(k, float(results[k]))
         for k in sorted(results.keys())])
    logging.info('%s: %s', prefix, logging_str)


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


def _get_step_date_label(global_step):
  # Date removing microseconds.
  date_str = datetime.datetime.now().isoformat().split('.')[0]
  return f'step_{global_step}_{date_str}'


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
