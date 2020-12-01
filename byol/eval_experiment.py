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

"""Linear evaluation or fine-tuning pipeline.

Use this experiment to evaluate a checkpoint from byol_experiment.
"""

import functools
from typing import Any, Generator, Mapping, NamedTuple, Optional, Text, Tuple, Union

from absl import logging
from acme.jax import utils as acme_utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from byol.utils import checkpointing
from byol.utils import dataset
from byol.utils import helpers
from byol.utils import networks
from byol.utils import schedules

# Type declarations.
OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
LogsDict = Mapping[Text, jnp.ndarray]


class _EvalExperimentState(NamedTuple):
  backbone_params: hk.Params
  classif_params: hk.Params
  backbone_state: hk.State
  backbone_opt_state: Union[None, OptState]
  classif_opt_state: OptState


class EvalExperiment:
  """Linear evaluation experiment."""

  def __init__(
      self,
      random_seed: int,
      num_classes: int,
      batch_size: int,
      max_steps: int,
      enable_double_transpose: bool,
      checkpoint_to_evaluate: Optional[Text],
      allow_train_from_scratch: bool,
      freeze_backbone: bool,
      network_config: Mapping[Text, Any],
      optimizer_config: Mapping[Text, Any],
      lr_schedule_config: Mapping[Text, Any],
      evaluation_config: Mapping[Text, Any],
      checkpointing_config: Mapping[Text, Any]):
    """Constructs the experiment.

    Args:
      random_seed: the random seed to use when initializing network weights.
      num_classes: the number of classes; used for the online evaluation.
      batch_size: the total batch size; should be a multiple of the number of
        available accelerators.
      max_steps: the number of training steps; used for the lr/target network
        ema schedules.
      enable_double_transpose: see dataset.py; only has effect on TPU.
      checkpoint_to_evaluate: the path to the checkpoint to evaluate.
      allow_train_from_scratch: whether to allow training without specifying a
        checkpoint to evaluate (training from scratch).
      freeze_backbone: whether the backbone resnet should remain frozen (linear
        evaluation) or be trainable (fine-tuning).
      network_config: the configuration for the network.
      optimizer_config: the configuration for the optimizer.
      lr_schedule_config: the configuration for the learning rate schedule.
      evaluation_config: the evaluation configuration.
      checkpointing_config: the configuration for checkpointing.
    """

    self._random_seed = random_seed
    self._enable_double_transpose = enable_double_transpose
    self._num_classes = num_classes
    self._lr_schedule_config = lr_schedule_config
    self._batch_size = batch_size
    self._max_steps = max_steps
    self._checkpoint_to_evaluate = checkpoint_to_evaluate
    self._allow_train_from_scratch = allow_train_from_scratch
    self._freeze_backbone = freeze_backbone
    self._optimizer_config = optimizer_config
    self._evaluation_config = evaluation_config

    # Checkpointed experiment state.
    self._experiment_state = None

    # Input pipelines.
    self._train_input = None
    self._eval_input = None

    backbone_fn = functools.partial(self._backbone_fn, **network_config)
    self.forward_backbone = hk.without_apply_rng(
        hk.transform_with_state(backbone_fn))
    self.forward_classif = hk.without_apply_rng(hk.transform(self._classif_fn))
    self.update_pmap = jax.pmap(self._update_func, axis_name='i')
    self.eval_batch_jit = jax.jit(self._eval_batch)

    self._is_backbone_training = not self._freeze_backbone

    self._checkpointer = checkpointing.Checkpointer(**checkpointing_config)

  def _should_transpose_images(self):
    """Should we transpose images (saves host-to-device time on TPUs)."""
    return (self._enable_double_transpose and
            jax.local_devices()[0].platform == 'tpu')

  def _backbone_fn(
      self,
      inputs: dataset.Batch,
      encoder_class: Text,
      encoder_config: Mapping[Text, Any],
      bn_decay_rate: float,
      is_training: bool,
  ) -> jnp.ndarray:
    """Forward of the encoder (backbone)."""
    bn_config = {'decay_rate': bn_decay_rate}
    encoder = getattr(networks, encoder_class)
    model = encoder(
        None,
        bn_config=bn_config,
        **encoder_config)

    if self._should_transpose_images():
      inputs = dataset.transpose_images(inputs)
    images = dataset.normalize_images(inputs['images'])
    return model(images, is_training=is_training)

  def _classif_fn(
      self,
      embeddings: jnp.ndarray,
  ) -> jnp.ndarray:
    classifier = hk.Linear(output_size=self._num_classes)
    return classifier(embeddings)

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, *,
           global_step: jnp.ndarray,
           rng: jnp.ndarray) -> Mapping[Text, np.ndarray]:
    """Performs a single training step."""

    if self._train_input is None:
      self._initialize_train(rng)

    inputs = next(self._train_input)
    self._experiment_state, scalars = self.update_pmap(
        self._experiment_state, global_step, inputs)

    scalars = helpers.get_first(scalars)
    return scalars

  def save_checkpoint(self, step: int, rng: jnp.ndarray):
    self._checkpointer.maybe_save_checkpoint(
        self._experiment_state, step=step, rng=rng,
        is_final=step >= self._max_steps)

  def load_checkpoint(self) -> Union[Tuple[int, jnp.ndarray], None]:
    checkpoint_data = self._checkpointer.maybe_load_checkpoint()
    if checkpoint_data is None:
      return None
    self._experiment_state, step, rng = checkpoint_data
    return step, rng

  def _initialize_train(self, rng):
    """BYOL's _ExperimentState initialization.

    Args:
      rng: random number generator used to initialize parameters. If working in
        a multi device setup, this need to be a ShardedArray.
      dummy_input: a dummy image, used to compute intermediate outputs shapes.

    Returns:
      Initial EvalExperiment state.

    Raises:
      RuntimeError: invalid or empty checkpoint.
    """
    self._train_input = acme_utils.prefetch(self._build_train_input())

    # Check we haven't already restored params
    if self._experiment_state is None:

      inputs = next(self._train_input)

      if self._checkpoint_to_evaluate is not None:
        # Load params from checkpoint
        checkpoint_data = checkpointing.load_checkpoint(
            self._checkpoint_to_evaluate)
        if checkpoint_data is None:
          raise RuntimeError('Invalid checkpoint.')
        backbone_params = checkpoint_data['experiment_state'].online_params
        backbone_state = checkpoint_data['experiment_state'].online_state
        backbone_params = helpers.bcast_local_devices(backbone_params)
        backbone_state = helpers.bcast_local_devices(backbone_state)
      else:
        if not self._allow_train_from_scratch:
          raise ValueError(
              'No checkpoint specified, but `allow_train_from_scratch` '
              'set to False')
        # Initialize with random parameters
        logging.info(
            'No checkpoint specified, initializing the networks from scratch '
            '(dry run mode)')
        backbone_params, backbone_state = jax.pmap(
            functools.partial(self.forward_backbone.init, is_training=True),
            axis_name='i')(rng=rng, inputs=inputs)

      init_experiment = jax.pmap(self._make_initial_state, axis_name='i')

      # Init uses the same RNG key on all hosts+devices to ensure everyone
      # computes the same initial state and parameters.
      init_rng = jax.random.PRNGKey(self._random_seed)
      init_rng = helpers.bcast_local_devices(init_rng)
      self._experiment_state = init_experiment(
          rng=init_rng,
          dummy_input=inputs,
          backbone_params=backbone_params,
          backbone_state=backbone_state)

      # Clear the backbone optimizer's state when the backbone is frozen.
      if self._freeze_backbone:
        self._experiment_state = _EvalExperimentState(
            backbone_params=self._experiment_state.backbone_params,
            classif_params=self._experiment_state.classif_params,
            backbone_state=self._experiment_state.backbone_state,
            backbone_opt_state=None,
            classif_opt_state=self._experiment_state.classif_opt_state,
        )

  def _make_initial_state(
      self,
      rng: jnp.ndarray,
      dummy_input: dataset.Batch,
      backbone_params: hk.Params,
      backbone_state: hk.Params,
  ) -> _EvalExperimentState:
    """_EvalExperimentState initialization."""

    # Initialize the backbone params
    # Always create the batchnorm weights (is_training=True), they will be
    # overwritten when loading the checkpoint.
    embeddings, _ = self.forward_backbone.apply(
        backbone_params, backbone_state, dummy_input, is_training=True)
    backbone_opt_state = self._optimizer(0.).init(backbone_params)

    # Initialize the classifier params and optimizer_state
    classif_params = self.forward_classif.init(rng, embeddings)
    classif_opt_state = self._optimizer(0.).init(classif_params)

    return _EvalExperimentState(
        backbone_params=backbone_params,
        classif_params=classif_params,
        backbone_state=backbone_state,
        backbone_opt_state=backbone_opt_state,
        classif_opt_state=classif_opt_state,
    )

  def _build_train_input(self) -> Generator[dataset.Batch, None, None]:
    """See base class."""
    num_devices = jax.device_count()
    global_batch_size = self._batch_size
    per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

    if ragged:
      raise ValueError(
          f'Global batch size {global_batch_size} must be divisible by '
          f'num devices {num_devices}')

    return dataset.load(
        dataset.Split.TRAIN_AND_VALID,
        preprocess_mode=dataset.PreprocessMode.LINEAR_TRAIN,
        transpose=self._should_transpose_images(),
        batch_dims=[jax.local_device_count(), per_device_batch_size])

  def _optimizer(self, learning_rate: float):
    """Build optimizer from config."""
    return optax.sgd(learning_rate, **self._optimizer_config)

  def _loss_fn(
      self,
      backbone_params: hk.Params,
      classif_params: hk.Params,
      backbone_state: hk.State,
      inputs: dataset.Batch,
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, hk.State]]:
    """Compute the classification loss function.

    Args:
      backbone_params: parameters of the encoder network.
      classif_params: parameters of the linear classifier.
      backbone_state: internal state of encoder network.
      inputs: inputs, containing `images` and `labels`.

    Returns:
      The classification loss and various logs.
    """
    embeddings, backbone_state = self.forward_backbone.apply(
        backbone_params,
        backbone_state,
        inputs,
        is_training=not self._freeze_backbone)

    logits = self.forward_classif.apply(classif_params, embeddings)
    labels = hk.one_hot(inputs['labels'], self._num_classes)
    loss = helpers.softmax_cross_entropy(logits, labels, reduction='mean')
    scaled_loss = loss / jax.device_count()

    return scaled_loss, (loss, backbone_state)

  def _update_func(
      self,
      experiment_state: _EvalExperimentState,
      global_step: jnp.ndarray,
      inputs: dataset.Batch,
  ) -> Tuple[_EvalExperimentState, LogsDict]:
    """Applies an update to parameters and returns new state."""
    # This function computes the gradient of the first output of loss_fn and
    # passes through the other arguments unchanged.

    # Gradient of the first output of _loss_fn wrt the backbone (arg 0) and the
    # classifier parameters (arg 1). The auxiliary outputs are returned as-is.
    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True, argnums=(0, 1))

    grads, aux_outputs = grad_loss_fn(
        experiment_state.backbone_params,
        experiment_state.classif_params,
        experiment_state.backbone_state,
        inputs,
    )
    backbone_grads, classifier_grads = grads
    train_loss, new_backbone_state = aux_outputs
    classifier_grads = jax.lax.psum(classifier_grads, axis_name='i')

    # Compute the decayed learning rate
    learning_rate = schedules.learning_schedule(
        global_step,
        batch_size=self._batch_size,
        total_steps=self._max_steps,
        **self._lr_schedule_config)

    # Compute and apply updates via our optimizer.
    classif_updates, new_classif_opt_state = \
        self._optimizer(learning_rate).update(
            classifier_grads,
            experiment_state.classif_opt_state)

    new_classif_params = optax.apply_updates(experiment_state.classif_params,
                                             classif_updates)

    if self._freeze_backbone:
      del backbone_grads, new_backbone_state  # Unused
      # The backbone is not updated.
      new_backbone_params = experiment_state.backbone_params
      new_backbone_opt_state = None
      new_backbone_state = experiment_state.backbone_state
    else:
      backbone_grads = jax.lax.psum(backbone_grads, axis_name='i')

      # Compute and apply updates via our optimizer.
      backbone_updates, new_backbone_opt_state = \
          self._optimizer(learning_rate).update(
              backbone_grads,
              experiment_state.backbone_opt_state)

      new_backbone_params = optax.apply_updates(
          experiment_state.backbone_params, backbone_updates)

    experiment_state = _EvalExperimentState(
        new_backbone_params,
        new_classif_params,
        new_backbone_state,
        new_backbone_opt_state,
        new_classif_opt_state,
    )

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = {'train_loss': train_loss}
    scalars = jax.lax.pmean(scalars, axis_name='i')

    return experiment_state, scalars

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, **unused_args):
    """See base class."""

    global_step = np.array(helpers.get_first(global_step))
    scalars = jax.device_get(self._eval_epoch(**self._evaluation_config))

    logging.info('[Step %d] Eval scalars: %s', global_step, scalars)
    return scalars

  def _eval_batch(
      self,
      backbone_params: hk.Params,
      classif_params: hk.Params,
      backbone_state: hk.State,
      inputs: dataset.Batch,
  ) -> LogsDict:
    """Evaluates a batch."""
    embeddings, backbone_state = self.forward_backbone.apply(
        backbone_params, backbone_state, inputs, is_training=False)
    logits = self.forward_classif.apply(classif_params, embeddings)
    labels = hk.one_hot(inputs['labels'], self._num_classes)
    loss = helpers.softmax_cross_entropy(logits, labels, reduction=None)
    top1_correct = helpers.topk_accuracy(logits, inputs['labels'], topk=1)
    top5_correct = helpers.topk_accuracy(logits, inputs['labels'], topk=5)
    # NOTE: Returned values will be summed and finally divided by num_samples.
    return {
        'eval_loss': loss,
        'top1_accuracy': top1_correct,
        'top5_accuracy': top5_correct
    }

  def _eval_epoch(self, subset: Text, batch_size: int):
    """Evaluates an epoch."""
    num_samples = 0.
    summed_scalars = None

    backbone_params = helpers.get_first(self._experiment_state.backbone_params)
    classif_params = helpers.get_first(self._experiment_state.classif_params)
    backbone_state = helpers.get_first(self._experiment_state.backbone_state)
    split = dataset.Split.from_string(subset)

    dataset_iterator = dataset.load(
        split,
        preprocess_mode=dataset.PreprocessMode.EVAL,
        transpose=self._should_transpose_images(),
        batch_dims=[batch_size])

    for inputs in dataset_iterator:
      num_samples += inputs['labels'].shape[0]
      scalars = self.eval_batch_jit(
          backbone_params,
          classif_params,
          backbone_state,
          inputs,
      )

      # Accumulate the sum of scalars for each step.
      scalars = jax.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
      if summed_scalars is None:
        summed_scalars = scalars
      else:
        summed_scalars = jax.tree_multimap(jnp.add, summed_scalars, scalars)

    mean_scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
    return mean_scalars
