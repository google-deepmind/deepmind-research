# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A reference training pipeline for Perceiver/Perceiver IO on ImageNet.

We use the Jaxline (https://github.com/deepmind/jaxline) training framework.
Two sets of hyperparameters are provided, the hyperparameters we used for the
Perceiver IO paper, and scaled-down hyperparameters for local testing.
This script should run out-of-the-box with the local hyper parameters.
The scaled-up hyperparameters requires a distributed learning setup to run,
and this script will need to be adapted to your specific setup.
"""

import functools
from typing import Generator, Mapping, Text, Tuple

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils
from ml_collections import config_dict
import numpy as np
import optax


from perceiver import io_processors
from perceiver import perceiver
from perceiver.train import dataset
from perceiver.train import utils

FLAGS = flags.FLAGS

OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
Scalars = Mapping[Text, jnp.ndarray]


N_TRAIN_EXAMPLES = dataset.Split.TRAIN_AND_VALID.num_examples
N_CLASSES = 1000
# Only local/debug parameters are supported out of the box.
# To use the scaled-up hyperparameters, please adapt this script to your
# training setup and set this flag to False
IS_LOCAL = True


def get_training_steps(batch_size, n_epochs):
  return (N_TRAIN_EXAMPLES * n_epochs) // batch_size


def get_config():
  """Return config object for training."""
  use_debug_settings = IS_LOCAL
  config = base_config.get_base_config()

  # Experiment config.
  local_batch_size = 2
  # Modify this to adapt to your custom distributed learning setup
  num_devices = 1
  config.train_batch_size = local_batch_size * num_devices
  config.n_epochs = 110

  def _default_or_debug(default_value, debug_value):
    return debug_value if use_debug_settings else default_value

  n_train_examples = N_TRAIN_EXAMPLES
  num_classes = N_CLASSES

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              optimizer=dict(
                  base_lr=5e-4,
                  max_norm=10.0,  # < 0 to turn off.
                  schedule_type='constant_cosine',
                  weight_decay=1e-1,
                  decay_pos_embs=True,
                  scale_by_batch=True,
                  cosine_decay_kwargs=dict(
                      init_value=0.0,
                      warmup_epochs=0,
                      end_value=0.0,
                  ),
                  step_decay_kwargs=dict(
                      decay_boundaries=[0.5, 0.8, 0.95],
                      decay_rate=0.1,
                  ),
                  constant_cosine_decay_kwargs=dict(
                      constant_fraction=0.5,
                      end_value=0.0,
                  ),
                  optimizer='lamb',
                  # Optimizer-specific kwargs:
                  adam_kwargs=dict(
                      b1=0.9,
                      b2=0.999,
                      eps=1e-8,
                  ),
                  lamb_kwargs=dict(
                      b1=0.9,
                      b2=0.999,
                      eps=1e-6,
                  ),
              ),
              # Don't specify output_channels - it's not used for
              # classifiers.
              model=dict(
                  perceiver_kwargs=dict(
                      input_preprocessor=dict(
                          prep_type='pixels',
                          # Channels for conv/conv1x1 preprocessing:
                          num_channels=64,
                          # -------------------------
                          # Position encoding arguments:
                          # -------------------------
                          position_encoding_type='fourier',
                          concat_or_add_pos='concat',
                          spatial_downsample=1,
                          # If >0, project position to this size:
                          project_pos_dim=-1,
                          trainable_position_encoding_kwargs=dict(
                              num_channels=258,  # Match default # for Fourier.
                              init_scale=0.02,
                          ),
                          fourier_position_encoding_kwargs=dict(
                              num_bands=64,
                              max_resolution=(224, 224),
                              sine_only=False,
                              concat_pos=True,
                          ),
                      ),
                      encoder=dict(
                          num_self_attends_per_block=_default_or_debug(6, 2),
                          # Weights won't be shared if num_blocks is set to 1.
                          num_blocks=_default_or_debug(8, 2),
                          z_index_dim=512,
                          num_z_channels=1024,
                          num_cross_attend_heads=1,
                          num_self_attend_heads=8,
                          cross_attend_widening_factor=1,
                          self_attend_widening_factor=1,
                          dropout_prob=0.0,
                          # Position encoding for the latent array.
                          z_pos_enc_init_scale=0.02,
                          cross_attention_shape_for_attn='kv',
                          use_query_residual=True,
                          ),
                      decoder=dict(
                          num_z_channels=1024,
                          use_query_residual=True,
                          # Position encoding for the output logits.
                          position_encoding_type='trainable',
                          trainable_position_encoding_kwargs=dict(
                              num_channels=1024,
                              init_scale=0.02,
                          ),
                      ),
                  ),
              ),
              training=dict(
                  images_per_epoch=n_train_examples,
                  label_smoothing=0.1,
                  n_epochs=config.get_oneway_ref('n_epochs'),
                  batch_size=config.get_oneway_ref('train_batch_size')
              ),
              data=dict(
                  num_classes=num_classes,
                  # Run on smaller images to debug.
                  im_dim=_default_or_debug(224, 32),
                  augmentation=dict(
                      # Typical randaug params:
                      # num_layers in [1, 3]
                      # magnitude in [5, 30]
                      # Set randaugment to None to disable.
                      randaugment=dict(
                          num_layers=4,
                          magnitude=5),
                      cutmix=True,
                      # Mixup alpha should be in [0, 1].
                      # Set to None to disable.
                      mixup_alpha=0.2,
                  ),
                  ),
              evaluation=dict(
                  subset='test',
                  batch_size=2,
              ),
          )
      )
  )

  # Training loop config.
  config.training_steps = get_training_steps(
      config.get_oneway_ref('train_batch_size'),
      config.get_oneway_ref('n_epochs'))
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60
  config.save_checkpoint_interval = 300
  config.eval_specific_checkpoint_dir = ''
  config.best_model_eval_metric = 'eval_top_1_acc'
  config.checkpoint_dir = '/tmp/perceiver_imagnet_checkpoints'
  config.train_checkpoint_all_hosts = False

  # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
  config.lock()

  return config


class Experiment(experiment.AbstractExperiment):
  """ImageNet experiment."""

  # A map from object properties that will be checkpointed to their name
  # in a checkpoint. Currently we assume that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_state': 'state',
      '_opt_state': 'opt_state',
  }

  def __init__(self, mode, init_rng, config):
    """Initializes experiment."""

    super(Experiment, self).__init__(mode=mode, init_rng=init_rng)

    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    # Checkpointed experiment state.
    self._params = None
    self._state = None
    self._opt_state = None

    # Input pipelines.
    self._train_input = None
    self._eval_input = None

    self.forward = hk.transform_with_state(self._forward_fn)

    # NOTE: We "donate" the `params, state, opt_state` arguments which allows
    # JAX (on some backends) to reuse the device memory associated with these
    # inputs to store the outputs of our function (which also start with
    # `params, state, opt_state`).
    self._update_func = jax.pmap(self._update_func, axis_name='i',
                                 donate_argnums=(0, 1, 2))
    self._eval_batch = jax.jit(self._eval_batch)

  def _forward_fn(
      self,
      inputs: dataset.Batch,
      is_training: bool,
  ) -> jnp.ndarray:

    images = inputs['images']

    perceiver_kwargs = self.config.model.perceiver_kwargs
    input_preprocessor = io_processors.ImagePreprocessor(
        **perceiver_kwargs['input_preprocessor'])
    encoder = perceiver.PerceiverEncoder(**perceiver_kwargs['encoder'])
    decoder = perceiver.ClassificationDecoder(
        self.config.data.num_classes,
        **perceiver_kwargs['decoder'])
    model = perceiver.Perceiver(
        encoder=encoder,
        decoder=decoder,
        input_preprocessor=input_preprocessor)

    return model(images, is_training=is_training)

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step: int, rng: jnp.ndarray,
           *unused_args, **unused_kwargs):
    """See base class."""

    if self._train_input is None:
      self._initialize_train()

    inputs = next(self._train_input)

    self._params, self._state, self._opt_state, scalars = (
        self._update_func(
            self._params, self._state, self._opt_state, inputs, rng, global_step
            ))

    scalars = jl_utils.get_first(scalars)
    return scalars

  def _initialize_train(self):
    self._train_input = jl_utils.py_prefetch(self._build_train_input)

    total_batch_size = self.config.training.batch_size
    steps_per_epoch = (
        self.config.training.images_per_epoch / self.config.training.batch_size)
    total_steps = self.config.training.n_epochs * steps_per_epoch
    # Scale by the (negative) learning rate.
    self._lr_schedule = utils.get_learning_rate_schedule(
        total_batch_size, steps_per_epoch, total_steps, self.config.optimizer)

    self._optimizer = utils.make_optimizer(
        self.config.optimizer,
        self._lr_schedule)

    # Check we haven't already restored params
    if self._params is None:
      logging.info('Initializing parameters.')

      inputs = next(self._train_input)

      init_net = jax.pmap(lambda *a: self.forward.init(*a, is_training=True))
      init_opt = jax.pmap(self._optimizer.init)

      # Init uses the same RNG key on all hosts+devices to ensure everyone
      # computes the same initial state.
      init_rng = jl_utils.bcast_local_devices(self.init_rng)

      self._params, self._state = init_net(init_rng, inputs)
      self._opt_state = init_opt(self._params)

  def _load_data(self, split, is_training, batch_dims):
    """Wrapper for dataset loading."""

    return dataset.load(
        split=split,
        is_training=is_training,
        batch_dims=batch_dims,
        im_dim=self.config.data.im_dim,
        augmentation_settings=self.config.data.augmentation,
        )

  def _build_train_input(self) -> Generator[dataset.Batch, None, None]:
    """See base class."""
    num_devices = jax.device_count()
    global_batch_size = self.config.training.batch_size
    per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

    if ragged:
      raise ValueError(
          f'Global batch size {global_batch_size} must be divisible by '
          f'num devices {num_devices}')

    split = dataset.Split.TRAIN_AND_VALID

    return self._load_data(
        split=split,
        is_training=True,
        batch_dims=[jax.local_device_count(), per_device_batch_size])

  def _one_hot(self, value):
    """One-hot encoding potentially over a sequence of labels."""
    y = jax.nn.one_hot(value, self.config.data.num_classes)
    return y

  def _loss_fn(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: dataset.Batch,
      rng: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Tuple[Scalars, hk.State]]:
    logits, state = self.forward.apply(
        params, state, rng, inputs, is_training=True)

    label = self._one_hot(inputs['labels'])
    # Handle cutmix/mixup label mixing:
    if 'mix_labels' in inputs:
      logging.info('Using mixup or cutmix!')
      mix_label = self._one_hot(inputs['mix_labels'])
      mix_ratio = inputs['ratio'][:, None]
      label = mix_ratio * label + (1. - mix_ratio) * mix_label

    # Apply label-smoothing to one-hot labels.
    label_smoothing = self.config.training.label_smoothing
    if not (label_smoothing >= 0. and label_smoothing < 1.):
      raise ValueError(
          f"'label_smoothing is {label_smoothing} and should be in [0, 1)")
    if label_smoothing > 0:
      smooth_positives = 1. - label_smoothing
      smooth_negatives = label_smoothing / self.config.data.num_classes
      label = smooth_positives * label + smooth_negatives

    loss_w_batch = utils.softmax_cross_entropy(logits, label)
    loss = jnp.mean(loss_w_batch, dtype=loss_w_batch.dtype)
    scaled_loss = loss / jax.device_count()

    metrics = utils.topk_correct(logits, inputs['labels'], prefix='')
    metrics = jax.tree_map(jnp.mean, metrics)

    top_1_acc = metrics['top_1_acc']
    top_5_acc = metrics['top_5_acc']

    loss_scalars = dict(
        loss=loss,
        top_1_acc=top_1_acc,
        top_5_acc=top_5_acc,
    )

    return scaled_loss, (loss_scalars, state)

  def _update_func(
      self,
      params: hk.Params,
      state: hk.State,
      opt_state: OptState,
      inputs: dataset.Batch,
      rng: jnp.ndarray,
      global_step: int,
  ) -> Tuple[hk.Params, hk.State, OptState, Scalars]:
    """Applies an update to parameters and returns new state."""
    # This function computes the gradient of the first output of loss_fn and
    # passes through the other arguments unchanged.
    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
    scaled_grads, (loss_scalars, state) = grad_loss_fn(
        params, state, inputs, rng)
    grads = jax.lax.psum(scaled_grads, axis_name='i')

    # Grab the learning rate to log before performing the step.
    learning_rate = self._lr_schedule(global_step)

    # Compute and apply updates via our optimizer.
    updates, opt_state = self._optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    n_params = 0
    for k in params.keys():
      for l in params[k]:
        n_params = n_params + np.prod(params[k][l].shape)

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = {'learning_rate': learning_rate,
               'n_params (M)': float(n_params/1e6),
               'global_gradient_norm': optax.global_norm(grads)}
    loss_scalars = {f'train_{k}': v for k, v in loss_scalars.items()}
    scalars.update(loss_scalars)
    scalars = jax.lax.pmean(scalars, axis_name='i')

    return params, state, opt_state, scalars

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, **unused_args):
    """See base class."""
    global_step = np.array(jl_utils.get_first(global_step))
    scalars = jax.device_get(self._eval_epoch(jl_utils.get_first(rng)))

    logging.info('[Step %d] Eval scalars: %s', global_step, scalars)
    return scalars

  def _eval_batch(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: dataset.Batch,
      rng: jnp.ndarray,
  ) -> Scalars:
    """Evaluates a batch."""
    logits, _ = self.forward.apply(
        params, state, rng, inputs, is_training=False)

    labels = self._one_hot(inputs['labels'])
    loss = utils.softmax_cross_entropy(logits, labels)

    metrics = utils.topk_correct(logits, inputs['labels'], prefix='')
    metrics = jax.tree_map(jnp.mean, metrics)
    top_1_acc = metrics['top_1_acc']
    top_5_acc = metrics['top_5_acc']

    bs = logits.shape[0]

    top_1_acc = jnp.expand_dims(top_1_acc, axis=0) * bs
    top_5_acc = jnp.expand_dims(top_5_acc, axis=0) * bs

    # NOTE: Returned values will be summed and finally divided by num_samples.
    return {
        'eval_loss': loss,
        'eval_top_1_acc': top_1_acc, 'eval_top_5_acc': top_5_acc}

  def _build_eval_input(self) -> Generator[dataset.Batch, None, None]:
    split = dataset.Split.from_string(self.config.evaluation.subset)

    return self._load_data(
        split=split,
        is_training=False,
        batch_dims=[self.config.evaluation.batch_size])

  def _eval_epoch(self, rng):
    """Evaluates an epoch."""
    num_samples = 0.
    summed_scalars = None

    params = jl_utils.get_first(self._params)
    state = jl_utils.get_first(self._state)

    for inputs in self._build_eval_input():
      num_samples += inputs['labels'].shape[0]
      scalars = self._eval_batch(params, state, inputs, rng)

      # Accumulate the sum of scalars for each step.
      scalars = jax.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
      if summed_scalars is None:
        summed_scalars = scalars
      else:
        summed_scalars = jax.tree_multimap(jnp.add, summed_scalars, scalars)

    mean_scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
    return mean_scalars


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))
