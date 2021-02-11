# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
r"""Basic Jaxline ImageNet experiment."""

import importlib
import sys
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
from nfnets import dataset
from nfnets import optim
from nfnets import utils
# pylint: disable=logging-format-interpolation

FLAGS = flags.FLAGS


# We define the experiment launch config in the same file as the experiment to
# keep things self-contained in a single file, but one might consider moving the
# config and/or sweep functions to a separate file, if necessary.
def get_config():
  """Return config object for training."""
  config = base_config.get_base_config()

  # Experiment config.
  train_batch_size = 1024  # Global batch size.
  images_per_epoch = 1281167
  num_epochs = 90
  steps_per_epoch = images_per_epoch / train_batch_size
  config.training_steps = ((images_per_epoch * num_epochs) // train_batch_size)
  config.random_seed = 0
  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              lr=0.1,
              num_epochs=num_epochs,
              label_smoothing=0.1,
              model='ResNet',
              image_size=224,
              use_ema=False,
              ema_decay=0.9999,  # Quatros nuevos amigos
              ema_start=0,
              which_ema='tf1_ema',
              augment_name=None,  # 'mixup_cutmix',
              augment_before_mix=True,
              eval_preproc='crop_resize',
              train_batch_size=train_batch_size,
              eval_batch_size=50,
              eval_subset='test',
              num_classes=1000,
              which_dataset='imagenet',
              fake_data=False,
              which_loss='softmax_cross_entropy',  # For now, must be softmax
              transpose=True,  # Use the double-transpose trick?
              bfloat16=False,
              lr_schedule=dict(
                  name='WarmupCosineDecay',
                  kwargs=dict(num_steps=config.training_steps,
                              start_val=0,
                              min_val=0,
                              warmup_steps=5*steps_per_epoch),
                  ),
              lr_scale_by_bs=True,
              optimizer=dict(
                  name='SGD',
                  kwargs={'momentum': 0.9, 'nesterov': True,
                          'weight_decay': 1e-4,},
              ),
              model_kwargs=dict(
                  width=4,
                  which_norm='BatchNorm',
                  norm_kwargs=dict(create_scale=True,
                                   create_offset=True,
                                   decay_rate=0.9,
                                   ),  # cross_replica_axis='i'),
                  variant='ResNet50',
                  activation='relu',
                  drop_rate=0.0,
                  ),
          ),))

  # Training loop config: log and checkpoint every minute
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60
  config.save_checkpoint_interval = 60
  config.eval_specific_checkpoint_dir = ''

  return config


class Experiment(experiment.AbstractExperiment):
  """Imagenet experiment."""
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_state': 'state',
      '_ema_params': 'ema_params',
      '_ema_state': 'ema_state',
      '_opt_state': 'opt_state',
  }

  def __init__(self, mode, config, init_rng):
    super().__init__(mode=mode)
    self.mode = mode
    self.config = config
    self.init_rng = init_rng

    # Checkpointed experiment state.
    self._params = None
    self._state = None
    self._ema_params = None
    self._ema_state = None
    self._opt_state = None

    # Input pipelines.
    self._train_input = None
    self._eval_input = None

    # Get model, loaded in from the zoo
    self.model_module = importlib.import_module(
        ('nfnets.'+ self.config.model.lower()))
    self.net = hk.transform_with_state(self._forward_fn)

    # Assign image sizes
    if self.config.get('override_imsize', False):
      self.train_imsize = self.config.image_size
      self.test_imsize = self.config.get('eval_image_size', self.train_imsize)
    else:
      variant_dict = getattr(self.model_module, self.config.model).variant_dict
      variant_dict = variant_dict[self.config.model_kwargs.variant]
      self.train_imsize = variant_dict.get('train_imsize',
                                           self.config.image_size)
      # Test imsize defaults to model-specific value, then to config imsize
      test_imsize = self.config.get('eval_image_size', self.config.image_size)
      self.test_imsize = variant_dict.get('test_imsize', test_imsize)

    donate_argnums = (0, 1, 2, 6, 7) if self.config.use_ema else (0, 1, 2)
    self.train_fn = jax.pmap(self._train_fn, axis_name='i',
                             donate_argnums=donate_argnums)
    self.eval_fn = jax.pmap(self._eval_fn, axis_name='i')

  def _initialize_train(self):
    self._train_input = self._build_train_input()
    # Initialize net and EMA copy of net if no params available.
    if self._params is None:
      inputs = next(self._train_input)
      init_net = jax.pmap(lambda *a: self.net.init(*a, is_training=True),
                          axis_name='i')
      init_rng = jl_utils.bcast_local_devices(self.init_rng)
      self._params, self._state = init_net(init_rng, inputs)
      if self.config.use_ema:
        self._ema_params, self._ema_state = init_net(init_rng, inputs)
      num_params = hk.data_structures.tree_size(self._params)
      logging.info(f'Net parameters: {num_params / jax.local_device_count()}')
    self._make_opt()

  def _make_opt(self):
    # Separate conv params and gains/biases
    def pred(mod, name, val):  # pylint:disable=unused-argument
      return (name in ['scale', 'offset', 'b']
              or 'gain' in name or 'bias' in name)
    gains_biases, weights = hk.data_structures.partition(pred, self._params)
    # Lr schedule with batch-based LR scaling
    if self.config.lr_scale_by_bs:
      max_lr = (self.config.lr * self.config.train_batch_size) / 256
    else:
      max_lr = self.config.lr
    lr_sched_fn = getattr(optim, self.config.lr_schedule.name)
    lr_schedule = lr_sched_fn(max_val=max_lr, **self.config.lr_schedule.kwargs)
    # Optimizer; no need to broadcast!
    opt_kwargs = {key: val for key, val in self.config.optimizer.kwargs.items()}
    opt_kwargs['lr'] = lr_schedule
    opt_module = getattr(optim, self.config.optimizer.name)
    self.opt = opt_module([{'params': gains_biases, 'weight_decay': None},
                           {'params': weights}], **opt_kwargs)
    if self._opt_state is None:
      self._opt_state = self.opt.states()
    else:
      self.opt.plugin(self._opt_state)

  def _forward_fn(self, inputs, is_training):
    net_kwargs = {'num_classes': self.config.num_classes,
                  **self.config.model_kwargs}
    net = getattr(self.model_module, self.config.model)(**net_kwargs)
    if self.config.get('transpose', False):
      images = jnp.transpose(inputs['images'], (3, 0, 1, 2))  # HWCN -> NHWC
    else:
      images = inputs['images']
    if self.config.bfloat16 and self.mode == 'train':
      images = utils.to_bf16(images)
    return net(images, is_training=is_training)['logits']

  def _one_hot(self, value):
    """One-hot encoding potentially over a sequence of labels."""
    y = jax.nn.one_hot(value, self.config.num_classes)
    return y

  def _loss_fn(self, params, state, inputs, rng):
    logits, state = self.net.apply(params, state, rng, inputs, is_training=True)
    y = self._one_hot(inputs['labels'])
    if 'mix_labels' in inputs:  # Handle cutmix/mixup label mixing
      logging.info('Using mixup or cutmix!')
      y1 = self._one_hot(inputs['mix_labels'])
      y = inputs['ratio'][:, None] * y + (1. - inputs['ratio'][:, None]) * y1
    if self.config.label_smoothing > 0:  # get smoothy
      spositives = 1. - self.config.label_smoothing
      snegatives = self.config.label_smoothing / self.config.num_classes
      y = spositives * y + snegatives
    if self.config.bfloat16:  # Cast logits to float32
      logits = logits.astype(jnp.float32)
    which_loss = getattr(utils, self.config.which_loss)
    loss = which_loss(logits, y, reduction='mean')
    metrics = utils.topk_correct(logits, inputs['labels'], prefix='train_')
    # Average top-1 and top-5 correct labels
    metrics = jax.tree_map(jnp.mean, metrics)
    metrics['train_loss'] = loss  # Metrics will be pmeaned so don't divide here
    scaled_loss = loss / jax.device_count()  # Grads get psummed so do divide
    return scaled_loss, (metrics, state)

  def _train_fn(self, params, states, opt_states,
                inputs, rng, global_step,
                ema_params, ema_states):
    """Runs one batch forward + backward and run a single opt step."""
    grad_fn = jax.grad(self._loss_fn, argnums=0, has_aux=True)
    if self.config.bfloat16:
      in_params, states = jax.tree_map(utils.to_bf16, (params, states))
    else:
      in_params = params
    grads, (metrics, states) = grad_fn(in_params, states, inputs, rng)
    if self.config.bfloat16:
      states, metrics, grads = jax.tree_map(utils.from_bf16,
                                            (states, metrics, grads))
    # Sum gradients and average losses for pmap
    grads = jax.lax.psum(grads, 'i')
    metrics = jax.lax.pmean(metrics, 'i')
    # Compute updates and update parameters
    metrics['learning_rate'] = self.opt._hyperparameters['lr'](global_step)  # pylint: disable=protected-access
    params, opt_states = self.opt.step(params, grads, opt_states, global_step)
    if ema_params is not None:
      ema_fn = getattr(utils, self.config.get('which_ema', 'tf1_ema'))
      ema = lambda x, y: ema_fn(x, y, self.config.ema_decay, global_step)
      ema_params = jax.tree_multimap(ema, ema_params, params)
      ema_states = jax.tree_multimap(ema, ema_states, states)
    return {'params': params, 'states': states, 'opt_states': opt_states,
            'ema_params': ema_params, 'ema_states': ema_states,
            'metrics': metrics}

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step, rng, *unused_args, **unused_kwargs):
    if self._train_input is None:
      self._initialize_train()
    inputs = next(self._train_input)
    out = self.train_fn(params=self._params, states=self._state,
                        opt_states=self._opt_state, inputs=inputs,
                        rng=rng, global_step=global_step,
                        ema_params=self._ema_params, ema_states=self._ema_state)
    self._params, self._state = out['params'], out['states']
    self._opt_state = out['opt_states']
    self._ema_params, self._ema_state = out['ema_params'], out['ema_states']
    self.opt.plugin(self._opt_state)
    return jl_utils.get_first(out['metrics'])

  def _build_train_input(self):
    num_devices = jax.device_count()
    global_batch_size = self.config.train_batch_size
    bs_per_device, ragged = divmod(global_batch_size, num_devices)
    if ragged:
      raise ValueError(
          f'Global batch size {global_batch_size} must be divisible by '
          f'num devices {num_devices}')
    return dataset.load(
        dataset.Split.TRAIN_AND_VALID, is_training=True,
        batch_dims=[jax.local_device_count(), bs_per_device],
        transpose=self.config.get('transpose', False),
        image_size=(self.train_imsize,) * 2,
        augment_name=self.config.augment_name,
        augment_before_mix=self.config.get('augment_before_mix', True),
        name=self.config.which_dataset,
        fake_data=self.config.get('fake_data', False))

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, **unused_args):
    metrics = self._eval_epoch(self._params, self._state)
    if self.config.use_ema:
      ema_metrics = self._eval_epoch(self._ema_params, self._ema_state)
      metrics.update({f'ema_{key}': val for key, val in ema_metrics.items()})
    logging.info(f'[Step {global_step}] Eval scalars: {metrics}')
    return metrics

  def _eval_epoch(self, params, state):
    """Evaluates an epoch."""
    num_samples = 0.
    summed_metrics = None

    for inputs in self._build_eval_input():
      num_samples += np.prod(inputs['labels'].shape[:2])  # Account for pmaps
      metrics = self.eval_fn(params, state, inputs)
      # Accumulate the sum of metrics for each step.
      metrics = jax.tree_map(lambda x: jnp.sum(x[0], axis=0), metrics)
      if summed_metrics is None:
        summed_metrics = metrics
      else:
        summed_metrics = jax.tree_multimap(jnp.add, summed_metrics, metrics)
    mean_metrics = jax.tree_map(lambda x: x / num_samples, summed_metrics)
    return jax.device_get(mean_metrics)

  def _eval_fn(self, params, state, inputs):
    """Evaluate a single batch and return loss and top-k acc."""
    logits, _ = self.net.apply(params, state, None, inputs, is_training=False)
    y = self._one_hot(inputs['labels'])
    which_loss = getattr(utils, self.config.which_loss)
    loss = which_loss(logits, y, reduction=None)
    metrics = utils.topk_correct(logits, inputs['labels'], prefix='eval_')
    metrics['eval_loss'] = loss
    return jax.lax.psum(metrics, 'i')

  def _build_eval_input(self):
    """Builds the evaluation input pipeline."""
    bs_per_device = (self.config.eval_batch_size // jax.local_device_count())
    split = dataset.Split.from_string(self.config.eval_subset)
    eval_preproc = self.config.get('eval_preproc', 'crop_resize')
    return dataset.load(split, is_training=False,
                        batch_dims=[jax.local_device_count(), bs_per_device],
                        transpose=self.config.get('transpose', False),
                        image_size=(self.test_imsize,) * 2,
                        name=self.config.which_dataset,
                        eval_preproc=eval_preproc,
                        fake_data=self.config.get('fake_data', False))

if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  platform.main(Experiment, sys.argv[1:])
