# Copyright 2021 Deepmind Technologies Limited.
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

"""JAXline experiment to perform robust adversarial training."""

import functools
import os
from typing import Callable, Optional, Tuple

from absl import flags
from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import utils as jl_utils
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from adversarial_robustness.jax import attacks
from adversarial_robustness.jax import datasets
from adversarial_robustness.jax import model_zoo
from adversarial_robustness.jax import utils

FLAGS = flags.FLAGS


def get_config():
  """Return config object for training."""
  config = base_config.get_base_config()

  # Batch size, training steps and data.
  num_classes = 10
  num_epochs = 400
  # Gowal et al. (2020) and Rebuffi et al. (2021) use 1024 as batch size.
  # Reducing this batch size may require further adjustments to the batch
  # normalization decay or the learning rate. If you have to use a batch size
  # of 256, reduce the number of emulated workers to 1 (it should match the
  # results of using a batch size of 1024 with 4 workers).
  train_batch_size = 1024
  def steps_from_epochs(n):
    return max(int(n * 50_000 / train_batch_size), 1)
  num_steps = steps_from_epochs(num_epochs)
  test_batch_size = train_batch_size
  # Specify the path to the downloaded data. You can download data from
  # https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness.
  # If the path is set to "cifar10_ddpm.npz" and is not found in the current
  # directory, the corresponding data will be downloaded.
  extra_npz = 'cifar10_ddpm.npz'  # Can be `None`.

  # Learning rate.
  learning_rate = .1 * max(train_batch_size / 256, 1.)
  learning_rate_warmup = steps_from_epochs(10)
  use_cosine_schedule = True
  if use_cosine_schedule:
    learning_rate_fn = utils.get_cosine_schedule(learning_rate, num_steps,
                                                 learning_rate_warmup)
  else:
    learning_rate_fn = utils.get_step_schedule(learning_rate, num_steps,
                                               learning_rate_warmup)

  # Model definition.
  model_ctor = model_zoo.WideResNet
  model_kwargs = dict(
      num_classes=num_classes,
      depth=28,
      width=10,
      activation='swish')

  # Attack used during training (can be None).
  epsilon = 8 / 255
  train_attack = attacks.UntargetedAttack(
      attacks.PGD(
          attacks.Adam(optax.piecewise_constant_schedule(
              init_value=.1,
              boundaries_and_scales={5: .1})),
          num_steps=10,
          initialize_fn=attacks.linf_initialize_fn(epsilon),
          project_fn=attacks.linf_project_fn(epsilon, bounds=(0., 1.))),
      loss_fn=attacks.untargeted_kl_divergence)

  # Attack used during evaluation (can be None).
  eval_attack = attacks.UntargetedAttack(
      attacks.PGD(
          attacks.Adam(learning_rate_fn=optax.piecewise_constant_schedule(
              init_value=.1,
              boundaries_and_scales={20: .1, 30: .01})),
          num_steps=40,
          initialize_fn=attacks.linf_initialize_fn(epsilon),
          project_fn=attacks.linf_project_fn(epsilon, bounds=(0., 1.))),
      loss_fn=attacks.untargeted_margin)

  config.experiment_kwargs = config_dict.ConfigDict(dict(config=dict(
      epsilon=epsilon,
      num_classes=num_classes,
      # Results from various publications use 4 worker machines, which results
      # in slight differences when using less worker machines. To compensate for
      # such discrepancies, we emulate these additional workers. Set to zero,
      # when using more than 4 workers.
      emulated_workers=4,
      dry_run=False,
      save_final_checkpoint_as_npy=True,
      model=dict(
          constructor=model_ctor,
          kwargs=model_kwargs),
      training=dict(
          batch_size=train_batch_size,
          learning_rate=learning_rate_fn,
          weight_decay=5e-4,
          swa_decay=.995,
          use_cutmix=False,
          supervised_batch_ratio=.3 if extra_npz is not None else 1.,
          extra_data_path=extra_npz,
          extra_label_smoothing=.1,
          attack=train_attack),
      evaluation=dict(
          # If `interval` is positive, synchronously evaluate at regular
          # intervals. Setting it to zero will not evaluate while training,
          # unless `--jaxline_mode` is set to `train_eval_multithreaded`, which
          # asynchronously evaluates checkpoints.
          interval=steps_from_epochs(40),
          batch_size=test_batch_size,
          attack=eval_attack),
  )))

  config.checkpoint_dir = '/tmp/jaxline/robust'
  config.train_checkpoint_all_hosts = False
  config.training_steps = num_steps
  config.interval_type = 'steps'
  config.log_train_data_interval = steps_from_epochs(.5)
  config.log_tensors_interval = steps_from_epochs(.5)
  config.save_checkpoint_interval = steps_from_epochs(40)
  config.eval_specific_checkpoint_dir = ''
  return config


class Experiment(experiment.AbstractExperiment):
  """CIFAR-10 experiment."""

  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_avg_params': 'avg_params',
      '_opt_state': 'opt_state',
      '_state': 'state',
  }

  def __init__(self, mode, config, init_rng):
    super().__init__(mode=mode)
    self.config = config

    self._params = None  # Network weights.
    self._avg_params = None  # Averaged network weights.
    self._state = None  # Network state (e.g., batch statistics).
    self._opt_state = None  # Optimizer state.

    # Build model.
    self.model = hk.transform_with_state(self._get_model())

    if mode == 'train':
      self._initialize_training(init_rng)
      if self.config.evaluation.interval > 0:
        self._last_evaluation_scalars = {}
        self._initialize_evaluation()
    elif mode == 'eval':
      self._initialize_evaluation()
    elif mode == 'train_eval_multithreaded':
      self._initialize_training(init_rng)
      self._initialize_evaluation()
    else:
      raise ValueError(f'Unknown mode: "{mode}"')

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step, rng, *unused_args, **unused_kwargs):
    # Get next inputs.
    supervised_inputs = next(self.supervised_train_input)
    if self.extra_train_input is None:
      extra_inputs = None
    else:
      extra_inputs = next(self.extra_train_input)

    # Perform step.
    (self._params, self._avg_params, self._state, self._opt_state,
     scalars) = self.train_fn(
         params=self._params,
         avg_params=self._avg_params,
         state=self._state,
         opt_state=self._opt_state,
         global_step=global_step,
         supervised_inputs=supervised_inputs,
         extra_inputs=extra_inputs,
         rng=rng)
    scalars = jl_utils.get_first(scalars)

    # Save final checkpoint.
    if self.config.save_final_checkpoint_as_npy and not self.config.dry_run:
      global_step_value = jl_utils.get_first(global_step)
      if global_step_value == FLAGS.config.get('training_steps', 1) - 1:
        f_np = lambda x: np.array(jax.device_get(jl_utils.get_first(x)))
        np_params = jax.tree_map(f_np, self._avg_params or self._params)
        np_state = jax.tree_map(f_np, self._state)
        path_npy = os.path.join(FLAGS.config.checkpoint_dir, 'checkpoint.npy')
        with tf.io.gfile.GFile(path_npy, 'wb') as fp:
          np.save(fp, (np_params, np_state))
        logging.info('Saved final checkpoint at %s', path_npy)

    # Run synchronous evaluation.
    if self.config.evaluation.interval <= 0:
      return scalars

    global_step_value = jl_utils.get_first(global_step)
    if (global_step_value % self.config.evaluation.interval != 0 and
        global_step_value != FLAGS.config.get('training_steps', 1) - 1):
      return _merge_eval_scalars(scalars, self._last_evaluation_scalars)
    logging.info('Running synchronous evaluation...')
    eval_scalars = self.evaluate(global_step, rng)
    f_list = lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x
    self._last_evaluation_scalars = jax.tree_map(f_list, eval_scalars)
    logging.info('(eval) global_step: %d, %s', global_step_value,
                 self._last_evaluation_scalars)
    return _merge_eval_scalars(scalars, self._last_evaluation_scalars)

  def _train_fn(self, params, avg_params, state, opt_state, global_step,
                supervised_inputs, extra_inputs, rng):
    scalars = {}
    images, labels, target_probs = self.concatenate(supervised_inputs,
                                                    extra_inputs)

    # Apply CutMix.
    if self.config.training.use_cutmix:
      aug_rng, rng = jax.random.split(rng)
      images, target_probs = utils.cutmix(aug_rng, images, target_probs,
                                          split=self._repeat_batch)

    # Perform adversarial attack.
    if self.config.training.attack is None:
      adv_images = None
      grad_fn = jax.grad(self._cross_entropy_loss_fn, has_aux=True)
    else:
      attack = self.config.training.attack
      attack_rng, rng = jax.random.split(rng)
      def logits_fn(x):
        x = self.normalize_fn(x)
        return self.model.apply(params, state, rng, x, is_training=False,
                                test_local_stats=True)[0]
      if attack.expects_labels():
        if self.config.training.use_cutmix:
          raise ValueError('Use `untargeted_kl_divergence` when using CutMix.')
        target_labels = labels
      else:
        assert attack.expects_probabilities()
        if self.config.training.use_cutmix:
          # When using CutMix, regress the attack away from mixed labels.
          target_labels = target_probs
        else:
          target_labels = jax.nn.softmax(logits_fn(images))
      adv_images = attack(logits_fn, attack_rng, images, target_labels)
      grad_fn = jax.grad(self._trades_loss_fn, has_aux=True)

    # Compute loss and gradients.
    scaled_grads, (state, loss_scalars) = grad_fn(
        params, state, images, adv_images, labels, target_probs, rng)
    grads = jax.lax.psum(scaled_grads, axis_name='i')
    scalars.update(loss_scalars)

    updates, opt_state = self.optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Stochastic weight averaging.
    if self.config.training.swa_decay > 0:
      avg_params = utils.ema_update(global_step, avg_params, params,
                                    decay_rate=self.config.training.swa_decay)

    learning_rate = self.config.training.learning_rate(global_step)
    scalars['learning_rate'] = learning_rate
    scalars = jax.lax.pmean(scalars, axis_name='i')
    return params, avg_params, state, opt_state, scalars

  def _cross_entropy_loss_fn(self, params, state, images, adv_images, labels,
                             target_probs, rng):
    scalars = {}
    images = self.normalize_fn(images)
    logits, state = self.model.apply(
        params, state, rng, images, is_training=True)
    loss = jnp.mean(utils.cross_entropy(logits, target_probs))
    loss += self.config.training.weight_decay * utils.weight_decay(params)
    if not self.config.training.use_cutmix:
      scalars['top_1_acc'] = utils.accuracy(logits, labels)
    scalars['train_loss'] = loss
    scaled_loss = loss / jax.device_count()
    return scaled_loss, (state, scalars)

  def _trades_loss_fn(self, params, state, images, adv_images, labels,
                      target_probs, rng, beta=6.):
    """Calculates TRADES loss (https://arxiv.org/pdf/1901.08573)."""
    scalars = {}

    def apply_fn(x, **norm_kwargs):
      x = self.normalize_fn(x)
      return self.model.apply(params, state, rng, x, **norm_kwargs)

    # Clean images.
    clean_logits, _ = apply_fn(images, is_training=False, test_local_stats=True)
    if not self.config.training.use_cutmix:
      scalars['top_1_acc'] = utils.accuracy(clean_logits, labels)

    # Adversarial images. Update BN stats with adversarial images.
    adv_logits, state = apply_fn(adv_images, is_training=True)
    if not self.config.training.use_cutmix:
      scalars['top_1_adv_acc'] = utils.accuracy(adv_logits, labels)

    # Compute loss.
    clean_loss = jnp.mean(utils.cross_entropy(clean_logits, target_probs))
    adv_loss = jnp.mean(utils.kl_divergence(adv_logits, clean_logits))
    reg_loss = self.config.training.weight_decay * utils.weight_decay(params)
    loss = clean_loss + beta * adv_loss + reg_loss
    scalars['train_loss'] = loss

    scaled_loss = loss / jax.device_count()
    return scaled_loss, (state, scalars)

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, *unused_args, **unused_kwargs):
    scalars = self.eval_epoch(self._params, self._state, rng)
    if self._avg_params:
      avg_scalars = self.eval_epoch(self._avg_params or self._params,
                                    self._state, rng)
      for k, v in avg_scalars.items():
        scalars[k + '_swa'] = v
    return scalars

  def eval_epoch(self, params, state, rng):
    host_id = jax.host_id()
    num_samples = 0
    batch_axis = 1
    summed_scalars = None
    # Converting to numpy here allows us to reset the generator.
    eval_input = tfds.as_numpy(self.eval_input)
    for all_inputs in eval_input:
      # The inputs are send to multiple workers.
      inputs = jax.tree_map(lambda x: x[host_id], all_inputs)
      num_samples += jax.device_count() * inputs['image'].shape[batch_axis]
      scalars = jl_utils.get_first(self.eval_fn(params, state, inputs, rng))
      # Accumulate the sum of scalars for each step.
      scalars = jax.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
      if summed_scalars is None:
        summed_scalars = scalars
      else:
        summed_scalars = jax.tree_multimap(jnp.add, summed_scalars, scalars)
    mean_scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
    return mean_scalars

  def _eval_fn(self, params, state, inputs, rng):
    images = inputs['image']
    labels = inputs['label']

    attack_rng, rng = jax.random.split(rng)
    def logits_fn(x):
      x = self.normalize_fn(x)
      return self.model.apply(params, state, rng, x, is_training=False,
                              test_local_stats=False)[0]

    # Clean accuracy.
    logits = logits_fn(images)
    predicted_label = jnp.argmax(logits, axis=-1)
    correct = jnp.equal(predicted_label, labels).astype(jnp.float32)
    scalars = {'top_1_acc': correct}

    # Adversarial accuracy.
    if self.config.evaluation.attack is not None:
      attack = self.config.evaluation.attack
      assert attack.expects_labels()
      adv_images = attack(logits_fn, attack_rng, images, labels)
      adv_logits = logits_fn(adv_images)
      predicted_label = jnp.argmax(adv_logits, axis=-1)
      correct = jnp.equal(predicted_label, labels).astype(jnp.float32)
      scalars['top_1_adv_acc'] = correct

    # Returned values will be summed and finally divided by num_samples.
    return jax.lax.psum(scalars, axis_name='i')

  def _initialize_training(self, rng):
    # Initialize inputs.
    if self.config.emulated_workers > 0:
      per_device_workers, ragged = divmod(self.config.emulated_workers,
                                          jax.host_count())
      if ragged:
        raise ValueError('Number of emulated workers must be divisible by the '
                         'number of physical workers `jax.host_count()`.')
      self._repeat_batch = per_device_workers
    else:
      self._repeat_batch = 1
    self.supervised_train_input = jl_utils.py_prefetch(
        self._supervised_train_dataset)
    if self.config.training.extra_data_path is None:
      self.extra_train_input = None
    else:
      self.extra_train_input = jl_utils.py_prefetch(
          self._extra_train_dataset)
    self.normalize_fn = datasets.cifar10_normalize

    # Optimizer.
    self.optimizer = utils.sgd_momentum(self.config.training.learning_rate,
                                        momentum=.9, nesterov=True)

    # Initialize parameters.
    if self._params is None:
      logging.info('Initializing parameters randomly rather than restoring '
                   'from checkpoint.')
      # Create inputs to initialize the network state.
      images, _, _ = jax.pmap(self.concatenate)(
          next(self.supervised_train_input),
          next(self.extra_train_input) if self.extra_train_input is not None
          else None)
      images = jax.pmap(self.normalize_fn)(images)
      # Initialize weights and biases.
      init_net = jax.pmap(
          lambda *a: self.model.init(*a, is_training=True), axis_name='i')
      init_rng = jl_utils.bcast_local_devices(rng)
      self._params, self._state = init_net(init_rng, images)
      # Setup weight averaging.
      if self.config.training.swa_decay > 0:
        self._avg_params = self._params
      else:
        self._avg_params = None
      # Initialize optimizer state.
      init_opt = jax.pmap(self.optimizer.init, axis_name='i')
      self._opt_state = init_opt(self._params)

    # Initialize step function.
    self.train_fn = jax.pmap(self._train_fn, axis_name='i',
                             donate_argnums=(0, 1, 2, 3))

  def _initialize_evaluation(self):
    load_fn = (datasets.load_dummy_data if self.config.dry_run else
               datasets.load_cifar10)
    self.eval_input = _dataset(
        functools.partial(load_fn, subset='test'),
        is_training=False, total_batch_size=self.config.evaluation.batch_size)
    self.normalize_fn = datasets.cifar10_normalize
    self.eval_fn = jax.pmap(self._eval_fn, axis_name='i')

  def _supervised_train_dataset(self) -> tfds.typing.Tree[np.ndarray]:
    """Creates the training dataset."""
    load_fn = (datasets.load_dummy_data if self.config.dry_run else
               datasets.load_cifar10)
    load_fn = functools.partial(load_fn, subset='train',
                                repeat=self._repeat_batch)
    ds = _dataset(load_fn, is_training=True, repeat=self._repeat_batch,
                  total_batch_size=self.config.training.batch_size,
                  ratio=self.config.training.supervised_batch_ratio)
    return tfds.as_numpy(ds)

  def _extra_train_dataset(self) -> tfds.typing.Tree[np.ndarray]:
    """Creates the training dataset."""
    load_fn = (datasets.load_dummy_data if self.config.dry_run else
               datasets.load_extra)
    load_fn = functools.partial(
        load_fn, path_npz=self.config.training.extra_data_path)
    ds = _dataset(
        load_fn, is_training=True, repeat=self._repeat_batch,
        total_batch_size=self.config.training.batch_size,
        one_minus_ratio=self.config.training.supervised_batch_ratio)
    return tfds.as_numpy(ds)

  def _get_model(self) -> Callable[..., chex.Array]:
    config = self.config.model
    def forward_fn(inputs, **norm_kwargs):
      model_instance = config.constructor(**config.kwargs.to_dict())
      return model_instance(inputs, **norm_kwargs)
    return forward_fn

  def concatenate(
      self,
      supervised_inputs: chex.ArrayTree,
      extra_inputs: chex.ArrayTree
  ) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Concatenate inputs."""
    num_classes = self.config.num_classes
    supervised_images = supervised_inputs['image']
    supervised_labels = supervised_inputs['label']
    if extra_inputs is None:
      images = supervised_images
      labels = supervised_labels
      target_probs = hk.one_hot(labels, num_classes)
    else:
      extra_images = extra_inputs['image']
      images = jnp.concatenate([supervised_images, extra_images], axis=0)
      extra_labels = extra_inputs['label']
      labels = jnp.concatenate([supervised_labels, extra_labels], axis=0)
      supervised_one_hot_labels = hk.one_hot(supervised_labels, num_classes)
      extra_one_hot_labels = hk.one_hot(extra_labels, num_classes)
      if self.config.training.extra_label_smoothing > 0:
        pos = 1. - self.config.training.extra_label_smoothing
        neg = self.config.training.extra_label_smoothing / num_classes
        extra_one_hot_labels = pos * extra_one_hot_labels + neg
      target_probs = jnp.concatenate(
          [supervised_one_hot_labels, extra_one_hot_labels], axis=0)
    return images, labels, target_probs


def _dataset(load_fn,
             is_training: bool,
             total_batch_size: int,
             ratio: Optional[float] = None,
             one_minus_ratio: Optional[float] = None,
             repeat: int = 1) -> tf.data.Dataset:
  """Creates a dataset."""
  num_devices = jax.device_count()
  per_device_batch_size, ragged = divmod(total_batch_size, num_devices)
  if ragged:
    raise ValueError(
        f'Global batch size {total_batch_size} must be divisible by the '
        f'total number of devices {num_devices}')
  if repeat > 1:
    if per_device_batch_size % repeat:
      raise ValueError(
          f'Per device batch size {per_device_batch_size} must be divisible '
          f'by the number of repeated batches {repeat}')
    per_device_batch_size //= repeat
  if ratio is None and one_minus_ratio is None:
    pass  # Use full batch size.
  elif one_minus_ratio is None:
    per_device_batch_size = max(
        1, min(round(per_device_batch_size * ratio),
               per_device_batch_size - 1))
  elif ratio is None:
    batch_size = max(1, min(round(per_device_batch_size * one_minus_ratio),
                            per_device_batch_size - 1))
    per_device_batch_size = per_device_batch_size - batch_size
  else:
    raise ValueError('Only one of `ratio` or `one_minus_ratio` must be '
                     'specified')
  if repeat > 1:
    per_device_batch_size *= repeat
  # When testing, we need to batch data across all devices (not just local
  # devices).
  num_local_devices = jax.local_device_count()
  if is_training:
    batch_sizes = [num_local_devices, per_device_batch_size]
  else:
    num_hosts = jax.host_count()
    assert num_hosts * num_local_devices == num_devices
    batch_sizes = [num_hosts, num_local_devices, per_device_batch_size]
  return load_fn(batch_sizes, is_training=is_training)


def _merge_eval_scalars(a, b):
  if b is None:
    return a
  for k, v in b.items():
    a['eval_' + k] = v
  return a
