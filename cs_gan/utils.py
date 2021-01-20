# python3

# Copyright 2019 DeepMind Technologies Limited and Google LLC
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
"""Tools for latent optimisation."""
import collections
import os

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from cs_gan import nets

tfd = tfp.distributions


class ModelOutputs(
    collections.namedtuple('AdversarialModelOutputs',
                           ['optimization_components', 'debug_ops'])):
  """All the information produced by the adversarial module.

  Fields:

    * `optimization_components`: A dictionary. Each entry in this dictionary
      corresponds to a module to train using their own optimizer. The keys are
      names of the components, and the values are `common.OptimizationComponent`
      instances. The keys of this dict can be made keys of the configuration
      used by the main train loop, to define the configuration of the
      optimization details for each module.
    * `debug_ops`: A dictionary, from string to a scalar `tf.Tensor`. Quantities
      used for tracking training.
  """


class OptimizationComponent(
    collections.namedtuple('OptimizationComponent', ['loss', 'vars'])):
  """Information needed by the optimizer to train modules.

  Usage:
      `optimizer.minimize(
          opt_compoment.loss, var_list=opt_component.vars)`

  Fields:

    * `loss`: A `tf.Tensor` the loss of the module.
    * `vars`: A list of variables, the ones which will be used to minimize the
      loss.
  """


def cross_entropy_loss(logits, expected):
  """The cross entropy classification loss between logits and expected values.

  The loss proposed by the original GAN paper: https://arxiv.org/abs/1406.2661.

  Args:
    logits: a `tf.Tensor`, the model produced logits.
    expected: a `tf.Tensor`, the expected output.

  Returns:
    A scalar `tf.Tensor`, the average loss obtained on the given inputs.

  Raises:
    ValueError: if the logits do not have shape [batch_size, 2].
  """

  num_logits = logits.get_shape()[1]
  if num_logits != 2:
    raise ValueError(('Invalid number of logits for cross_entropy_loss! '
                      'cross_entropy_loss supports only 2 output logits!'))
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=expected))


def optimise_and_sample(init_z, module, data, is_training):
  """Optimising generator latent variables and sample."""

  if module.num_z_iters is None or module.num_z_iters == 0:
    z_final = init_z
  else:
    init_loop_vars = (0, _project_z(init_z, module.z_project_method))
    loop_cond = lambda i, _: i < module.num_z_iters
    def loop_body(i, z):
      loop_samples = module.generator(z, is_training)
      gen_loss = module.gen_loss_fn(data, loop_samples)
      z_grad = tf.gradients(gen_loss, z)[0]
      z -= module.z_step_size * z_grad
      z = _project_z(z, module.z_project_method)
      return i + 1, z

    # Use the following static loop for debugging
    # z = init_z
    # for _ in xrange(num_z_iters):
    #   _, z = loop_body(0, z)
    # z_final = z

    _, z_final = tf.while_loop(loop_cond,
                               loop_body,
                               init_loop_vars)

  return module.generator(z_final, is_training), z_final


def get_optimisation_cost(initial_z, optimised_z):
  optimisation_cost = tf.reduce_mean(
      tf.reduce_sum((optimised_z - initial_z)**2, -1))
  return optimisation_cost


def _project_z(z, project_method='clip'):
  """To be used for projected gradient descent over z."""
  if project_method == 'norm':
    z_p = tf.nn.l2_normalize(z, axis=-1)
  elif project_method == 'clip':
    z_p = tf.clip_by_value(z, -1, 1)
  else:
    raise ValueError('Unknown project_method: {}'.format(project_method))
  return z_p


class DataProcessor(object):

  def preprocess(self, x):
    return x * 2 - 1

  def postprocess(self, x):
    return (x + 1) / 2.


def _get_np_data(data_processor, dataset, split='train'):
  """Get the dataset as numpy arrays."""
  index = 0 if split == 'train' else 1
  if dataset == 'mnist':
    # Construct the dataset.
    x, _ = tf.keras.datasets.mnist.load_data()[index]
    # Note: tf dataset is binary so we convert it to float.
    x = x.astype(np.float32)
    x = x / 255.
    x = x.reshape((-1, 28, 28, 1))

  if dataset == 'cifar':
    x, _ = tf.keras.datasets.cifar10.load_data()[index]
    x = x.astype(np.float32)
    x = x / 255.

  if data_processor:
    # Normalize data if a processor is given.
    x = data_processor.preprocess(x)
  return x


def make_output_dir(output_dir):
  logging.info('Creating output dir %s', output_dir)
  if not tf.gfile.IsDirectory(output_dir):
    tf.gfile.MakeDirs(output_dir)


def get_ckpt_dir(output_dir):
  ckpt_dir = os.path.join(output_dir, 'ckpt')
  if not tf.gfile.IsDirectory(ckpt_dir):
    tf.gfile.MakeDirs(ckpt_dir)
  return ckpt_dir


def get_real_data_for_eval(num_eval_samples, dataset, split='valid'):
  data = _get_np_data(data_processor=None, dataset=dataset, split=split)
  data = data[:num_eval_samples]
  return tf.constant(data)


def get_summaries(ops):
  summaries = []
  for name, op in ops.items():
    # Ensure to log the value ops before writing them in the summary.
    # We do this instead of a hook to ensure IS/FID are never computed twice.
    print_op = tf.print(name, [op], output_stream=tf.logging.info)
    with tf.control_dependencies([print_op]):
      summary = tf.summary.scalar(name, op)
      summaries.append(summary)
  return summaries


def get_train_dataset(data_processor, dataset, batch_size):
  """Creates the training data tensors."""
  x_train = _get_np_data(data_processor, dataset, split='train')
  # Create the TF dataset.
  dataset = tf.data.Dataset.from_tensor_slices(x_train)

  # Shuffle and repeat the dataset for training.
  # This is required because we want to do multiple passes through the entire
  # dataset when training.
  dataset = dataset.shuffle(100000).repeat()

  # Batch the data and return the data batch.
  one_shot_iterator = dataset.batch(batch_size).make_one_shot_iterator()
  data_batch = one_shot_iterator.get_next()
  return data_batch


def get_generator(dataset):
  if dataset == 'mnist':
    return nets.MLPGeneratorNet()
  if dataset == 'cifar':
    return nets.ConvGenNet()


def get_metric_net(dataset, num_outputs=2, use_sn=True):
  if dataset == 'mnist':
    return nets.MLPMetricNet(num_outputs)
  if dataset == 'cifar':
    return nets.ConvMetricNet(num_outputs, use_sn)


def make_prior(num_latents):
  # Zero mean, unit variance prior.
  prior_mean = tf.zeros(shape=(num_latents), dtype=tf.float32)
  prior_scale = tf.ones(shape=(num_latents), dtype=tf.float32)

  return tfd.Normal(loc=prior_mean, scale=prior_scale)

