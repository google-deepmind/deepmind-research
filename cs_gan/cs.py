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
"""GAN modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import sonnet as snt
import tensorflow.compat.v1 as tf

from cs_gan import utils


class CS(object):
  """Compressed Sensing Module."""

  def __init__(self, metric_net, generator,
               num_z_iters, z_step_size, z_project_method):
    """Constructs the module.

    Args:
      metric_net: the measurement network.
      generator: The generator network. A sonnet module. For examples, see
        `nets.py`.
      num_z_iters: an integer, the number of latent optimisation steps.
      z_step_size: an integer, latent optimisation step size.
      z_project_method: the method for projecting latent after optimisation,
        a string from {'norm', 'clip'}.
    """

    self._measure = metric_net
    self.generator = generator
    self.num_z_iters = num_z_iters
    self.z_project_method = z_project_method
    self._log_step_size_module = snt.TrainableVariable(
        [],
        initializers={'w': tf.constant_initializer(math.log(z_step_size))})
    self.z_step_size = tf.exp(self._log_step_size_module())

  def connect(self, data, generator_inputs):
    """Connects the components and returns the losses, outputs and debug ops.

    Args:
      data: a `tf.Tensor`: `[batch_size, ...]`. There are no constraints on the
        rank
        of this tensor, but it has to be compatible with the shapes expected
        by the discriminator.
      generator_inputs: a `tf.Tensor`: `[g_in_batch_size, ...]`. It does not
        have to have the same batch size as the `data` tensor. There are not
        constraints on the rank of this tensor, but it has to be compatible
        with the shapes the generator network supports as inputs.

    Returns:
      An `ModelOutputs` instance.
    """

    samples, optimised_z = utils.optimise_and_sample(
        generator_inputs, self, data, is_training=True)
    optimisation_cost = utils.get_optimisation_cost(generator_inputs,
                                                    optimised_z)
    debug_ops = {}

    initial_samples = self.generator(generator_inputs, is_training=True)
    generator_loss = tf.reduce_mean(self.gen_loss_fn(data, samples))
    # compute the RIP loss
    # (\sqrt{F(x_1 - x_2)^2} - \sqrt{(x_1 - x_2)^2})^2
    # as a triplet loss for 3 pairs of images.

    r1 = self._get_rip_loss(samples, initial_samples)
    r2 = self._get_rip_loss(samples, data)
    r3 = self._get_rip_loss(initial_samples, data)
    rip_loss = tf.reduce_mean((r1 + r2 + r3) / 3.0)
    total_loss = generator_loss + rip_loss
    optimization_components = self._build_optimization_components(
        generator_loss=total_loss)
    debug_ops['rip_loss'] = rip_loss
    debug_ops['recons_loss'] = tf.reduce_mean(
        tf.norm(snt.BatchFlatten()(samples)
                - snt.BatchFlatten()(data), axis=-1))

    debug_ops['z_step_size'] = self.z_step_size
    debug_ops['opt_cost'] = optimisation_cost
    debug_ops['gen_loss'] = generator_loss

    return utils.ModelOutputs(
        optimization_components, debug_ops)

  def _get_rip_loss(self, img1, img2):
    r"""Compute the RIP loss from two images.

      The RIP loss: (\sqrt{F(x_1 - x_2)^2} - \sqrt{(x_1 - x_2)^2})^2

    Args:
      img1: an image (x_1), 4D tensor of shape [batch_size, W, H, C].
      img2: an other image (x_2), 4D tensor of shape [batch_size, W, H, C].
    """

    m1 = self._measure(img1)
    m2 = self._measure(img2)

    img_diff_norm = tf.norm(snt.BatchFlatten()(img1)
                            - snt.BatchFlatten()(img2), axis=-1)
    m_diff_norm = tf.norm(m1 - m2, axis=-1)

    return tf.square(img_diff_norm - m_diff_norm)

  def _get_measurement_error(self, target_img, sample_img):
    """Compute the measurement error of sample images given the targets."""

    m_targets = self._measure(target_img)
    m_samples = self._measure(sample_img)

    return tf.reduce_sum(tf.square(m_targets - m_samples), -1)

  def gen_loss_fn(self, data, samples):
    """Generator loss as latent optimisation's error function."""
    return self._get_measurement_error(data, samples)

  def _build_optimization_components(
      self, generator_loss=None, discriminator_loss=None):
    """Create the optimization components for this module."""

    metric_vars = _get_and_check_variables(self._measure)
    generator_vars = _get_and_check_variables(self.generator)
    step_vars = _get_and_check_variables(self._log_step_size_module)

    assert discriminator_loss is None
    optimization_components = utils.OptimizationComponent(
        generator_loss, generator_vars + metric_vars + step_vars)
    return optimization_components


def _get_and_check_variables(module):
  module_variables = module.get_all_variables()
  if not module_variables:
    raise ValueError(
        'Module {} has no variables! Variables needed for training.'.format(
            module.module_name))

  # TensorFlow optimizers require lists to be passed in.
  return list(module_variables)
