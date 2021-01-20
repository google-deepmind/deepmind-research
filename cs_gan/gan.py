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
import collections
import math

import sonnet as snt
import tensorflow.compat.v1 as tf

from cs_gan import utils


class GAN(object):
  """Standard generative adversarial network setup.

  The aim of the generator is to generate samples which fool a discriminator.
  Does not make any assumptions about the discriminator and generator loss
  functions.

  Trained module components:

    * discriminator
    * generator


  For the standard GAN algorithm, generator_inputs is a vector of noise (either
  Gaussian or uniform).
  """

  def __init__(self, discriminator, generator,
               num_z_iters=None, z_step_size=None,
               z_project_method=None, optimisation_cost_weight=None):
    """Constructs the module.

    Args:
      discriminator: The discriminator network. A sonnet module. See `nets.py`.
      generator: The generator network. A sonnet module. For examples, see
        `nets.py`.
      num_z_iters: an integer, the number of latent optimisation steps.
      z_step_size: an integer, latent optimisation step size.
      z_project_method: the method for projecting latent after optimisation,
        a string from {'norm', 'clip'}.
      optimisation_cost_weight: a float, how much to penalise the distance of z
        moved by latent optimisation.
    """
    self._discriminator = discriminator
    self.generator = generator
    self.num_z_iters = num_z_iters
    self.z_project_method = z_project_method
    if z_step_size:
      self._log_step_size_module = snt.TrainableVariable(
          [],
          initializers={'w': tf.constant_initializer(math.log(z_step_size))})
      self.z_step_size = tf.exp(self._log_step_size_module())
    self._optimisation_cost_weight = optimisation_cost_weight

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

    # Pass in the labels to the discriminator in case we are using a
    # discriminator which makes use of labels. The labels can be None.
    disc_data_logits = self._discriminator(data)
    disc_sample_logits = self._discriminator(samples)

    disc_data_loss = utils.cross_entropy_loss(
        disc_data_logits,
        tf.ones(tf.shape(disc_data_logits[:, 0]), dtype=tf.int32))

    disc_sample_loss = utils.cross_entropy_loss(
        disc_sample_logits,
        tf.zeros(tf.shape(disc_sample_logits[:, 0]), dtype=tf.int32))

    disc_loss = disc_data_loss + disc_sample_loss

    generator_loss = utils.cross_entropy_loss(
        disc_sample_logits,
        tf.ones(tf.shape(disc_sample_logits[:, 0]), dtype=tf.int32))

    optimization_components = self._build_optimization_components(
        discriminator_loss=disc_loss, generator_loss=generator_loss,
        optimisation_cost=optimisation_cost)

    debug_ops = {}
    debug_ops['disc_data_loss'] = disc_data_loss
    debug_ops['disc_sample_loss'] = disc_sample_loss
    debug_ops['disc_loss'] = disc_loss
    debug_ops['gen_loss'] = generator_loss
    debug_ops['opt_cost'] = optimisation_cost
    if hasattr(self, 'z_step_size'):
      debug_ops['z_step_size'] = self.z_step_size

    return utils.ModelOutputs(
        optimization_components, debug_ops)

  def gen_loss_fn(self, data, samples):
    """Generator loss as latent optimisation's error function."""
    del data
    disc_sample_logits = self._discriminator(samples)
    generator_loss = utils.cross_entropy_loss(
        disc_sample_logits,
        tf.ones(tf.shape(disc_sample_logits[:, 0]), dtype=tf.int32))
    return generator_loss

  def _build_optimization_components(
      self, generator_loss=None, discriminator_loss=None,
      optimisation_cost=None):
    """Create the optimization components for this module."""

    discriminator_vars = _get_and_check_variables(self._discriminator)
    generator_vars = _get_and_check_variables(self.generator)
    if hasattr(self, '_log_step_size_module'):
      step_vars = _get_and_check_variables(self._log_step_size_module)
      generator_vars += step_vars

    optimization_components = collections.OrderedDict()
    optimization_components['disc'] = utils.OptimizationComponent(
        discriminator_loss, discriminator_vars)
    if self._optimisation_cost_weight:
      generator_loss += self._optimisation_cost_weight * optimisation_cost
    optimization_components['gen'] = utils.OptimizationComponent(
        generator_loss, generator_vars)
    return optimization_components

  def get_variables(self):
    disc_vars = _get_and_check_variables(self._discriminator)
    gen_vars = _get_and_check_variables(self.generator)
    return disc_vars, gen_vars


def _get_and_check_variables(module):
  module_variables = module.get_all_variables()
  if not module_variables:
    raise ValueError(
        'Module {} has no variables! Variables needed for training.'.format(
            module.module_name))

  # TensorFlow optimizers require lists to be passed in.
  return list(module_variables)

