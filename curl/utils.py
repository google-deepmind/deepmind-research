################################################################################
# Copyright 2019 DeepMind Technologies Limited
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
################################################################################
"""Some common utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def generate_gaussian(logits, sigma_nonlin, sigma_param):
  """Generate a Gaussian distribution given a selected parameterisation."""

  mu, sigma = tf.split(value=logits, num_or_size_splits=2, axis=1)

  if sigma_nonlin == 'exp':
    sigma = tf.exp(sigma)
  elif sigma_nonlin == 'softplus':
    sigma = tf.nn.softplus(sigma)
  else:
    raise ValueError('Unknown sigma_nonlin {}'.format(sigma_nonlin))

  if sigma_param == 'var':
    sigma = tf.sqrt(sigma)
  elif sigma_param != 'std':
    raise ValueError('Unknown sigma_param {}'.format(sigma_param))

  return tfp.distributions.Normal(loc=mu, scale=sigma)


def construct_prior_probs(batch_size, n_y, n_y_active):
  """Construct the uniform prior probabilities.

  Args:
    batch_size: int, the size of the batch.
    n_y: int, the number of categorical cluster components.
    n_y_active: tf.Variable, the number of components that are currently in use.

  Returns:
    Tensor representing the prior probability matrix, size of [batch_size, n_y].
  """
  probs = tf.ones((batch_size, n_y_active)) / tf.cast(
      n_y_active, dtype=tf.float32)
  paddings1 = tf.stack([tf.constant(0), tf.constant(0)], axis=0)
  paddings2 = tf.stack([tf.constant(0), n_y - n_y_active], axis=0)
  paddings = tf.stack([paddings1, paddings2], axis=1)
  probs = tf.pad(probs, paddings, constant_values=1e-12)
  probs.set_shape((batch_size, n_y))
  logging.info('Prior shape: %s', str(probs.shape))
  return probs


def maybe_center_crop(layer, target_hw):
  """Center crop the layer to match a target shape."""
  l_height, l_width = layer.shape.as_list()[1:3]
  t_height, t_width = target_hw
  assert t_height <= l_height and t_width <= l_width

  if (l_height - t_height) % 2 != 0 or (l_width - t_width) % 2 != 0:
    logging.warn(
        'It is impossible to center-crop [%d, %d] into [%d, %d].'
        ' Crop will be uneven.', t_height, t_width, l_height, l_width)

  border = int((l_height - t_height) / 2)
  x_0, x_1 = border, l_height - border
  border = int((l_width - t_width) / 2)
  y_0, y_1 = border, l_width - border
  layer_cropped = layer[:, x_0:x_1, y_0:y_1, :]
  return layer_cropped
