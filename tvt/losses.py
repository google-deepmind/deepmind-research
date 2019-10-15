# Lint as: python2, python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf


def sum_time_average_batch(tensor, name=None):
  """Computes the mean over B assuming tensor is of shape [T, B]."""
  tensor.get_shape().assert_has_rank(2)
  return tf.reduce_mean(tf.reduce_sum(tensor, axis=0), axis=0, name=name)


def combine_logged_values(*logged_values_dicts):
  """Combine logged values dicts. Throws if there are any repeated keys."""
  combined_dict = dict()
  for logged_values in logged_values_dicts:
    for k, v in six.iteritems(logged_values):
      if k in combined_dict:
        raise ValueError('Key "%s" is repeated in loss logging.' % k)
      combined_dict[k] = v
  return combined_dict


def reconstruction_losses(
    recons,
    targets,
    image_cost,
    action_cost,
    reward_cost):
  """Reconstruction losses."""
  if image_cost > 0.0:
    # Neg log prob of obs image given Bernoulli(recon image) distribution.
    negative_image_log_prob = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets.image, logits=recons.image)
    nll_per_time = tf.reduce_sum(negative_image_log_prob, [-3, -2, -1])
    image_loss = image_cost * nll_per_time
    image_loss = sum_time_average_batch(image_loss)
  else:
    image_loss = tf.constant(0.)

  if action_cost > 0.0 and recons.last_action is not tuple():
    # Labels have shape (T, B), logits have shape (T, B, num_actions).
    action_loss = action_cost * tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets.last_action, logits=recons.last_action)
    action_loss = sum_time_average_batch(action_loss)
  else:
    action_loss = tf.constant(0.)

  if reward_cost > 0.0 and recons.last_reward is not tuple():
    # MSE loss for reward.
    recon_last_reward = recons.last_reward
    recon_last_reward = tf.squeeze(recon_last_reward, -1)
    reward_loss = 0.5 * reward_cost * tf.square(
        recon_last_reward - targets.last_reward)
    reward_loss = sum_time_average_batch(reward_loss)
  else:
    reward_loss = tf.constant(0.)

  total_loss = image_loss + action_loss + reward_loss

  logged_values = dict(
      recon_loss_image=image_loss,
      recon_loss_action=action_loss,
      recon_loss_reward=reward_loss,
      total_reconstruction_loss=total_loss,)

  return total_loss, logged_values


def read_regularization_loss(
    read_info,
    strength_cost,
    strength_tolerance,
    strength_reg_mode,
    key_norm_cost,
    key_norm_tolerance):
  """Computes the sum of read strength and read key regularization losses."""

  if (strength_cost <= 0.) and (key_norm_cost <= 0.):
    read_reg_loss = tf.constant(0.)
    return read_reg_loss, dict(read_regularization_loss=read_reg_loss)

  if hasattr(read_info, 'read_strengths'):
    read_strengths = read_info.read_strengths
    read_keys = read_info.read_keys
  else:
    read_strengths = read_info.strengths
    read_keys = read_info.keys

  if read_info == tuple():
    raise ValueError('Make sure read regularization costs are zero when '
                     'not outputting read info.')

  read_reg_loss = tf.constant(0.)
  if strength_cost > 0.:
    strength_hinged = tf.maximum(strength_tolerance, read_strengths)
    if strength_reg_mode == 'L2':
      strength_loss = 0.5 * tf.square(strength_hinged)
    elif strength_reg_mode == 'L1':
      # Read strengths are always positive.
      strength_loss = strength_hinged
    else:
      raise ValueError(
          'Strength regularization mode "{}" is not supported.'.format(
              strength_reg_mode))

    # Sum across read heads to reduce from [T, B, n_reads] to [T, B].
    strength_loss = strength_cost * tf.reduce_sum(strength_loss, axis=2)

  if key_norm_cost > 0.:
    key_norm_norms = tf.norm(read_keys, axis=-1)
    key_norm_norms_hinged = tf.maximum(key_norm_tolerance, key_norm_norms)
    key_norm_loss = 0.5 * tf.square(key_norm_norms_hinged)

    # Sum across read heads to reduce from [T, B, n_reads] to [T, B].
    key_norm_loss = key_norm_cost * tf.reduce_sum(key_norm_loss, axis=2)

    read_reg_loss += key_norm_cost * key_norm_loss

  if strength_cost > 0.:
    strength_loss = sum_time_average_batch(strength_loss)
  else:
    strength_loss = tf.constant(0.)

  if key_norm_cost > 0.:
    key_norm_loss = sum_time_average_batch(key_norm_loss)
  else:
    key_norm_loss = tf.constant(0.)

  read_reg_loss = strength_loss + key_norm_loss

  logged_values = dict(
      read_reg_strength_loss=strength_loss,
      read_reg_key_norm_loss=key_norm_loss,
      total_read_reg_loss=read_reg_loss)

  return read_reg_loss, logged_values
