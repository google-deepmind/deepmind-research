# Copyright 2019 DeepMind Technologies Limited
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

"""Utility Functions for the GECO-objective.

(GECO is described in `Taming VAEs`, see https://arxiv.org/abs/1810.00597).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf


class MovingAverage(snt.AbstractModule):
  """A thin wrapper around snt.MovingAverage.

  The module adds the option not to differentiate through the last element that
  is added to the moving average, specified by means of the kwarg
  `differentiable`.
  """

  def __init__(self, decay, local=True, differentiable=False,
               name='snt_moving_average'):
    super(MovingAverage, self).__init__(name=name)
    self._differentiable = differentiable
    self._moving_average = snt.MovingAverage(
        decay=decay, local=local, name=name)

  def _build(self, inputs):
    if not self._differentiable:
      inputs = tf.stop_gradient(inputs)
    return self._moving_average(inputs)


class LagrangeMultiplier(snt.AbstractModule):
  """A lagrange multiplier sonnet module."""

  def __init__(self,
               rate=1e-2,
               name='snt_lagrange_multiplier'):
    """Initializer for the sonnet module.

    Args:
      rate: Scalar used to scale the magnitude of gradients of the Lagrange
          multipliers, defaulting to 1e-2.
      name: Name of the Lagrange multiplier sonnet module.
    """
    super(LagrangeMultiplier, self).__init__(name=name)
    self._rate = rate

  def _build(self, ma_constraint):
    """Connects the module to the graph.

    Args:
      ma_constraint: A loss minus a target value, denoting a constraint that
          shall be less or equal than zero.

    Returns:
      An op, which when added to a loss and calling minimize on the loss
      results in the optimizer minimizing w.r.t. to the model's parameters and
      maximizing w.r.t. the Lagrande multipliers, hence enforcing the
      constraints.
    """
    lagmul = snt.get_lagrange_multiplier(
        shape=ma_constraint.shape, rate=self._rate,
        initializer=np.ones(ma_constraint.shape))
    return lagmul


def _sample_gumbel(shape, eps=1e-20):
  """Transforms a uniform random variable to be standard Gumbel distributed."""

  return -tf.log(
      -tf.log(tf.random_uniform(shape, minval=0, maxval=1) + eps) + eps)


def _topk_mask(score, k):
  """Returns a mask for the top-k elements in score."""

  _, indices = tf.nn.top_k(score, k=k)
  return tf.scatter_nd(tf.expand_dims(indices, -1), tf.ones(k),
                       tf.squeeze(score).shape.as_list())


def ce_loss(logits, labels, mask=None, top_k_percentage=None,
            deterministic=False):
  """Computes the cross-entropy loss.

  Optionally a mask and a top-k percentage for the used pixels can be specified.

  The top-k mask can be produced deterministically or sampled.
  Args:
    logits: A tensor of shape (b,h,w,num_classes)
    labels: A tensor of shape (b,h,w,num_classes)
    mask: None or a tensor of shape (b,h,w).
    top_k_percentage: None or a float in (0.,1.]. If None, a standard
      cross-entropy loss is calculated.
    deterministic: A Boolean indicating whether or not to produce the
      prospective top-k mask deterministically.

  Returns:
    A dictionary holding the mean and the pixelwise sum of the loss for the
    batch as well as the employed loss mask.
  """
  num_classes = logits.shape.as_list()[-1]
  y_flat = tf.reshape(logits, (-1, num_classes), name='reshape_y')
  t_flat = tf.reshape(labels, (-1, num_classes), name='reshape_t')
  if mask is None:
    mask = tf.ones(shape=(t_flat.shape.as_list()[0],))
  else:
    assert mask.shape.as_list()[:3] == labels.shape.as_list()[:3],\
      'The loss mask shape differs from the target shape: {} vs. {}.'.format(
          mask.shape.as_list(), labels.shape.as_list()[:3])
    mask = tf.reshape(mask, (-1,), name='reshape_mask')

  n_pixels_in_batch = y_flat.shape.as_list()[0]
  xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=t_flat, logits=y_flat)

  if top_k_percentage is not None:
    assert 0.0 < top_k_percentage <= 1.0
    k_pixels = tf.cast(tf.floor(n_pixels_in_batch * top_k_percentage), tf.int32)

    stopgrad_xe = tf.stop_gradient(xe)
    norm_xe = stopgrad_xe / tf.reduce_sum(stopgrad_xe)

    if deterministic:
      score = tf.log(norm_xe)
    else:
      # Use the Gumbel trick to sample the top-k pixels, equivalent to sampling
      # from a categorical distribution over pixels whose probabilities are
      # given by the normalized cross-entropy loss values. This is done by
      # adding Gumbel noise to the logarithmic normalized cross-entropy loss
      # (followed by choosing the top-k pixels).
      score = tf.log(norm_xe) + _sample_gumbel(norm_xe.shape.as_list())

    score = score + tf.log(mask)
    top_k_mask = _topk_mask(score, k_pixels)
    mask = mask * top_k_mask

  # Calculate batch-averages for the sum and mean of the loss
  batch_size = labels.shape.as_list()[0]
  xe = tf.reshape(xe, shape=(batch_size, -1))
  mask = tf.reshape(mask, shape=(batch_size, -1))
  ce_sum_per_instance = tf.reduce_sum(mask * xe, axis=1)
  ce_sum = tf.reduce_mean(ce_sum_per_instance, axis=0)
  ce_mean = tf.reduce_sum(mask * xe) / tf.reduce_sum(mask)

  return {'mean': ce_mean, 'sum': ce_sum, 'mask': mask}
