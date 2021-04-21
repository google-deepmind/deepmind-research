# Copyright 2021 DeepMind Technologies Limited.
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

"""Helpers to compute loss metrics."""

import scipy.stats
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


TASK_CLASSIFICATION = 'classification'
TASK_NORMALIZED_REGRESSION = 'normalized_regression'
TASK_UNNORMALIZED_REGRESSION = 'unnormalized_regression'
TASK_GROUNDED_UNNORMALIZED_REGRESSION = 'grounded_unnormalized_regression'
REGRESSION_TASKS = [TASK_NORMALIZED_REGRESSION, TASK_UNNORMALIZED_REGRESSION,
                    TASK_GROUNDED_UNNORMALIZED_REGRESSION]
ALL_TASKS = [TASK_CLASSIFICATION] + REGRESSION_TASKS

LOSS_MSE = 'mse'
LOSS_SOFTMAX_CROSS_ENTROPY = 'softmax_cross_entropy'
ALL_LOSSES = [LOSS_SOFTMAX_CROSS_ENTROPY, LOSS_MSE]


def normalize_regression_loss(regression_loss, predictions):
  # Normalize loss such that:
  # 1) E_{x uniform}[loss(x, prediction)] does not depend on prediction
  # 2) E_{x uniform, prediction uniform}[loss(x, prediction)] is as before.
  # Divides MSE regression loss by E[(prediction-x)^2]; assumes x=[-1,1]
  normalization = 2./3.
  normalized_loss = regression_loss / ((1./3 + predictions**2) / normalization)
  return normalized_loss


def equal32(x, y):
  return tf.cast(tf.equal(x, y), tf.float32)


def mse_loss(predicted, targets):
  return (predicted - targets) ** 2


def get_std_factor_from_confidence_percent(percent):
  dec = percent/100.
  inv_dec = 1 - dec
  return scipy.stats.norm.ppf(dec+inv_dec/2)


def get_all_metric_names(task_type, model_uncertainty, loss_config,  # pylint: disable=unused-argument
                         mode='eval', return_dict=True):
  """Get all the scalar fields produced by compute_loss_and_metrics."""
  names = ['regularization_loss', 'prediction_accuracy', str(mode)+'_loss']
  if task_type == TASK_CLASSIFICATION:
    names += ['classification_loss']
  else:
    names += ['regression_loss', 'avg_mu', 'var_mu']
    if model_uncertainty:
      names += ['uncertainty_loss', 'scaled_regression_loss',
                'uncertainty_plus_scaled_regression',
                'avg_sigma', 'var_sigma',
                'percent_in_conf_interval', 'error_sigma_correlation',
                'avg_prob']
  if return_dict:
    return {name: 0. for name in names}
  else:
    return names


def compute_loss_and_metrics(mu, log_sigma_sq,
                             regression_targets, labels,
                             task_type, model_uncertainty, loss_config,
                             regularization_loss=0., confidence_interval=95,
                             mode='train'):
  """Computes loss statistics and other metrics."""

  scalars_to_log = dict()
  vectors_to_log = dict()
  scalars_to_log['regularization_loss'] = regularization_loss
  vectors_to_log['mu'] = mu

  if task_type == TASK_CLASSIFICATION:
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=mu, labels=labels, name='cross_entropy')
    classification_loss = tf.reduce_mean(cross_entropy, name='class_loss')
    total_loss = classification_loss
    sigma = None
    scalars_to_log['classification_loss'] = classification_loss

    predicted_labels = tf.argmax(mu, axis=1)
    correct_predictions = equal32(predicted_labels, labels)

  else:
    regression_loss = mse_loss(mu, regression_targets)
    if 'mse_normalize' in loss_config and loss_config['mse_normalize']:
      assert task_type in [TASK_GROUNDED_UNNORMALIZED_REGRESSION,
                           TASK_NORMALIZED_REGRESSION]
      regression_loss = normalize_regression_loss(regression_loss, mu)

    avg_regression_loss = tf.reduce_mean(regression_loss)
    vectors_to_log['regression_loss'] = regression_loss
    scalars_to_log['regression_loss'] = avg_regression_loss

    scalars_to_log['avg_mu'] = tf.reduce_mean(mu)
    scalars_to_log['var_mu'] = tf.reduce_mean(mse_loss(mu, tf.reduce_mean(mu)))

    predicted_labels = tf.cast(mu > 0, tf.int64)
    correct_predictions = equal32(predicted_labels, labels)

    if model_uncertainty:
      # This implements Eq. (1) in https://arxiv.org/pdf/1612.01474.pdf
      inv_sigma_sq = tf.math.exp(-log_sigma_sq)
      scaled_regression_loss = regression_loss * inv_sigma_sq
      scaled_regression_loss = tf.reduce_mean(scaled_regression_loss)
      uncertainty_loss = tf.reduce_mean(log_sigma_sq)
      total_loss = uncertainty_loss + scaled_regression_loss

      scalars_to_log['uncertainty_loss'] = uncertainty_loss
      scalars_to_log['scaled_regression_loss'] = scaled_regression_loss
      scalars_to_log['uncertainty_plus_scaled_regression'] = total_loss

      sigma = tf.math.exp(log_sigma_sq / 2.)
      vectors_to_log['sigma'] = sigma
      scalars_to_log['avg_sigma'] = tf.reduce_mean(sigma)
      var_sigma = tf.reduce_mean(mse_loss(sigma, tf.reduce_mean(sigma)))
      scalars_to_log['var_sigma'] = var_sigma

      # Compute # of labels that fall into the confidence interval.
      std_factor = get_std_factor_from_confidence_percent(confidence_interval)
      lower_bound = mu - std_factor *  sigma
      upper_bound = mu + std_factor *  sigma
      preds = tf.logical_and(tf.greater(regression_targets, lower_bound),
                             tf.less(regression_targets, upper_bound))
      percent_in_conf_interval = tf.reduce_mean(tf.cast(preds, tf.float32))
      scalars_to_log['percent_in_conf_interval'] = percent_in_conf_interval*100

      error_sigma_corr = tfp.stats.correlation(x=regression_loss,
                                               y=sigma, event_axis=None)
      scalars_to_log['error_sigma_correlation'] = error_sigma_corr

      dists = tfp.distributions.Normal(mu, sigma)
      probs = dists.prob(regression_targets)
      scalars_to_log['avg_prob'] = tf.reduce_mean(probs)

    else:
      total_loss = avg_regression_loss

  loss_name = str(mode)+'_loss'
  total_loss = tf.add(total_loss, regularization_loss, name=loss_name)
  scalars_to_log[loss_name] = total_loss
  vectors_to_log['correct_predictions'] = correct_predictions
  scalars_to_log['prediction_accuracy'] = tf.reduce_mean(correct_predictions)

  # Validate that metrics outputted are exactly what is expected
  expected = get_all_metric_names(task_type, model_uncertainty,
                                  loss_config, mode, False)
  assert set(expected) == set(scalars_to_log.keys())

  return scalars_to_log, vectors_to_log
