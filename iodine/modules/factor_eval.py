# Copyright 2019 Deepmind Technologies Limited.
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
"""Factor Evaluation Module."""
# pylint: disable=unused-variable

import collections
import functools
from iodine.modules import utils
import shapeguard
import sonnet as snt
import tensorflow.compat.v1 as tf

Factor = collections.namedtuple("Factor", ["name", "size", "type"])


class FactorRegressor(snt.AbstractModule):
  """Assess representations by learning a linear mapping to latents."""

  def __init__(self, mapping=None, name="repres_content"):
    super().__init__(name=name)
    if mapping is None:
      self._mapping = [
          Factor("color", 3, "scalar"),
          Factor("shape", 4, "categorical"),
          Factor("scale", 1, "scalar"),
          Factor("x", 1, "scalar"),
          Factor("y", 1, "scalar"),
          Factor("orientation", 2, "angle"),
      ]
    else:
      self._mapping = [Factor(*m) for m in mapping]

  def _build(self, z, latent, visibility, pred_mask, true_mask):
    sg = shapeguard.ShapeGuard()
    z = sg.guard(z, "B, K, Z")
    pred_mask = sg.guard(pred_mask, "B, K, H, W, 1")
    true_mask = sg.guard(true_mask, "B, L, H, W, 1")

    visibility = sg.guard(visibility, "B, L")
    num_visible_obj = tf.reduce_sum(visibility)

    # Map z to predictions for all latents
    sg.M = sum([m.size for m in self._mapping])
    self.predictor = snt.Linear(sg.M, name="predict_latents")
    z_flat = sg.reshape(z, "B*K, Z")
    all_preds = sg.guard(self.predictor(z_flat), "B*K, M")
    all_preds = sg.reshape(all_preds, "B, 1, K, M")
    all_preds = tf.tile(all_preds, sg["1, L, 1, 1"])

    # prepare latents
    latents = {}
    mean_var_tot = {}
    for m in self._mapping:
      with tf.name_scope(m.name):
        # preprocess, reshape, and tile
        lat_preprocess = self.get_preprocessing(m)
        lat = sg.guard(
            lat_preprocess(latent[m.name]), "B, L, {}".format(m.size))
        # compute mean over latent by training a variable using mse
        if m.type in {"scalar", "angle"}:
          mvt = utils.OnlineMeanVarEstimator(
              axis=[0, 1], ddof=1, name="{}_mean_var".format(m.name))
          mean_var_tot[m.name] = mvt(lat, visibility[:, :, tf.newaxis])

        lat = tf.reshape(lat, sg["B, L, 1"] + [-1])
        lat = tf.tile(lat, sg["1, 1, K, 1"])
        latents[m.name] = lat

    # prepare predictions
    idx = 0
    predictions = {}
    for m in self._mapping:
      with tf.name_scope(m.name):
        assert m.name in latent, "{} not in {}".format(m.name, latent.keys())
        pred = all_preds[..., idx:idx + m.size]
        predictions[m.name] = sg.guard(pred, "B, L, K, {}".format(m.size))
        idx += m.size

    # compute error
    total_pairwise_errors = None
    for m in self._mapping:
      with tf.name_scope(m.name):
        error_fn = self.get_error_func(m)
        sg.guard(latents[m.name], "B, L, K, {}".format(m.size))
        sg.guard(predictions[m.name], "B, L, K, {}".format(m.size))
        err = error_fn(latents[m.name], predictions[m.name])
        sg.guard(err, "B, L, K")
        if total_pairwise_errors is None:
          total_pairwise_errors = err
        else:
          total_pairwise_errors += err

    # determine best assignment by comparing masks
    obj_mask = true_mask[:, :, tf.newaxis]
    pred_mask = pred_mask[:, tf.newaxis]
    pairwise_overlap = tf.reduce_sum(obj_mask * pred_mask, axis=[3, 4, 5])
    best_match = sg.guard(tf.argmax(pairwise_overlap, axis=2), "B, L")
    assignment = tf.one_hot(best_match, sg.K)
    assignment *= visibility[:, :, tf.newaxis]  # Mask non-visible objects

    # total error
    total_error = (
        tf.reduce_sum(assignment * total_pairwise_errors) / num_visible_obj)

    # compute scalars
    monitored_scalars = {}
    for m in self._mapping:
      with tf.name_scope(m.name):
        metric = self.get_metric(m)
        scalar = metric(
            latents[m.name],
            predictions[m.name],
            assignment[:, :, :, tf.newaxis],
            mean_var_tot.get(m.name),
            num_visible_obj,
        )
        monitored_scalars[m.name] = scalar
    return total_error, monitored_scalars, mean_var_tot, predictions, assignment

  @snt.reuse_variables
  def predict(self, z):
    sg = shapeguard.ShapeGuard()
    z = sg.guard(z, "B, Z")
    all_preds = sg.guard(self.predictor(z), "B, M")

    idx = 0
    predictions = {}
    for m in self._mapping:
      with tf.name_scope(m.name):
        pred = all_preds[:, idx:idx + m.size]
        predictions[m.name] = sg.guard(pred, "B, {}".format(m.size))
        idx += m.size
    return predictions

  @staticmethod
  def get_error_func(factor):
    if factor.type in {"scalar", "angle"}:
      return sse
    elif factor.type == "categorical":
      return functools.partial(
          tf.losses.softmax_cross_entropy, reduction="none")
    else:
      raise KeyError(factor.type)

  @staticmethod
  def get_metric(factor):
    if factor.type in {"scalar", "angle"}:
      return r2
    elif factor.type == "categorical":
      return accuracy
    else:
      raise KeyError(factor.type)

  @staticmethod
  def one_hot(f, nr_categories):
    return tf.one_hot(tf.cast(f[..., 0], tf.int32), depth=nr_categories)

  @staticmethod
  def angle_to_vector(theta):
    return tf.concat([tf.math.cos(theta), tf.math.sin(theta)], axis=-1)

  @staticmethod
  def get_preprocessing(factor):
    if factor.type == "scalar":
      return tf.identity
    elif factor.type == "categorical":
      return functools.partial(
          FactorRegressor.one_hot, nr_categories=factor.size)
    elif factor.type == "angle":
      return FactorRegressor.angle_to_vector
    else:
      raise KeyError(factor.type)


def sse(true, pred):
  # run our own sum squared error because we want to reduce sum over last dim
  return tf.reduce_sum(tf.square(true - pred), axis=-1)


def accuracy(labels, logits, assignment, mean_var_tot, num_vis):
  del mean_var_tot  # unused
  pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
  labels = tf.argmax(labels, axis=-1, output_type=tf.int32)
  correct = tf.cast(tf.equal(labels, pred), tf.float32)
  return tf.reduce_sum(correct * assignment[..., 0]) / num_vis


def r2(labels, pred, assignment, mean_var_tot, num_vis):
  del num_vis  # unused
  mean, var, _ = mean_var_tot
  # labels, pred: (B, L, K, n)
  ss_res = tf.reduce_sum(tf.square(labels - pred) * assignment, axis=2)
  ss_tot = var[tf.newaxis, tf.newaxis, :]  # (1, 1, n)
  return tf.reduce_mean(1.0 - ss_res / ss_tot)
