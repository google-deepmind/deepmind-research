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

"""Helpers to visualize gradients and other interpretability analysis."""

import numpy as np
import tensorflow.compat.v2 as tf


def rotate_by_right_angle_multiple(image, rot=90):
  """Rotate an image by right angles."""
  if rot not in [0, 90, 180, 270]:
    raise ValueError(f"Cannot rotate by non-90 degree angle {rot}")

  if rot in [90, -270]:
    image = np.transpose(image, (1, 0, 2))
    image = image[::-1]
  elif rot in [180, -180]:
    image = image[::-1, ::-1]
  elif rot in [270, -90]:
    image = np.transpose(image, (1, 0, 2))
    image = image[:, ::-1]

  return image


def compute_gradient(images, evaluator, is_training=False):
  inputs = tf.Variable(images[None], dtype=tf.float32)
  with tf.GradientTape() as tape:
    tape.watch(inputs)
    time_sigma = evaluator.model(inputs, None, is_training)
    grad_time = tape.gradient(time_sigma[:, 0], inputs)
  return grad_time, time_sigma


def compute_grads_for_rotations(images, evaluator, is_training=False):
  test_gradients, test_outputs = [], []
  for rotation in np.arange(0, 360, 90):
    images_rot = rotate_by_right_angle_multiple(images, rotation)
    grads, time_sigma = compute_gradient(images_rot, evaluator, is_training)
    grads = np.squeeze(grads.numpy())
    inv_grads = rotate_by_right_angle_multiple(grads, -rotation)
    test_gradients.append(inv_grads)
    test_outputs.append(time_sigma.numpy())
  return np.squeeze(test_gradients), np.squeeze(test_outputs)


def compute_grads_for_rotations_and_flips(images, evaluator):
  grads, time_sigma = compute_grads_for_rotations(images, evaluator)
  grads_f, time_sigma_f = compute_grads_for_rotations(images[::-1], evaluator)
  grads_f = grads_f[:, ::-1]
  all_grads = np.concatenate([grads, grads_f], 0)
  model_outputs = np.concatenate((time_sigma, time_sigma_f), 0)
  return all_grads, model_outputs


