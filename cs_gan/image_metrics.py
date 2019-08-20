# coding=utf-8
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
"""Compute image metrics: IS, FID."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_gan as tfgan


def inception_preprocess_fn(images):
  return tfgan.eval.preprocess_image(images * 255)


def compute_inception_score(
    images, max_classifier_batch_size=16, assert_data_ranges=True):
  """Computes the classifier score, using the given model.

  Note that this metric is highly sensitive to the number of images used
  to compute it: more is better.  A standard number is 50K images.

  For more details see: https://arxiv.org/abs/1606.03498.

  Args:
    images: A 4D tensor.
    max_classifier_batch_size: An integer. The maximum batch size size to pass
      through the classifier used to compute the metric.
    assert_data_ranges: Whether or not to assert the input is in a valid
      data range.

  Returns:
    A scalar `tf.Tensor` with the Inception score.
  """
  num_images = images.shape.as_list()[0]

  def _choose_batch_size(num_images, max_batch_size):
    for size in range(max_batch_size, 0, -1):
      if num_images % size == 0:
        return size

  batch_size = _choose_batch_size(num_images, max_classifier_batch_size)
  tf.logging.debug('Using batch_size=%s for score classifier', batch_size)
  num_batches = num_images // batch_size

  if assert_data_ranges:
    images_batch = images[0:batch_size]
    min_data_range_check = tf.assert_greater_equal(images_batch, 0.0)
    max_data_range_check = tf.assert_less_equal(images_batch, 1.0)
    control_deps = [min_data_range_check, max_data_range_check]
  else:
    control_deps = []

  # Do the preprocessing in the fn function to avoid having to keep all the
  # resized data in memory.
  def classifier_fn(images):
    return tfgan.eval.run_inception(inception_preprocess_fn(images))

  with tf.control_dependencies(control_deps):
    return tfgan.eval.classifier_score(
        images, classifier_fn=classifier_fn, num_batches=num_batches)


def compute_fid(
    real_images, other, max_classifier_batch_size=16,
    assert_data_ranges=True):
  """Computes the classifier score, using the given model.

  Note that this metric is highly sensitive to the number of images used
  to compute it: more is better.  A standard number is 50K images.

  For more details see: https://arxiv.org/abs/1606.03498.

  Args:
    real_images: A 4D tensor. Samples from the true data distribution.
    other: A 4D tensor. Data for which to compute the FID.
    max_classifier_batch_size: An integer. The maximum batch size size to pass
      through the classifier used to compute the metric.
    assert_data_ranges: Whether or not to assert the input is in a valid
      data range.

  Returns:
    A scalar `tf.Tensor` with the Inception score.
  """
  num_images = real_images.shape.as_list()[0]

  def _choose_batch_size(num_images, max_batch_size):
    for size in range(max_batch_size, 0, -1):
      if num_images % size == 0:
        return size

  batch_size = _choose_batch_size(num_images, max_classifier_batch_size)
  tf.logging.debug('Using batch_size=%s for score classifier', batch_size)
  num_batches = num_images // batch_size

  if assert_data_ranges:
    images_batch = tf.concat(
        [real_images[0:batch_size], other[0:batch_size]], axis=0)
    min_data_range_check = tf.assert_greater_equal(images_batch, 0.0)
    max_data_range_check = tf.assert_less_equal(images_batch, 1.0)
    control_deps = [min_data_range_check, max_data_range_check]
  else:
    control_deps = []

  # Do the preprocessing in the fn function to avoid having to keep all the
  # resized data in memory.
  def classifier_fn(images):
    return tfgan.eval.run_inception(
        inception_preprocess_fn(images),
        output_tensor=tfgan.eval.INCEPTION_FINAL_POOL)

  with tf.control_dependencies(control_deps):
    return tfgan.eval.frechet_classifier_distance(
        real_images, other,
        classifier_fn=classifier_fn, num_batches=num_batches)


def generate_big_batch(generator, generator_inputs, max_num_samples=100):
  """Generate samples when the number of samples is too big to do one pass."""
  num_samples = generator_inputs.shape.as_list()[0]
  max_num_samples = min(max_num_samples, num_samples)
  batched_shape = [num_samples // max_num_samples, max_num_samples]
  batched_shape += generator_inputs.shape.as_list()[1:]
  batched_generator_inputs = tf.reshape(generator_inputs, batched_shape)

  # Create the samples by sequentially doing forward passes through the
  # generator. This ensures we do not run out of memory.
  output_samples = tf.map_fn(
      fn=generator,
      elems=batched_generator_inputs,
      parallel_iterations=1,
      back_prop=False,
      swap_memory=True)

  samples = tf.reshape(output_samples,
                       [num_samples] + output_samples.shape.as_list()[2:])
  return samples


def get_image_metrics(real_images, other):
  return {
      'inception_score': compute_inception_score(other),
      'fid': compute_fid(real_images, other)}


def get_image_metrics_for_samples(
    real_images, generator, prior, data_processor, num_eval_samples):
  generator_inputs = prior.sample(num_eval_samples)
  samples = generate_big_batch(generator, generator_inputs)
  # Ensure the samples are in [0, 1].
  samples = data_processor.postprocess(samples)

  return {
      'inception_score': compute_inception_score(samples),
      'fid': compute_fid(real_images, samples)}
