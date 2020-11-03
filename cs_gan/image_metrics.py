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

import tensorflow_gan as tfgan


def get_image_metrics_for_samples(
    real_images, generator, prior, data_processor, num_eval_samples):
  """Compute inception score and FID."""
  max_classifier_batch = 10
  num_batches = num_eval_samples //  max_classifier_batch

  def sample_fn(arg):
    del arg
    samples = generator(prior.sample(max_classifier_batch))
    # Samples must be in [-1, 1], as expected by TFGAN.
    # Resizing to appropriate size is done by TFGAN.
    return samples

  fake_outputs = tfgan.eval.sample_and_run_inception(
      sample_fn,
      sample_inputs=[1.0] * num_batches)  # Dummy inputs.

  fake_logits = fake_outputs['logits']
  inception_score = tfgan.eval.classifier_score_from_logits(fake_logits)

  real_outputs = tfgan.eval.run_inception(
      data_processor.preprocess(real_images), num_batches=num_batches)
  fid = tfgan.eval.frechet_classifier_distance_from_activations(
      real_outputs['pool_3'], fake_outputs['pool_3'])

  return {
      'inception_score': inception_score,
      'fid': fid}
