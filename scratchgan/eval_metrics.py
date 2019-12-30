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
"""Evaluation metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub


def fid(generated_sentences, real_sentences):
  """Compute FID rn sentences using pretrained universal sentence encoder.

  Args:
    generated_sentences: list of N strings.
    real_sentences: list of N strings.

  Returns:
    Frechet distance between activations.
  """
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  real_embed = embed(real_sentences)
  generated_embed = embed(generated_sentences)
  distance = tfgan.eval.frechet_classifier_distance_from_activations(
      real_embed, generated_embed)

  # Restrict the thread pool size to prevent excessive CPU usage.
  config = tf.ConfigProto()
  config.intra_op_parallelism_threads = 16
  config.inter_op_parallelism_threads = 16
  with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    distance_np = session.run(distance)
  return distance_np
