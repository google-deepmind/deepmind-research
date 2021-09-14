# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Config for adult PSCF experiment."""

import ml_collections


def get_config():
  """Return the default configuration."""
  config = ml_collections.ConfigDict()

  config.num_steps = 10000  # Number of training steps to perform.
  config.batch_size = 128  # Batch size.
  config.learning_rate = 0.01  # Learning rate

  # Number of samples to draw for prediction.
  config.num_prediction_samples = 500

  # Batch size to use for prediction. Ideally as big as possible, but may need
  # to be reduced for memory reasons depending on the value of
  # `num_prediction_samples`.
  config.prediction_batch_size = 500

  # Multiplier for the likelihood term in the loss
  config.likelihood_multiplier = 5.

  # Multiplier for the MMD constraint term in the loss
  config.constraint_multiplier = 0.

  # Scaling factor to use in KL term.
  config.beta = 1.0

  # The number of samples we draw from each latent variable distribution.
  config.mmd_sample_size = 100

  # Directory into which results should be placed. By default it is the empty
  # string, in which case no saving will occur. The directory specified will be
  # created if it does not exist.
  config.output_dir = ''

  # The index of the step at which to turn on the constraint multiplier. For
  # steps prior to this the multiplier will be zero.
  config.constraint_turn_on_step = 0

  # The random seed for tensorflow that is applied to the graph iff the value is
  # non-negative. By default the seed is not constrained.
  config.seed = -1

  # When doing fair inference, don't sample when given a sample for the baseline
  # gender.
  config.baseline_passthrough = False

  return config
