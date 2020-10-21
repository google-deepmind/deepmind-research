# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited.
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
"""Tests for `utils.py`."""

from absl.testing import absltest

import haiku as hk
import jax
import numpy as np

from gated_linear_networks.examples import utils


class MeanStdEstimator(absltest.TestCase):

  def test_statistics(self):
    num_features = 100
    feature_size = 3
    samples = np.random.normal(
        loc=5., scale=2., size=(num_features, feature_size))
    true_mean = np.mean(samples, axis=0)
    true_std = np.std(samples, axis=0)

    def tick_(sample):
      return utils.MeanStdEstimator()(sample)

    init_fn, apply_fn = hk.without_apply_rng(hk.transform_with_state(tick_))
    tick = jax.jit(apply_fn)

    params, state = init_fn(rng=None, sample=samples[0])

    for sample in samples:
      (mean, std), state = tick(params, state, sample)

    np.testing.assert_array_almost_equal(mean, true_mean, decimal=5)
    np.testing.assert_array_almost_equal(std, true_std, decimal=5)


if __name__ == '__main__':
  absltest.main()
