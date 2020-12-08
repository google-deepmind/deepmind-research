# Copyright 2020 DeepMind Technologies Limited.
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

"""Tests for s3d."""

from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import numpy as np

from mmv.models import normalization
from mmv.models import s3d


class _CallableS3D:
  """Wrapper around S3D that take care of parameter book keeping."""

  def __init__(self, *args, **kwargs):
    self._model = hk.transform_with_state(
        lambda *a, **k:  # pylint: disable=g-long-lambda,unnecessary-lambda
        s3d.S3D(
            normalize_fn=normalization.get_normalize_fn(),
            *args, **kwargs)(*a, **k))
    self._rng = jax.random.PRNGKey(42)
    self._params, self._state = None, None

  def init(self, inputs, **kwargs):
    self._params, self._state = self._model.init(
        self._rng, inputs, is_training=True, **kwargs)

  def __call__(self, inputs, **kwargs):
    if self._params is None:
      self.init(inputs)
    output, _ = self._model.apply(
        self._params, self._state, self._rng, inputs, **kwargs)
    return output


class S3DTest(parameterized.TestCase):

  # Testing all layers is quite slow, added in comments for completeness.
  @parameterized.parameters(
      # dict(endpoint='Conv2d_1a_7x7', expected_size=(2, 8, 112, 112, 64)),
      # dict(endpoint='MaxPool_2a_3x3', expected_size=(2, 8, 56, 56, 64)),
      # dict(endpoint='Conv2d_2b_1x1', expected_size=(2, 8, 56, 56, 64)),
      # dict(endpoint='Conv2d_2c_3x3', expected_size=(2, 8, 56, 56, 192)),
      # dict(endpoint='MaxPool_3a_3x3', expected_size=(2, 8, 28, 28, 192)),
      # dict(endpoint='Mixed_3b', expected_size=(2, 8, 28, 28, 256)),
      # dict(endpoint='Mixed_3c', expected_size=(2, 8, 28, 28, 480)),
      # dict(endpoint='MaxPool_4a_3x3', expected_size=(2, 4, 14, 14, 480)),
      # dict(endpoint='Mixed_4b', expected_size=(2, 4, 14, 14, 512)),
      # dict(endpoint='Mixed_4c', expected_size=(2, 4, 14, 14, 512)),
      # dict(endpoint='Mixed_4d', expected_size=(2, 4, 14, 14, 512)),
      # dict(endpoint='Mixed_4e', expected_size=(2, 4, 14, 14, 528)),
      # dict(endpoint='Mixed_4f', expected_size=(2, 4, 14, 14, 832)),
      # dict(endpoint='MaxPool_5a_2x2', expected_size=(2, 2, 7, 7, 832)),
      # dict(endpoint='Mixed_5b', expected_size=(2, 2, 7, 7, 832)),
      # dict(endpoint='Mixed_5c', expected_size=(2, 2, 7, 7, 1024)),
      dict(endpoint='Embeddings', expected_size=(2, 1024)),
  )
  def test_endpoint_expected_output_dimensions(self, endpoint, expected_size):
    inputs = np.random.normal(size=(2, 16, 224, 224, 3))
    model = _CallableS3D()
    output = model(inputs, is_training=False, final_endpoint=endpoint)
    self.assertSameElements(output.shape, expected_size)

  def test_space_to_depth(self):
    inputs = np.random.normal(size=(2, 16//2, 224//2, 224//2, 3*2*2*2))
    model = _CallableS3D()
    output = model(inputs, is_training=False, final_endpoint='Conv2d_1a_7x7')
    self.assertSameElements(output.shape, (2, 8, 112, 112, 64))

if __name__ == '__main__':
  absltest.main()
