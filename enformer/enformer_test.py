# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test enformer model by applying random sequence as input.

Test:

$ python enformer_test.py
"""

import random
import unittest

import enformer
import numpy as np


class TestEnformer(unittest.TestCase):

  def test_enformer(self):
    model = enformer.Enformer(channels=1536, num_transformer_layers=11)
    inputs = _get_random_input()
    outputs = model(inputs, is_training=True)
    self.assertEqual(outputs['human'].shape, (1, enformer.TARGET_LENGTH, 5313))
    self.assertEqual(outputs['mouse'].shape, (1, enformer.TARGET_LENGTH, 1643))


def _get_random_input():
  seq = ''.join(
      [random.choice('ACGT') for _ in range(enformer.SEQUENCE_LENGTH)])
  return np.expand_dims(enformer.one_hot_encode(seq), 0).astype(np.float32)


if __name__ == '__main__':
  unittest.main()
