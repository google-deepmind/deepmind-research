# Copyright 2021 DeepMind Technologies Limited.
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

"""Generates the k-fold validation splits."""

import os
import pickle

from absl import app
from absl import flags
from absl import logging
import numpy as np

# pylint: disable=g-bad-import-order
import datasets


_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, required=True,
    help='Output directory to write the splits to')


K = 10


def main(argv):
  del argv
  valid_indices = datasets.load_splits()['valid']
  k_splits = np.split(valid_indices, K)
  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  for k_i, split in enumerate(k_splits):
    fname = os.path.join(_OUTPUT_DIR.value, f'{k_i}.pkl')
    with open(fname, 'wb') as f:
      pickle.dump(split, f)
    logging.info('Saved: %s', fname)


if __name__ == '__main__':
  app.run(main)
