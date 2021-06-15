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

"""Generate conformer features to be used for training/predictions."""

import multiprocessing as mp
import pickle
from typing import List

from absl import app
from absl import flags
import numpy as np

# pylint: disable=g-bad-import-order
import conformer_utils
import datasets

_SPLITS = flags.DEFINE_spaceseplist(
    'splits', ['test'], 'Splits to compute conformer features for.')

_OUTPUT_FILE = flags.DEFINE_string(
    'output_file',
    None,
    required=True,
    help='Output file name to write the generated conformer features to.')

_NUM_PROCS = flags.DEFINE_integer(
    'num_parallel_procs', 64,
    'Number of parallel processes to use for conformer generation.')


def generate_conformer_features(smiles: List[str]) -> List[np.ndarray]:
  # Conformer generation is a CPU-bound task and hence can get a boost from
  # parallel processing.
  # To avoid GIL, we choose multiprocessing instead of the
  # simpler multi-threading option here for parallel computing.
  with mp.Pool(_NUM_PROCS.value) as pool:
    return list(pool.map(conformer_utils.compute_conformer, smiles))


def main(_):
  smiles = datasets.load_smile_strings(with_labels=False)
  indices = set()
  for split in _SPLITS.value:
    indices.update(datasets.load_splits()[split])

  smiles = [smiles[i] for i in sorted(indices)]
  conformers = generate_conformer_features(smiles)
  smiles_to_conformers = dict(zip(smiles, conformers))

  with open(_OUTPUT_FILE.value, 'wb') as f:
    pickle.dump(smiles_to_conformers, f)


if __name__ == '__main__':
  app.run(main)
