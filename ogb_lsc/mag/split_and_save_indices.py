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

"""Split and save the train/valid/test indices.

Usage:

python3 split_and_save_indices.py --data_root="mag_data"
"""

import pathlib

from absl import app
from absl import flags
import numpy as np
import torch

Path = pathlib.Path


FLAGS = flags.FLAGS

flags.DEFINE_string('data_root', None, 'Data root directory')


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  mag_directory = Path(FLAGS.data_root) / 'mag240m_kddcup2021'
  raw_directory = mag_directory / 'raw'
  raw_directory.parent.mkdir(parents=True, exist_ok=True)
  splits_dict = torch.load(str(mag_directory / 'split_dict.pt'))
  for key, indices in splits_dict.items():
    np.save(str(raw_directory / f'{key}_idx.npy'), indices)


if __name__ == '__main__':
  flags.mark_flag_as_required('root')
  app.run(main)
