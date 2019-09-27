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

r"""Loads images from the 80M@200K training set and saves them in PNG format.

Usage:
    cd /path/to/deepmind_research
    python -m unsupervised_adversarial_training.save_example_images \
        --data_bin_path=/path/to/tiny_images.bin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
import numpy as np
from PIL import Image

DIR_NAME = os.path.dirname(__file__)
FLAGS = flags.FLAGS
flags.DEFINE_string('data_bin_path', None,
                    'path to 80M Tiny Images data binary')
flags.DEFINE_string('idxs_path', os.path.join(DIR_NAME, 'tiny_200K_idxs.txt'),
                    'path to file of indices indicating subset of 80M dataset')
flags.DEFINE_string('output_dir', os.path.join(DIR_NAME, 'images'),
                    'path to output directory for images')
flags.mark_flag_as_required('data_bin_path')

CIFAR_LABEL_IDX_TO_NAME = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
DATASET_SIZE = 79302017


def _load_dataset_as_array(ds_path):
  dataset = np.memmap(filename=ds_path, dtype=np.uint8, mode='r',
                      shape=(DATASET_SIZE, 3, 32, 32))
  return dataset.transpose([0, 3, 2, 1])


def main(unused_argv):
  dataset = _load_dataset_as_array(FLAGS.data_bin_path)

  # Load the indices and labels of the 80M@200K training set
  data_idxs, data_labels = np.loadtxt(
      FLAGS.idxs_path,
      delimiter=',',
      dtype=[('index', np.uint64), ('label', np.uint8)],
      unpack=True)

  # Save images as PNG files
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  for i in range(100):
    class_name = CIFAR_LABEL_IDX_TO_NAME[data_labels[i]]
    file_name = 'im{}_{}.png'.format(i, class_name)
    file_path = os.path.join(FLAGS.output_dir, file_name)
    img = dataset[data_idxs[i]]
    Image.fromarray(img).save(file_path)


if __name__ == '__main__':
  app.run(main)
