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

"""Trains a graph-based network to predict particle mobilities in glasses."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os

from absl import app
from absl import flags

from glassy_dynamics import train

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_directory',
    '',
    'Directory which contains the train and test datasets.')
flags.DEFINE_integer(
    'time_index',
    9,
    'The time index of the target mobilities.')
flags.DEFINE_integer(
    'max_files_to_load',
    None,
    'The maximum number of files to load from the train and test datasets.')
flags.DEFINE_string(
    'checkpoint_path',
    None,
    'Path used to store a checkpoint of the best model.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_file_pattern = os.path.join(FLAGS.data_directory, 'train/aggregated*')
  test_file_pattern = os.path.join(FLAGS.data_directory, 'test/aggregated*')
  train.train_model(
      train_file_pattern=train_file_pattern,
      test_file_pattern=test_file_pattern,
      max_files_to_load=FLAGS.max_files_to_load,
      time_index=FLAGS.time_index,
      checkpoint_path=FLAGS.checkpoint_path)


if __name__ == '__main__':
  app.run(main)
