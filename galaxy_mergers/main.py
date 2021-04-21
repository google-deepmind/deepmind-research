# Copyright 2021 DeepMind Technologies Limited.
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

"""Simple script to model evaluation on a checkpoint and dataset."""

import ast
from absl import app
from absl import flags
from absl import logging

from galaxy_mergers import evaluator

flags.DEFINE_string('checkpoint_path', '', 'Path to TF2 checkpoint to eval.')
flags.DEFINE_string('data_path', '', 'Path to TFRecord(s) with data.')
flags.DEFINE_string('filter_time_intervals', None,
                    'Merger time intervals on which to perform regression.'
                    'Specify None for the default time interval [-1,1], or'
                    ' a custom list of intervals, e.g. [[-0.2,0], [0.5,1]].')

FLAGS = flags.FLAGS


def main(_) -> None:
  if FLAGS.filter_time_intervals is not None:
    filter_time_intervals = ast.literal_eval(FLAGS.filter_time_intervals)
  else:
    filter_time_intervals = None
  config, ds, experiment = evaluator.get_config_dataset_evaluator(
      filter_time_intervals,
      FLAGS.checkpoint_path,
      config_override={
          'experiment_kwargs.data_config.dataset_path': FLAGS.data_path,
      })
  metrics, _, _ = evaluator.run_model_on_dataset(experiment, ds, config)
  logging.info('Evaluation complete. Metrics: %s', metrics)


if __name__ == '__main__':
  app.run(main)
