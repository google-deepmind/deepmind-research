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
r"""DM control suite and locomotion dataset examples.

Example:
Instructions:
> export TMP_PATH=/tmp/dataset
> export TASK_NAME=humanoid_run
> mkdir -p $TMP_PATH/$TASK_NAME
> gsutil cp gs://rl_unplugged/dm_control_suite/$TASK_NAME/train-00000-of-00100 \
$TMP_PATH/dm_control_suite/$TASK_NAME/train-00000-of-00001
> python dm_control_suite_example.py --path=$TMP_PATH \
--task_class=control_suite --task_name=$TASK_NAME
"""

from absl import app
from absl import flags
import tree

from rl_unplugged import dm_control_suite

flags.DEFINE_string('path', '/tmp/dataset', 'Path to dataset.')
flags.DEFINE_string('task_name', 'humanoid_run', 'Game.')
flags.DEFINE_enum('task_class', 'control_suite',
                  ['humanoid', 'rodent', 'control_suite'],
                  'Task classes.')

FLAGS = flags.FLAGS


def main(_):

  if FLAGS.task_class == 'control_suite':
    task = dm_control_suite.ControlSuite(task_name=FLAGS.task_name)
  elif FLAGS.task_class == 'humanoid':
    task = dm_control_suite.CmuThirdParty(task_name=FLAGS.task_name)
  elif FLAGS.task_class == 'rodent':
    task = dm_control_suite.Rodent(task_name=FLAGS.task_name)

  ds = dm_control_suite.dataset(root_path=FLAGS.path,
                                data_path=task.data_path,
                                shapes=task.shapes,
                                num_threads=1,
                                batch_size=2,
                                uint8_features=task.uint8_features,
                                num_shards=1,
                                shuffle_buffer_size=10)

  for sample in ds.take(1):
    print('Data spec')
    print(tree.map_structure(lambda x: (x.dtype, x.shape), sample.data))

  environment = task.environment
  timestep = environment.reset()
  print(tree.map_structure(lambda x: (x.dtype, x.shape), timestep.observation))


if __name__ == '__main__':
  app.run(main)
