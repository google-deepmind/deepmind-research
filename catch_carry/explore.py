# Copyright 2020 Deepmind Technologies Limited.
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

"""Simple script to launch viewer with an example environment."""

from absl import app
from absl import flags
from dm_control import viewer
from catch_carry import task_examples

FLAGS = flags.FLAGS
flags.DEFINE_enum('task', 'warehouse', ['warehouse', 'toss'],
                  'The task to visualize.')

TASKS = {
    'warehouse': task_examples.build_vision_warehouse,
    'toss': task_examples.build_vision_toss,
}


def main(unused_argv):
  viewer.launch(environment_loader=TASKS[FLAGS.task])

if __name__ == '__main__':
  app.run(main)

