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
from box_arrangement import task_examples

FLAGS = flags.FLAGS
flags.DEFINE_enum('task', 'go_to_target', [
    'go_to_target', 'move_box', 'move_box_or_go_to_target',
    'move_box_and_go_to_target'
], 'The task to visualize.')


TASKS = {
    'go_to_target': task_examples.go_to_k_targets,
    'move_box': task_examples.move_box,
    'move_box_or_go_to_target': task_examples.move_box_or_gtt,
    'move_box_and_go_to_target': task_examples.move_box_and_gtt,
}


def main(unused_argv):
  viewer.launch(environment_loader=TASKS[FLAGS.task])

if __name__ == '__main__':
  app.run(main)
