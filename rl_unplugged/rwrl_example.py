# pylint: disable=line-too-long
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
r"""RWRL dataset example.

Instructions:
> export TMP_PATH=/tmp/dataset/rwrl
> export DATA_PATH=combined_challenge_easy/quadruped/walk/offline_rl_challenge_easy
> mkdir -p $TMP_PATH/$DATA_PATH
> gsutil cp gs://rl_unplugged/rwrl/$DATA_PATH/episodes.tfrecord-00001-of-00015 \
$TMP_PATH/$DATA_PATH/episodes.tfrecord-00000-of-00001
> python rwrl_example.py --path=$TMP_PATH
"""
# pylint: enable=line-too-long

from absl import app
from absl import flags
import tree

from rl_unplugged import rwrl

flags.DEFINE_string('path', '/tmp/dataset', 'Path to dataset.')


def main(_):
  ds = rwrl.dataset(
      flags.FLAGS.path,
      combined_challenge='easy',
      domain='quadruped',
      task='walk',
      difficulty='easy',
      num_shards=1,
      shuffle_buffer_size=1)
  for replay_sample in ds.take(1):
    print(tree.map_structure(lambda x: (x.dtype, x.shape), replay_sample.data))
    break

if __name__ == '__main__':
  app.run(main)
