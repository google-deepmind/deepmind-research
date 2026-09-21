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
r"""Atari dataset example.

Instructions:
> mkdir -p /tmp/dataset/Asterix
> gsutil cp gs://rl_unplugged/atari/Asterix/run_1-00000-of-00100 \
    /tmp/dataset/Asterix/run_1-00000-of-00001
> python atari_example.py --path=/tmp/dataset --game=Asterix
"""

from absl import app
from absl import flags
from acme import specs
import tree

from rl_unplugged import atari

flags.DEFINE_string('path', '/tmp/dataset', 'Path to dataset.')
flags.DEFINE_string('game', 'Asterix', 'Game.')

FLAGS = flags.FLAGS


def main(_):
  ds = atari.dataset(FLAGS.path, FLAGS.game, 1,
                     num_shards=1,
                     shuffle_buffer_size=1)
  for sample in ds.take(1):
    print('Data spec')
    print(tree.map_structure(lambda x: (x.dtype, x.shape), sample.data))

  env = atari.environment(FLAGS.game)
  print('Environment spec')
  print(specs.make_environment_spec(env))
  print('Environment observation')
  timestep = env.reset()
  print(tree.map_structure(lambda x: (x.dtype, x.shape), timestep.observation))


if __name__ == '__main__':
  app.run(main)
