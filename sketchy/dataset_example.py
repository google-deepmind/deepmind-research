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

"""Example of loading sketchy data in tensorflow."""

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf

from sketchy import sketchy

flags.DEFINE_boolean('show_images', False, 'Enable to show example images.')
FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()

  # The example file contains only a few timesteps from a single episode.
  dataset = sketchy.load_frames('sketchy/example_data.tfrecords')
  dataset = dataset.prefetch(5)

  for example in dataset:
    print('---')
    for name, value in sorted(example.items()):
      print(name, value.dtype, value.shape)

    if FLAGS.show_images:
      plt.imshow(example['pixels/basket_front_left'])
      plt.show()


if __name__ == '__main__':
  app.run(main)
