# Copyright 2019 DeepMind Technologies Limited and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sonnet as snt
import tensorflow.compat.v1 as tf

from cs_gan import gan


class DummyGenerator(snt.AbstractModule):

  def __init__(self):
    super(DummyGenerator, self).__init__(name='dummy_generator')

  def _build(self, inputs, is_training):
    return snt.Linear(10)(inputs)


class GanTest(tf.test.TestCase):

  def testConnect(self):
    discriminator = snt.Linear(2)
    generator = DummyGenerator()
    model = gan.GAN(
        discriminator, generator,
        num_z_iters=0, z_step_size=0.1,
        z_project_method='none', optimisation_cost_weight=0.0)

    generator_inputs = tf.ones((16, 3), dtype=tf.float32)
    data = tf.ones((16, 10))
    opt_compoments, _ = model.connect(data, generator_inputs)

    self.assertIn('disc', opt_compoments)
    self.assertIn('gen', opt_compoments)

    self.assertCountEqual(
        opt_compoments['disc'].vars,
        discriminator.get_variables())
    self.assertCountEqual(
        opt_compoments['gen'].vars,
        generator.get_variables() + model._log_step_size_module.get_variables())


if __name__ == '__main__':
  tf.test.main()
