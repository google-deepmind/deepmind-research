# Copyright 2020 DeepMind Technologies Limited
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
"""Tests for pb_quad_trainer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow_datasets import testing
from pbb import pb_quad


class PBQuadTrainerTest(tf.test.TestCase):

  def testTrain(self):
    with tf.Graph().as_default():
      with testing.mock_data():
        layer_spec = (("mlp", (600, 600, 600, 10,), False),)
        trainer = pb_quad.PBQuad(layer_spec=layer_spec)
        trainer.build_train_ops()
        update_ops = trainer.update_ops
        reinit_ops = pb_quad.reinit_prior_to_posterior(
            trainer.posterior_prior_map)

        with tf.train.MonitoredTrainingSession() as sess:
          sess.run(reinit_ops)
          for _ in range(1000):
            update_ops_ = sess.run(update_ops)

            train_loss_, train_acc_ = update_ops_["train_loss"], update_ops_[
                "train_acc"]
          self.assertLessEqual(train_loss_, 1.25)
          self.assertGreaterEqual(train_acc_, 0.75)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.test.main()
