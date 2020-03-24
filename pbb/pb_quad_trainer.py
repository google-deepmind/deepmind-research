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
"""PAC Bayesian bound for training neural network.

Script for running PAC Bayes (PB) quadratic bound on MNIST.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

from pbb import pb_quad

FLAGS = flags.FLAGS
flags.DEFINE_integer("train_batch_size", 256, "Train batch size.")
flags.DEFINE_integer("test_batch_size", 10000, "Test batch size.")
flags.DEFINE_integer("train_data_size", 50000,
                     "Number of samples in training dataset.")
flags.DEFINE_float("prior_rho", 3e-2, "Prior distribution rho param.")
flags.DEFINE_float("learning_rate", 5e-3, "Learning rate.")
flags.DEFINE_float("momentum", 0.95, "Momentum.")
flags.DEFINE_integer("num_training_steps", 10001,
                     "Number of training iterations.")
flags.DEFINE_float("loss_p_min", 1e-4, "Lower bound for cross entropy loss.")
flags.DEFINE_integer("report_interval", 100,
                     "Test data stats reporting frequency.")
flags.DEFINE_float("delta", 0.05, "PAC bound confidence.")
flags.DEFINE_string("prediction_mode", "mean",
                    "Use mean weights to predict on test data.")

_LAYER_SPEC = (("mlp", (600, 600, 600, 10,), False),)


def main(_):
  """Trains PB quadratic bound with Gaussian prior distribution."""
  with tf.Graph().as_default():
    trainer = pb_quad.PBQuad(
        _LAYER_SPEC, FLAGS.delta, FLAGS.learning_rate,
        FLAGS.momentum, FLAGS.train_batch_size,
        FLAGS.test_batch_size, FLAGS.train_data_size,
        FLAGS.prior_rho, FLAGS.loss_p_min, FLAGS.prediction_mode)
    trainer.build_train_ops()
    update_ops = trainer.update_ops

    test_stats = trainer.eval()

    # Build ops for initializing prior to posterior.
    reinit_ops = pb_quad.reinit_prior_to_posterior(trainer.posterior_prior_map)

    with tf.train.MonitoredTrainingSession() as sess:
      sess.run(reinit_ops)
      for step in range(FLAGS.num_training_steps):
        update_ops_ = sess.run(update_ops)

        train_loss_, train_acc_ = update_ops_["train_loss"], update_ops_[
            "train_acc"]
        if step % FLAGS.report_interval == 0:
          test_loss_, test_acc_ = sess.run(
              [test_stats["loss"], test_stats["acc"]])

          kl_div_n_ = update_ops_["kl_div_n"]
          pac_ub_ = update_ops_["pac_ub"]
          total_loss_ = update_ops_["total_loss"]

          tf.logging.info(
              "Step: %d , Avg Train Loss: %f, Avg Train Acc: %f, Test loss: %f,"
              "Test Acc: %f, kl_div_n: %f, pac_ub: %f, total_loss: %f", step,
              train_loss_, train_acc_, test_loss_, test_acc_, kl_div_n_,
              pac_ub_, total_loss_)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
