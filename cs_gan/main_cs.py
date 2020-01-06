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
"""Training script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from cs_gan import cs
from cs_gan import file_utils
from cs_gan import utils

tfd = tfp.distributions

flags.DEFINE_string(
    'mode', 'recons', 'Model mode.')
flags.DEFINE_integer(
    'num_training_iterations', 10000000,
    'Number of training iterations.')
flags.DEFINE_integer(
    'batch_size', 64, 'Training batch size.')
flags.DEFINE_integer(
    'num_measurements', 25, 'The number of measurements')
flags.DEFINE_integer(
    'num_latents', 100, 'The number of latents')
flags.DEFINE_integer(
    'num_z_iters', 3, 'The number of latent optimisation steps.')
flags.DEFINE_float(
    'z_step_size', 0.01, 'Step size for latent optimisation.')
flags.DEFINE_string(
    'z_project_method', 'norm', 'The method to project z.')
flags.DEFINE_integer(
    'summary_every_step', 1000,
    'The interval at which to log debug ops.')
flags.DEFINE_integer(
    'export_every', 10,
    'The interval at which to export samples.')
flags.DEFINE_string(
    'dataset', 'mnist', 'The dataset used for learning (cifar|mnist.')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_string(
    'output_dir', '/tmp/cs_gan/cs', 'Location where to save output files.')


FLAGS = flags.FLAGS

# Log info level (for Hooks).
tf.logging.set_verbosity(tf.logging.INFO)


def main(argv):
  del argv

  utils.make_output_dir(FLAGS.output_dir)
  data_processor = utils.DataProcessor()
  images = utils.get_train_dataset(data_processor, FLAGS.dataset,
                                   FLAGS.batch_size)

  logging.info('Learning rate: %d', FLAGS.learning_rate)

  # Construct optimizers.
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

  # Create the networks and models.
  generator = utils.get_generator(FLAGS.dataset)
  metric_net = utils.get_metric_net(FLAGS.dataset, FLAGS.num_measurements)
  model = cs.CS(metric_net, generator,
                FLAGS.num_z_iters, FLAGS.z_step_size, FLAGS.z_project_method)
  prior = utils.make_prior(FLAGS.num_latents)
  generator_inputs = prior.sample(FLAGS.batch_size)

  model_output = model.connect(images, generator_inputs)
  optimization_components = model_output.optimization_components
  debug_ops = model_output.debug_ops
  reconstructions, _ = utils.optimise_and_sample(
      generator_inputs, model, images, is_training=False)

  global_step = tf.train.get_or_create_global_step()
  update_op = optimizer.minimize(
      optimization_components.loss,
      var_list=optimization_components.vars,
      global_step=global_step)

  sample_exporter = file_utils.FileExporter(
      os.path.join(FLAGS.output_dir, 'reconstructions'))

  # Hooks.
  debug_ops['it'] = global_step
  # Abort training on Nans.
  nan_hook = tf.train.NanTensorHook(optimization_components.loss)
  # Step counter.
  step_conter_hook = tf.train.StepCounterHook()

  checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      checkpoint_dir=utils.get_ckpt_dir(FLAGS.output_dir), save_secs=10 * 60)

  loss_summary_saver_hook = tf.train.SummarySaverHook(
      save_steps=FLAGS.summary_every_step,
      output_dir=os.path.join(FLAGS.output_dir, 'summaries'),
      summary_op=utils.get_summaries(debug_ops))

  hooks = [checkpoint_saver_hook, nan_hook, step_conter_hook,
           loss_summary_saver_hook]

  # Start training.
  with tf.train.MonitoredSession(hooks=hooks) as sess:
    logging.info('starting training')

    for i in range(FLAGS.num_training_iterations):
      sess.run(update_op)

      if i % FLAGS.export_every == 0:
        reconstructions_np, data_np = sess.run([reconstructions, images])
        # Create an object which gets data and does the processing.
        data_np = data_processor.postprocess(data_np)
        reconstructions_np = data_processor.postprocess(reconstructions_np)
        sample_exporter.save(reconstructions_np, 'reconstructions')
        sample_exporter.save(data_np, 'data')


if __name__ == '__main__':
  app.run(main)
