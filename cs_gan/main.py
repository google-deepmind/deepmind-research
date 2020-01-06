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

from cs_gan import file_utils
from cs_gan import gan
from cs_gan import image_metrics
from cs_gan import utils

flags.DEFINE_integer(
    'num_training_iterations', 200000,
    'Number of training iterations.')
flags.DEFINE_integer(
    'batch_size', 64, 'Training batch size.')
flags.DEFINE_integer(
    'num_latents', 128, 'The number of latents')
flags.DEFINE_integer(
    'summary_every_step', 1000,
    'The interval at which to log debug ops.')
flags.DEFINE_integer(
    'image_metrics_every_step', 2000,
    'The interval at which to log (expensive) image metrics.')
flags.DEFINE_integer(
    'export_every', 10,
    'The interval at which to export samples.')
flags.DEFINE_integer(
    'num_eval_samples', 10000,
    'The number of samples used to evaluate FID/IS')
flags.DEFINE_string(
    'dataset', 'cifar', 'The dataset used for learning (cifar|mnist.')
flags.DEFINE_float(
    'optimisation_cost_weight', 3., 'weight for latent optimisation cost.')
flags.DEFINE_integer(
    'num_z_iters', 3, 'The number of latent optimisation steps.'
    'It falls back to vanilla GAN when num_z_iters is set to 0.')
flags.DEFINE_float(
    'z_step_size', 0.01, 'Step size for latent optimisation.')
flags.DEFINE_string(
    'z_project_method', 'norm', 'The method to project z.')
flags.DEFINE_string(
    'output_dir', '/tmp/cs_gan/gan', 'Location where to save output files.')
flags.DEFINE_float('disc_lr', 2e-4, 'Discriminator Learning rate.')
flags.DEFINE_float('gen_lr', 2e-4, 'Generator Learning rate.')
flags.DEFINE_bool(
    'run_real_data_metrics', False,
    'Whether or not to run image metrics on real data.')
flags.DEFINE_bool(
    'run_sample_metrics', True,
    'Whether or not to run image metrics on samples.')


FLAGS = flags.FLAGS

# Log info level (for Hooks).
tf.logging.set_verbosity(tf.logging.INFO)


def main(argv):
  del argv

  utils.make_output_dir(FLAGS.output_dir)
  data_processor = utils.DataProcessor()
  images = utils.get_train_dataset(data_processor, FLAGS.dataset,
                                   FLAGS.batch_size)

  logging.info('Generator learning rate: %d', FLAGS.gen_lr)
  logging.info('Discriminator learning rate: %d', FLAGS.disc_lr)

  # Construct optimizers.
  disc_optimizer = tf.train.AdamOptimizer(FLAGS.disc_lr, beta1=0.5, beta2=0.999)
  gen_optimizer = tf.train.AdamOptimizer(FLAGS.gen_lr, beta1=0.5, beta2=0.999)

  # Create the networks and models.
  generator = utils.get_generator(FLAGS.dataset)
  metric_net = utils.get_metric_net(FLAGS.dataset)
  model = gan.GAN(metric_net, generator,
                  FLAGS.num_z_iters, FLAGS.z_step_size,
                  FLAGS.z_project_method, FLAGS.optimisation_cost_weight)
  prior = utils.make_prior(FLAGS.num_latents)
  generator_inputs = prior.sample(FLAGS.batch_size)

  model_output = model.connect(images, generator_inputs)
  optimization_components = model_output.optimization_components
  debug_ops = model_output.debug_ops
  samples = generator(generator_inputs, is_training=False)

  global_step = tf.train.get_or_create_global_step()
  # We pass the global step both to the disc and generator update ops.
  # This means that the global step will not be the same as the number of
  # iterations, but ensures that hooks which rely on global step work correctly.
  disc_update_op = disc_optimizer.minimize(
      optimization_components['disc'].loss,
      var_list=optimization_components['disc'].vars,
      global_step=global_step)

  gen_update_op = gen_optimizer.minimize(
      optimization_components['gen'].loss,
      var_list=optimization_components['gen'].vars,
      global_step=global_step)

  # Get data needed to compute FID. We also compute metrics on
  # real data as a sanity check and as a reference point.
  eval_real_data = utils.get_real_data_for_eval(FLAGS.num_eval_samples,
                                                FLAGS.dataset,
                                                split='train')

  def sample_fn(x):
    return utils.optimise_and_sample(x, module=model,
                                     data=None, is_training=False)[0]

  if FLAGS.run_sample_metrics:
    sample_metrics = image_metrics.get_image_metrics_for_samples(
        eval_real_data, sample_fn,
        prior, data_processor,
        num_eval_samples=FLAGS.num_eval_samples)
  else:
    sample_metrics = {}

  if FLAGS.run_real_data_metrics:
    data_metrics = image_metrics.get_image_metrics(
        eval_real_data, eval_real_data)
  else:
    data_metrics = {}

  sample_exporter = file_utils.FileExporter(
      os.path.join(FLAGS.output_dir, 'samples'))

  # Hooks.
  debug_ops['it'] = global_step
  # Abort training on Nans.
  nan_disc_hook = tf.train.NanTensorHook(optimization_components['disc'].loss)
  nan_gen_hook = tf.train.NanTensorHook(optimization_components['gen'].loss)
  # Step counter.
  step_conter_hook = tf.train.StepCounterHook()

  checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      checkpoint_dir=utils.get_ckpt_dir(FLAGS.output_dir), save_secs=10 * 60)

  loss_summary_saver_hook = tf.train.SummarySaverHook(
      save_steps=FLAGS.summary_every_step,
      output_dir=os.path.join(FLAGS.output_dir, 'summaries'),
      summary_op=utils.get_summaries(debug_ops))

  metrics_summary_saver_hook = tf.train.SummarySaverHook(
      save_steps=FLAGS.image_metrics_every_step,
      output_dir=os.path.join(FLAGS.output_dir, 'summaries'),
      summary_op=utils.get_summaries(sample_metrics))

  hooks = [checkpoint_saver_hook, metrics_summary_saver_hook,
           nan_disc_hook, nan_gen_hook, step_conter_hook,
           loss_summary_saver_hook]

  # Start training.
  with tf.train.MonitoredSession(hooks=hooks) as sess:
    logging.info('starting training')

    for key, value in sess.run(data_metrics).items():
      logging.info('%s: %d', key, value)

    for i in range(FLAGS.num_training_iterations):
      sess.run(disc_update_op)
      sess.run(gen_update_op)

      if i % FLAGS.export_every == 0:
        samples_np, data_np = sess.run([samples, images])
        # Create an object which gets data and does the processing.
        data_np = data_processor.postprocess(data_np)
        samples_np = data_processor.postprocess(samples_np)
        sample_exporter.save(samples_np, 'samples')
        sample_exporter.save(data_np, 'data')


if __name__ == '__main__':
  app.run(main)
