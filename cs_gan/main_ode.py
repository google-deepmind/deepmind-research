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
    'num_training_iterations', 1200000,
    'Number of training iterations.')
flags.DEFINE_string(
    'ode_mode', 'rk4', 'Integration method.')
flags.DEFINE_integer(
    'batch_size', 64, 'Training batch size.')
flags.DEFINE_float(
    'grad_reg_weight', 0.02, 'Step size for latent optimisation.')
flags.DEFINE_string(
    'opt_name', 'gd', 'Name of the optimiser (gd|adam).')
flags.DEFINE_bool(
    'schedule_lr', True, 'The method to project z.')
flags.DEFINE_bool(
    'reg_first_grad_only', True, 'Whether only to regularise the first grad.')
flags.DEFINE_integer(
    'num_latents', 128, 'The number of latents')
flags.DEFINE_integer(
    'summary_every_step', 1000,
    'The interval at which to log debug ops.')
flags.DEFINE_integer(
    'image_metrics_every_step', 1000,
    'The interval at which to log (expensive) image metrics.')
flags.DEFINE_integer(
    'export_every', 10,
    'The interval at which to export samples.')
# Use 50k to reproduce scores from the paper. Default to 10k here to avoid the
# runtime error caused by too large graph with 50k samples on some machines.
flags.DEFINE_integer(
    'num_eval_samples', 10000,
    'The number of samples used to evaluate FID/IS.')
flags.DEFINE_string(
    'dataset', 'cifar', 'The dataset used for learning (cifar|mnist).')
flags.DEFINE_string(
    'output_dir', '/tmp/ode_gan/gan', 'Location where to save output files.')
flags.DEFINE_float('disc_lr', 4e-2, 'Discriminator Learning rate.')
flags.DEFINE_float('gen_lr', 4e-2, 'Generator Learning rate.')
flags.DEFINE_bool(
    'run_real_data_metrics', False,
    'Whether or not to run image metrics on real data.')
flags.DEFINE_bool(
    'run_sample_metrics', True,
    'Whether or not to run image metrics on samples.')


FLAGS = flags.FLAGS

# Log info level (for Hooks).
tf.logging.set_verbosity(tf.logging.INFO)


def _copy_vars(v_list):
  """Copy variables in v_list."""
  t_list = []
  for v in v_list:
    t_list.append(tf.identity(v))
  return t_list


def _restore_vars(v_list, t_list):
  """Restore variables in v_list from t_list."""
  ops = []
  for v, t in zip(v_list, t_list):
    ops.append(v.assign(t))
  return ops


def _scale_vars(s, v_list):
  """Scale all variables in v_list by s."""
  return [s * v for v in v_list]


def _acc_grads(g_sum, g_w, g):
  """Accumulate gradients in g, weighted by g_w."""
  return [g_sum_i + g_w * g_i for g_sum_i, g_i in zip(g_sum, g)]


def _compute_reg_grads(gen_grads, disc_vars):
  """Compute gradients norm (this is an upper-bpund of the full-batch norm)."""
  gen_norm = tf.accumulate_n([tf.reduce_sum(u * u) for u in gen_grads])
  disc_reg_grads = tf.gradients(gen_norm, disc_vars)
  return disc_reg_grads


def run_model(prior, images, model, disc_reg_weight):
  """Run the model with new data and samples.

  Args:
    prior: the noise source as the generator input.
    images: images sampled from dataset.
    model: a GAN model defined in gan.py.
    disc_reg_weight: regularisation weight for discrmininator gradients.

  Returns:
    debug_ops: statistics from the model, see gan.py for more detials.
    disc_grads: discriminator gradients.
    gen_grads: generator gradients.
  """
  generator_inputs = prior.sample(FLAGS.batch_size)
  model_output = model.connect(images, generator_inputs)
  optimization_components = model_output.optimization_components

  disc_grads = tf.gradients(
      optimization_components['disc'].loss,
      optimization_components['disc'].vars)

  gen_grads = tf.gradients(
      optimization_components['gen'].loss,
      optimization_components['gen'].vars)

  if disc_reg_weight > 0.0:
    reg_grads = _compute_reg_grads(gen_grads,
                                   optimization_components['disc'].vars)
    disc_grads = _acc_grads(disc_grads, disc_reg_weight, reg_grads)

  debug_ops = model_output.debug_ops

  return debug_ops, disc_grads, gen_grads


def update_model(model, disc_grads, gen_grads, disc_opt, gen_opt,
                 global_step, update_scale):
  """Update model with gradients."""

  disc_vars, gen_vars = model.get_variables()

  with tf.control_dependencies(gen_grads + disc_grads):
    disc_update_op = disc_opt.apply_gradients(
        zip(_scale_vars(update_scale, disc_grads),
            disc_vars))

    gen_update_op = gen_opt.apply_gradients(
        zip(_scale_vars(update_scale, gen_grads),
            gen_vars),
        global_step=global_step)

    update_op = tf.group([disc_update_op, gen_update_op])

  return update_op


def main(argv):
  del argv

  utils.make_output_dir(FLAGS.output_dir)
  data_processor = utils.DataProcessor()
  # Compute the batch-size multiplier
  if FLAGS.ode_mode == 'rk2':
    batch_mul = 2
  elif FLAGS.ode_mode == 'rk4':
    batch_mul = 4
  else:
    batch_mul = 1
  images = utils.get_train_dataset(data_processor, FLAGS.dataset,
                                   int(FLAGS.batch_size * batch_mul))
  image_splits = tf.split(images, batch_mul)

  logging.info('Generator learning rate: %d', FLAGS.gen_lr)
  logging.info('Discriminator learning rate: %d', FLAGS.disc_lr)

  global_step = tf.train.get_or_create_global_step()
  # Construct optimizers.
  if FLAGS.opt_name == 'adam':
    disc_opt = tf.train.AdamOptimizer(FLAGS.disc_lr, beta1=0.5, beta2=0.999)
    gen_opt = tf.train.AdamOptimizer(FLAGS.gen_lr, beta1=0.5, beta2=0.999)
  elif FLAGS.opt_name == 'gd':
    if FLAGS.schedule_lr:
      gd_disc_lr = tf.train.piecewise_constant(
          global_step,
          values=[FLAGS.disc_lr / 4., FLAGS.disc_lr, FLAGS.disc_lr / 2.],
          boundaries=[500, 400000])
      gd_gen_lr = tf.train.piecewise_constant(
          global_step,
          values=[FLAGS.gen_lr / 4., FLAGS.gen_lr, FLAGS.gen_lr / 2.],
          boundaries=[500, 400000])
    else:
      gd_disc_lr = FLAGS.disc_lr
      gd_gen_lr = FLAGS.gen_lr
    disc_opt = tf.train.GradientDescentOptimizer(gd_disc_lr)
    gen_opt = tf.train.GradientDescentOptimizer(gd_gen_lr)
  else:
    raise ValueError('Unknown ODE mode!')

  # Create the networks and models.
  generator = utils.get_generator(FLAGS.dataset)
  metric_net = utils.get_metric_net(FLAGS.dataset, use_sn=False)
  model = gan.GAN(metric_net, generator)
  prior = utils.make_prior(FLAGS.num_latents)

  # Setup ODE parameters.
  if FLAGS.ode_mode == 'rk2':
    ode_grad_weights = [0.5, 0.5]
    step_scale = [1.0]
  elif FLAGS.ode_mode == 'rk4':
    ode_grad_weights = [1. / 6., 1. / 3., 1. / 3., 1. / 6.]
    step_scale = [0.5, 0.5, 1.]
  elif FLAGS.ode_mode == 'euler':
    # Euler update
    ode_grad_weights = [1.0]
    step_scale = []
  else:
    raise ValueError('Unknown ODE mode!')

  # Extra steps for RK updates.
  num_extra_steps = len(step_scale)

  if FLAGS.reg_first_grad_only:
    first_reg_weight = FLAGS.grad_reg_weight / ode_grad_weights[0]
    other_reg_weight = 0.0
  else:
    first_reg_weight = FLAGS.grad_reg_weight
    other_reg_weight = FLAGS.grad_reg_weight

  debug_ops, disc_grads, gen_grads = run_model(prior, image_splits[0],
                                               model, first_reg_weight)

  disc_vars, gen_vars = model.get_variables()

  final_disc_grads = _scale_vars(ode_grad_weights[0], disc_grads)
  final_gen_grads = _scale_vars(ode_grad_weights[0], gen_grads)

  restore_ops = []
  # Preparing for further RK steps.
  if num_extra_steps > 0:
    # copy the variables before they are changed by update_op
    saved_disc_vars = _copy_vars(disc_vars)
    saved_gen_vars = _copy_vars(gen_vars)

    # Enter RK loop.
    with tf.control_dependencies(saved_disc_vars + saved_gen_vars):
      step_deps = []
      for i_step in range(num_extra_steps):
        with tf.control_dependencies(step_deps):
        # Compute gradient steps for intermediate updates.
          update_op = update_model(
              model, disc_grads, gen_grads, disc_opt, gen_opt,
              None, step_scale[i_step])
          with tf.control_dependencies([update_op]):
            _, disc_grads, gen_grads = run_model(
                prior, image_splits[i_step + 1], model, other_reg_weight)

            # Accumlate gradients for final update.
            final_disc_grads = _acc_grads(final_disc_grads,
                                          ode_grad_weights[i_step + 1],
                                          disc_grads)
            final_gen_grads = _acc_grads(final_gen_grads,
                                         ode_grad_weights[i_step + 1],
                                         gen_grads)

            # Make new restore_op for each step.
            restore_ops = []
            restore_ops += _restore_vars(disc_vars, saved_disc_vars)
            restore_ops += _restore_vars(gen_vars, saved_gen_vars)

            step_deps = restore_ops

  with tf.control_dependencies(restore_ops):
    update_op = update_model(
        model, final_disc_grads, final_gen_grads, disc_opt, gen_opt,
        global_step, 1.0)

  samples = generator(prior.sample(FLAGS.batch_size), is_training=False)

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
  nan_disc_hook = tf.train.NanTensorHook(debug_ops['disc_loss'])
  nan_gen_hook = tf.train.NanTensorHook(debug_ops['gen_loss'])
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
      sess.run(update_op)

      if i % FLAGS.export_every == 0:
        samples_np, data_np = sess.run([samples, image_splits[0]])
        # Create an object which gets data and does the processing.
        data_np = data_processor.postprocess(data_np)
        samples_np = data_processor.postprocess(samples_np)
        sample_exporter.save(samples_np, 'samples')
        sample_exporter.save(data_np, 'data')


if __name__ == '__main__':
  tf.enable_resource_variables()
  app.run(main)
