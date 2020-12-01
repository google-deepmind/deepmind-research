# Lint as: python3
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
"""Online MNIST classification example with Bernoulli GLN."""

from absl import app
from absl import flags

import haiku as hk
import jax
import jax.numpy as jnp
import rlax

from gated_linear_networks import bernoulli
from gated_linear_networks.examples import utils

FLAGS = flags.FLAGS

# Small example network, achieves ~95% test set accuracy =======================
# Network parameters.
flags.DEFINE_integer('num_layers', 2, '')
flags.DEFINE_integer('neurons_per_layer', 100, '')
flags.DEFINE_integer('context_dim', 1, '')

# Learning rate schedule.
flags.DEFINE_float('max_lr', 0.003, '')
flags.DEFINE_float('lr_constant', 1.0, '')
flags.DEFINE_float('lr_decay', 0.1, '')

# Logging parameters.
flags.DEFINE_integer('evaluate_every', 1000, '')


def main(unused_argv):
  # Load MNIST dataset =========================================================
  mnist_data, info = utils.load_deskewed_mnist(
      name='mnist', batch_size=-1, with_info=True)
  num_classes = info.features['label'].num_classes

  (train_images, train_labels) = (mnist_data['train']['image'],
                                  mnist_data['train']['label'])

  (test_images, test_labels) = (mnist_data['test']['image'],
                                mnist_data['test']['label'])

  # Build a (binary) GLN classifier ============================================
  def network_factory():

    def gln_factory():
      output_sizes = [FLAGS.neurons_per_layer] * FLAGS.num_layers + [1]
      return bernoulli.GatedLinearNetwork(
          output_sizes=output_sizes, context_dim=FLAGS.context_dim)

    return bernoulli.LastNeuronAggregator(gln_factory)

  def extract_features(image):
    mean, stddev = utils.MeanStdEstimator()(image)
    standardized_img = (image - mean) / (stddev + 1.)
    inputs = rlax.sigmoid(standardized_img)
    side_info = standardized_img
    return inputs, side_info

  def inference_fn(image, *args, **kwargs):
    inputs, side_info = extract_features(image)
    return network_factory().inference(inputs, side_info, *args, **kwargs)

  def update_fn(image, *args, **kwargs):
    inputs, side_info = extract_features(image)
    return network_factory().update(inputs, side_info, *args, **kwargs)

  init_, inference_ = hk.without_apply_rng(
      hk.transform_with_state(inference_fn))
  _, update_ = hk.without_apply_rng(hk.transform_with_state(update_fn))

  # Map along class dimension to create a one-vs-all classifier ================
  @jax.jit
  def init(dummy_image, key):
    """One-vs-all classifier init fn."""
    dummy_images = jnp.stack([dummy_image] * num_classes, axis=0)
    keys = jax.random.split(key, num_classes)
    return jax.vmap(init_, in_axes=(0, 0))(keys, dummy_images)

  @jax.jit
  def accuracy(params, state, image, label):
    """One-vs-all classifier inference fn."""
    fn = jax.vmap(inference_, in_axes=(0, 0, None))
    predictions, unused_state = fn(params, state, image)
    return (jnp.argmax(predictions) == label).astype(jnp.float32)

  @jax.jit
  def update(params, state, step, image, label):
    """One-vs-all classifier update fn."""

    # Learning rate schedules.
    learning_rate = jnp.minimum(
        FLAGS.max_lr, FLAGS.lr_constant / (1. + FLAGS.lr_decay * step))

    # Update weights and report log-loss.
    targets = hk.one_hot(jnp.asarray(label), num_classes)

    fn = jax.vmap(update_, in_axes=(0, 0, None, 0, None))
    out = fn(params, state, image, targets, learning_rate)
    (params, unused_predictions, log_loss), state = out
    return (jnp.mean(log_loss), params), state

  # Train on train split =======================================================
  dummy_image = train_images[0]
  params, state = init(dummy_image, jax.random.PRNGKey(42))

  for step, (image, label) in enumerate(zip(train_images, train_labels), 1):
    (unused_loss, params), state = update(
        params,
        state,
        step,
        image,
        label,
    )

    # Evaluate on test split ===================================================
    if not step % FLAGS.evaluate_every:
      batch_accuracy = jax.vmap(accuracy, in_axes=(None, None, 0, 0))
      accuracies = batch_accuracy(params, state, test_images, test_labels)
      total_accuracy = float(jnp.mean(accuracies))

      # Report statistics.
      print({
          'step': step,
          'accuracy': float(total_accuracy),
      })


if __name__ == '__main__':
  app.run(main)
