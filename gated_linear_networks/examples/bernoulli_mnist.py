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

MAX_TRAIN_STEPS = flags.DEFINE_integer(
    name='max_train_steps',
    default=None,
    help='Maximum number of training steps to perform (None=no limit)',
)

# Small example network, achieves ~95% test set accuracy =======================
# Network parameters.
NUM_LAYERS = flags.DEFINE_integer(
    name='num_layers',
    default=2,
    help='Number of network layers',
)
NEURONS_PER_LAYER = flags.DEFINE_integer(
    name='neurons_per_layer',
    default=100,
    help='Number of neurons per layer',
)
CONTEXT_DIM = flags.DEFINE_integer(
    name='context_dim',
    default=1,
    help='Context vector size',
)

# Learning rate schedule.
MAX_LR = flags.DEFINE_float(
    name='max_lr',
    default=0.003,
    help='Maximum learning rate',
)
LR_CONSTANT = flags.DEFINE_float(
    name='lr_constant',
    default=1.0,
    help='Learning rate constant parameter',
)
LR_DECAY = flags.DEFINE_float(
    name='lr_decay',
    default=0.1,
    help='Learning rate decay parameter',
)

# Logging parameters.
EVALUATE_EVERY = flags.DEFINE_integer(
    name='evaluate_every',
    default=1000,
    help='Number of training steps per evaluation epoch',
)


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
      output_sizes = [NEURONS_PER_LAYER.value] * NUM_LAYERS.value + [1]
      return bernoulli.GatedLinearNetwork(
          output_sizes=output_sizes, context_dim=CONTEXT_DIM.value)

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
        MAX_LR.value, LR_CONSTANT.value / (1. + LR_DECAY.value * step))

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
    if not step % EVALUATE_EVERY.value:
      batch_accuracy = jax.vmap(accuracy, in_axes=(None, None, 0, 0))
      accuracies = batch_accuracy(params, state, test_images, test_labels)
      total_accuracy = float(jnp.mean(accuracies))

      # Report statistics.
      print({
          'step': step,
          'accuracy': float(total_accuracy),
      })

    if MAX_TRAIN_STEPS.value is not None and step >= MAX_TRAIN_STEPS.value:
      return


if __name__ == '__main__':
  app.run(main)
