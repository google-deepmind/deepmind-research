# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluates a JAX checkpoint on CIFAR-10/100 or MNIST."""

import functools

from absl import app
from absl import flags
import haiku as hk
import numpy as np
import optax
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tqdm

from adversarial_robustness.jax import attacks
from adversarial_robustness.jax import datasets
from adversarial_robustness.jax import model_zoo

_CKPT = flags.DEFINE_string(
    'ckpt', None, 'Path to checkpoint.')
_DATASET = flags.DEFINE_enum(
    'dataset', 'cifar10', ['cifar10', 'cifar100', 'mnist'],
    'Dataset on which the checkpoint is evaluated.')
_WIDTH = flags.DEFINE_integer(
    'width', 16, 'Width of WideResNet.')
_DEPTH = flags.DEFINE_integer(
    'depth', 70, 'Depth of WideResNet.')
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 100, 'Batch size.')
_NUM_BATCHES = flags.DEFINE_integer(
    'num_batches', 0,
    'Number of batches to evaluate (zero means the whole dataset).')


def main(unused_argv):
  print(f'Loading "{_CKPT.value}"')
  print(f'Using a WideResNet with depth {_DEPTH.value} and width '
        f'{_WIDTH.value}.')

  # Create dataset.
  if _DATASET.value == 'mnist':
    _, data_test = tf.keras.datasets.mnist.load_data()
    normalize_fn = datasets.mnist_normalize
  elif _DATASET.value == 'cifar10':
    _, data_test = tf.keras.datasets.cifar10.load_data()
    normalize_fn = datasets.cifar10_normalize
  else:
    assert _DATASET.value == 'cifar100'
    _, data_test = tf.keras.datasets.cifar100.load_data()
    normalize_fn = datasets.cifar100_normalize

  # Create model.
  @hk.transform_with_state
  def model_fn(x, is_training=False):
    model = model_zoo.WideResNet(
        num_classes=10, depth=_DEPTH.value, width=_WIDTH.value,
        activation='swish')
    return model(normalize_fn(x), is_training=is_training)

  # Build dataset.
  images, labels = data_test
  samples = (images.astype(np.float32) / 255.,
             np.squeeze(labels, axis=-1).astype(np.int64))
  data = tf.data.Dataset.from_tensor_slices(samples).batch(_BATCH_SIZE.value)
  test_loader = tfds.as_numpy(data)

  # Load model parameters.
  rng_seq = hk.PRNGSequence(0)
  if _CKPT.value == 'dummy':
    for images, _ in test_loader:
      break
    params, state = model_fn.init(next(rng_seq), images, is_training=True)
    # Reset iterator.
    test_loader = tfds.as_numpy(data)
  else:
    params, state = np.load(_CKPT.value, allow_pickle=True)

  # Create adversarial attack. We run a PGD-40 attack with margin loss.
  epsilon = 8 / 255
  eval_attack = attacks.UntargetedAttack(
      attacks.PGD(
          attacks.Adam(learning_rate_fn=optax.piecewise_constant_schedule(
              init_value=.1,
              boundaries_and_scales={20: .1, 30: .01})),
          num_steps=40,
          initialize_fn=attacks.linf_initialize_fn(epsilon),
          project_fn=attacks.linf_project_fn(epsilon, bounds=(0., 1.))),
      loss_fn=attacks.untargeted_margin)

  def logits_fn(x, rng):
    return model_fn.apply(params, state, rng, x)[0]

  # Evaluation.
  correct = 0
  adv_correct = 0
  total = 0
  batch_count = 0
  total_batches = min((10_000 - 1) // _BATCH_SIZE.value + 1, _NUM_BATCHES.value)
  for images, labels in tqdm.tqdm(test_loader, total=total_batches):
    rng = next(rng_seq)
    loop_logits_fn = functools.partial(logits_fn, rng=rng)

    # Clean examples.
    outputs = loop_logits_fn(images)
    correct += (np.argmax(outputs, 1) == labels).sum().item()

    # Adversarial examples.
    adv_images = eval_attack(loop_logits_fn, next(rng_seq), images, labels)
    outputs = loop_logits_fn(adv_images)
    predicted = np.argmax(outputs, 1)
    adv_correct += (predicted == labels).sum().item()

    total += labels.shape[0]
    batch_count += 1
    if _NUM_BATCHES.value > 0 and batch_count >= _NUM_BATCHES.value:
      break
  print(f'Accuracy on the {total} test images: {100 * correct / total:.2f}%')
  print(f'Robust accuracy: {100 * adv_correct / total:.2f}%')


if __name__ == '__main__':
  flags.mark_flag_as_required('ckpt')
  try:
    tf.config.set_visible_devices([], 'GPU')  # Prevent TF from using the GPU.
  except tf.errors.NotFoundError:
    pass
  app.run(main)
