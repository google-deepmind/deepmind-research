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

from absl import app
from absl import flags
import haiku as hk
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tqdm

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
    normalize_fn = model_zoo.mnist_normalize
  elif _DATASET.value == 'cifar10':
    _, data_test = tf.keras.datasets.cifar10.load_data()
    normalize_fn = model_zoo.cifar10_normalize
  else:
    assert _DATASET.value == 'cifar100'
    _, data_test = tf.keras.datasets.cifar100.load_data()
    normalize_fn = model_zoo.cifar100_normalize

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

  # Evaluation.
  correct = 0
  total = 0
  batch_count = 0
  total_batches = min((10_000 - 1) // _BATCH_SIZE.value + 1, _NUM_BATCHES.value)
  for images, labels in tqdm.tqdm(test_loader, total=total_batches):
    outputs = model_fn.apply(params, state, next(rng_seq), images)[0]
    predicted = np.argmax(outputs, 1)
    total += labels.shape[0]
    correct += (predicted == labels).sum().item()
    batch_count += 1
    if _NUM_BATCHES.value > 0 and batch_count >= _NUM_BATCHES.value:
      break
  print(f'Accuracy on the {total} test images: {100 * correct / total:.2f}%')


if __name__ == '__main__':
  flags.mark_flag_as_required('ckpt')
  app.run(main)
