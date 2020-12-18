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

"""Evaluates a PyTorch checkpoint on CIFAR-10/100 or MNIST."""

from absl import app
from absl import flags
import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import tqdm

from adversarial_robustness.pytorch import model_zoo

_CKPT = flags.DEFINE_string(
    'ckpt', None, 'Path to checkpoint.')
_DATASET = flags.DEFINE_enum(
    'dataset', 'cifar10', ['cifar10', 'cifar100', 'mnist'],
    'Dataset on which the checkpoint is evaluated.')
_WIDTH = flags.DEFINE_integer(
    'width', 16, 'Width of WideResNet.')
_DEPTH = flags.DEFINE_integer(
    'depth', 70, 'Depth of WideResNet.')
_USE_CUDA = flags.DEFINE_boolean(
    'use_cuda', True, 'Whether to use CUDA.')
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 100, 'Batch size.')
_NUM_BATCHES = flags.DEFINE_integer(
    'num_batches', 0,
    'Number of batches to evaluate (zero means the whole dataset).')


def main(unused_argv):
  print(f'Loading "{_CKPT.value}"')
  print(f'Using a WideResNet with depth {_DEPTH.value} and width '
        f'{_WIDTH.value}.')

  # Create model and dataset.
  if _DATASET.value == 'mnist':
    model = model_zoo.WideResNet(
        num_classes=10, depth=_DEPTH.value, width=_WIDTH.value,
        activation_fn=model_zoo.Swish, mean=.5, std=.5, padding=2,
        num_input_channels=1)
    dataset_fn = datasets.MNIST
  elif _DATASET.value == 'cifar10':
    model = model_zoo.WideResNet(
        num_classes=10, depth=_DEPTH.value, width=_WIDTH.value,
        activation_fn=model_zoo.Swish, mean=model_zoo.CIFAR10_MEAN,
        std=model_zoo.CIFAR10_STD)
    dataset_fn = datasets.CIFAR10
  else:
    assert _DATASET.value == 'cifar100'
    model = model_zoo.WideResNet(
        num_classes=100, depth=_DEPTH.value, width=_WIDTH.value,
        activation_fn=model_zoo.Swish, mean=model_zoo.CIFAR100_MEAN,
        std=model_zoo.CIFAR100_STD)
    dataset_fn = datasets.CIFAR100

  # Load model.
  if _CKPT.value != 'dummy':
    params = torch.load(_CKPT.value)
    model.load_state_dict(params)
  if _USE_CUDA.value:
    model.cuda()
  model.eval()
  print('Successfully loaded.')

  # Load dataset.
  transform_chain = transforms.Compose([transforms.ToTensor()])
  ds = dataset_fn(root='/tmp/data', train=False, transform=transform_chain,
                  download=True)
  test_loader = data.DataLoader(ds, batch_size=_BATCH_SIZE.value, shuffle=False,
                                num_workers=0)

  # Evaluation.
  correct = 0
  total = 0
  batch_count = 0
  total_batches = min((10_000 - 1) // _BATCH_SIZE.value + 1, _NUM_BATCHES.value)
  with torch.no_grad():
    for images, labels in tqdm.tqdm(test_loader, total=total_batches):
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      batch_count += 1
      if _NUM_BATCHES.value > 0 and batch_count >= _NUM_BATCHES.value:
        break
  print(f'Accuracy on the {total} test images: {100 * correct / total:.2f}%')


if __name__ == '__main__':
  flags.mark_flag_as_required('ckpt')
  app.run(main)
