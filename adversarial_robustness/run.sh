#!/bin/sh
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

python3 -m venv adversarial_robustness_venv
source adversarial_robustness_venv/bin/activate
pip install -r adversarial_robustness/requirements.txt

python3 -m adversarial_robustness.jax.eval \
  --ckpt=dummy \
  --arch=wrn-10-1 \
  --dataset=cifar10 \
  --batch_size=1 \
  --num_batches=1

python3 -m adversarial_robustness.pytorch.eval \
  --ckpt=dummy \
  --arch=wrn-10-1 \
  --dataset=cifar10 \
  --batch_size=1 \
  --num_batches=1 \
  --nouse_cuda
