#!/bin/sh
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
set -e

python3 -m venv /tmp/gln_venv
source /tmp/gln_venv/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -r gated_linear_networks/requirements.txt

# Run MNIST example with Bernoulli GLN
python3 -m gated_linear_networks.examples.bernoulli_mnist \
  --num_layers=2 \
  --neurons_per_layer=100 \
  --context_dim=1 \
  --max_train_steps=2000
