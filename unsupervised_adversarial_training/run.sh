#!/bin/sh
# Copyright 2019 Deepmind Technologies Limited.
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

# Usage:
# user@host:/path/to/deepmind_research$ unsupervised_adversarial_training/run.sh

# Sets up virtual environment, install dependencies, and runs evaluation script
python3 -m venv /tmp/uat_venv
source /tmp/uat_venv/bin/activate
pip install -U pip
pip install -r unsupervised_adversarial_training/requirements.txt

python -m unsupervised_adversarial_training.quick_eval_cifar \
  --attack_fn_name=fgsm \
  --num_batches=1
