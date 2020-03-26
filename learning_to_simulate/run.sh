#!/bin/bash
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

# Fail on any error.
set -e

# Display commands being run.
set -x

virtualenv --python=python3 /tmp/learning_to_simulate
source /tmp/learning_to_simulate/bin/activate

# Install dependencies.
pip install --upgrade -r learning_to_simulate/requirements.txt

python -m learning_to_simulate.model_demo

rm -r /tmp/learning_to_simulate
