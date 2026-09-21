#!/bin/bash
# Copyright 2021 DeepMind Technologies Limited
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

# This script installs tandem_dqn in a clean virtualenv and runs an
# example training loop. It is designed to be run from the parent directory,
# e.g.:
#
# git clone git@github.com:deepmind/deepmind-research.git
# cd deepmind_research
# tandem_dqn/run.sh

# Fail on errors, display commands.
set -e
set -x

TMP_DIR=`mktemp -d`

# Set up environment.
virtualenv --python=python3.6 "${TMP_DIR}/env"
source "${TMP_DIR}/env/bin/activate"

# Install deps.
pip install --upgrade -r tandem_dqn/requirements.txt

# Run small-scale example for a few thousand steps.
python -m tandem_dqn.run_tandem \
  --environment_name=space_invaders \
  --replay_capacity=10000 --target_network_update_period=40 \
  --num_iterations=2 --num_train_frames=5000 --num_eval_frames=500 \
  --jax_platform_name=cpu

# Clean up.
rm -r ${TMP_DIR}
echo "Test run finished."
