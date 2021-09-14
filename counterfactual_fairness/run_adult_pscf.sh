#!/bin/bash
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Fail on any error.
set -e

# Display commands being run.
set -x

TMP_DIR=`mktemp -d`

virtualenv --python=python3.6 "${TMP_DIR}/env"
source "${TMP_DIR}/env/bin/activate"

# Install dependencies.
pip install --upgrade -r counterfactual_fairness/requirements.txt

# Download minimal dataset
DATA_DIR="${TMP_DIR}/adult"
bash counterfactual_fairness/download_dataset.sh ${TMP_DIR}

# Train for a few steps.
python -m counterfactual_fairness.adult_pscf --config="counterfactual_fairness/adult_pscf_config.py" --dataset_dir=${DATA_DIR} --num_steps=10

# Clean up.
rm -r ${TMP_DIR}
echo "Test run complete."
