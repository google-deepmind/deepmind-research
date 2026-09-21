#!/bin/bash
# Copyright 2020 DeepMind Technologies Limited.
#
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

# A script to execute a single configuration name on a given dataset.
if [[ "$#" -eq 3 ]]; then
  readonly CONFIG_NAME="$1"
  readonly NUM_SWEEPS="$2"
  readonly DATASET="$3"
else
   echo "You must provide exactly three arguments - the configuration name, " \
   "the number of sweeps it contains and the dataset name. For example:"
   echo "./launch_local.sh sym_metric_hgn_plus_plus_sweep 1 " \
   "toy_physics/mass_spring"
   exit 2
fi
echo "Running with config ${CONFIG_NAME} on ${DATASET}."

readonly EXPERIMENT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
readonly TRAIN_FILE="${EXPERIMENT_DIR}/jaxline_train.py"
readonly CONFIG_FILE="${EXPERIMENT_DIR}/jaxline_configs.py"

for sweep_id in $(seq 0 $((NUM_SWEEPS - 1))); do
  python3 "${TRAIN_FILE}" \
    --config="${CONFIG_FILE}:${CONFIG_NAME},${sweep_id},${DATASET}" \
    --jaxline_mode="train" \
    --logtostderr
done
