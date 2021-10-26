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

# Script to execute a single configuration on all datasets.
if [[ "$#" -eq 2 ]]; then
  readonly CONFIG_NAME="$1"
  readonly NUM_SWEEPS="$2"
else
   echo "You must provide exactly two arguments - the configuration name and " \
   "how many sweeps it contains. For example:"
   echo "./launch_all.sh sym_metric_hgn_plus_plus_sweep 1"
   exit 2
fi

DATASETS=(
  "toy_physics/mass_spring"
  "toy_physics/mass_spring_colors"
  "toy_physics/mass_spring_colors_friction"
  "toy_physics/pendulum"
  "toy_physics/pendulum_colors"
  "toy_physics/pendulum_colors_friction"
  "toy_physics/two_body"
  "toy_physics/two_body_colors"
  "toy_physics/double_pendulum"
  "toy_physics/double_pendulum_colors"
  "toy_physics/double_pendulum_colors_friction"
  "molecular_dynamics/lj_4"
  "molecular_dynamics/lj_16"
  "multi_agent/rock_paper_scissors"
  "multi_agent/matching_pennies"
  "mujoco_room/circle"
  "mujoco_room/spiral"
)

readonly DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for dataset in "${DATASETS[@]}"; do
  "${DIR}/launch_local.sh" "${CONFIG_NAME}" "${NUM_SWEEPS}" "${dataset}"
done
