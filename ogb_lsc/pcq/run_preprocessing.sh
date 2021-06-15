#!/bin/bash
# Copyright 2021 DeepMind Technologies Limited.
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
set -x

while getopts ":r:" opt; do
  case ${opt} in
    r )
      TASK_ROOT=$OPTARG
      ;;
    \? )
      echo "Usage: run_training.sh -r <Task root directory>"
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# Get this script's directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


DATA_ROOT=${TASK_ROOT}/data/


python "${SCRIPT_DIR}"/generate_validation_splits.py \
  --output_dir="${DATA_ROOT}/k_fold_splits"

mkdir -p ${DATA_ROOT}/preprocessed/
python "${SCRIPT_DIR}"/generate_conformer_features.py \
  --splits="test valid train" \
  --num_parallel_procs=32 \
  --output_file="${DATA_ROOT}/preprocessed/smile_to_conformer.pkl"
