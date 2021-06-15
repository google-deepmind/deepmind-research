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

echo "
These scripts are provided for illustrative purposes. It is not practical for
actual training since it only uses a single machine, and likely requires
reducing the batch size and/or model size to fit on a single GPU.

For the actual submission we used training distributed in different ways:
* We used 4x Cloud TPU v4s to implement batch parallelism, so batch
  size is effectively 8 times larger than the values in the config.
* We used separate CPUs to subsample neighborhoods around each central node, to
  maximize TPU usage by always having a batch ready for the next iteration.
* We ran the online early-stopping evaluator on a separate machine with an
  NVIDIA V100 GPU.
* We run identical replicas of all of the above for each of the 20 models
  trained.

Using those mechanisms, training runs at ~2 steps per second, reaching 500k
steps in under 3 days.
"

echo "
Pre-requisites (See README):
* Python dependencies have been installed.
* pre-processed data is available in the task dir.
"

read -p "Press enter to continue"

# During early stopping seed/selection we ensembled 5 iterations.
EPOCHS_TO_ENSEMBLE=5

DATA_ROOT=${TASK_ROOT}/data/
CHECKPOINT_DIR=${TASK_ROOT}/checkpoints/

# We run two seeds for each model of the k=10 k-fold.
BASE_SEED=100
for SEED_OFFSET in 0 10; do
for K_FOLD_INDEX in {0..9}; do
  MODEL_SEED=`expr ${BASE_SEED} + ${SEED_OFFSET} + ${K_FOLD_INDEX}`
  SUFFIX=k${K_FOLD_INDEX}_seed${MODEL_SEED}
  echo "Running k=${K_FOLD_INDEX} with init seed ${MODEL_SEED}"

  # This runs training (each model is trained on train split + 90% of the
  # validation split) with early stopping, storing both "latest" model and
  # "best" early-stopped model at `--config.checkpoint_dir`.

  # Models are early stopped based on accuracy of on 10% of the validation data
  # (each K_FOLD_INDEX leaves a different 10% of data out from training) left
  # out from training.

  # Models are stored at the end of training. Intermediate models can also be
  # stored while training by sending a SIGINT signal (Ctrl+C) which will not
  # interrupt the training.

  # It is possible to interrupt training using (Ctrl+\) and then continue
  # passing the corresponding `--config.restore_path=${RESTORE_PATH}` which is
  # stored when interrupting.

  python "${SCRIPT_DIR}"/experiment.py \
      --jaxline_mode="train_eval_multithreaded" \
      --config="${SCRIPT_DIR}"/config.py  \
      --config.random_seed=${MODEL_SEED} \
      --config.checkpoint_dir=${CHECKPOINT_DIR}/${SUFFIX} \
      --config.experiment_kwargs.config.dataset_kwargs.data_root=${DATA_ROOT} \
      --config.experiment_kwargs.config.dataset_kwargs.k_fold_split_id=${K_FOLD_INDEX} \
      --config.experiment_kwargs.config.num_eval_iterations_to_ensemble=${EPOCHS_TO_ENSEMBLE} \
      --config.experiment_kwargs.config.eval.split="valid"

done
done

# Each of the 20 (two for each value of k) jobs generate paths of the form:
# RESTORE_PATH=${--config.checkpoint_dir}/models/best/step_${STEP}_${TIMESTAMP}
# From each pair, the best one is selected (based on the validation accuracy
# as reported on the logs or in tensorboard events also stored at
# `${--config.checkpoint_dir}/eval`). These can then be used as the
# "RESTORE_PATHS" for `./run_pretrained_eval.sh`.


echo "Done"
