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
      echo "Usage: run_pretrained_eval.sh -r <Task root directory>"
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
Note this script may take several days to run with default parameters on
a single machine. Reducing EPOCHS_TO_ENSEMBLE from 50 to < 5 should yield
slighly lower validation performance but possibly similar test performance.
"

echo "
Pre-requisites (See README):
* Python dependencies have been installed.
* pre-processed data is available in the task dir.
* pre-trained model weights are available in the task dir.
"

read -p "Press enter to continue"


# Can set this to "valid" or "test".
# On test the results will be ensembled for 10 models, and a submission file
# will be created.
# On validation, the results will be "gathered" for 10 models. This is because
# each model is trained on the train split + 90% of the validation split,
# and only evaluated on the remaining 10%, such that each validation paper
# is left out from training in exactly one of the 10 models.
SPLIT="test"

# We used 50 epochs in the submission to sample different subgraphs around
# each central node.
# For each of the 10 models in the ensemble, a single epoch takes about 1
# 10-25 minutes (depending on the GPU) on the test set, and about 2-3 minutes
# for the corresponding k-fold split of the validation set.
EPOCHS_TO_ENSEMBLE=50

DATA_ROOT=${TASK_ROOT}/data/
MODELS_ROOT=${TASK_ROOT}/models/
CHECKPOINT_DIR=${TASK_ROOT}/checkpoints/
OUTPUT_DIR=${TASK_ROOT}/predictions/

# We run two seeds for each model of the k=10 k-fold
# first seed group:  [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
# second seed group: [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
# Thes are the seeds that was selected based on cross validation for each fold.
BEST_SEEDS=(100 111 102 113 104 105 106 107 108 109)
for K_FOLD_INDEX in {0..9}; do
  SEED=${BEST_SEEDS[${K_FOLD_INDEX}]}
  RESTORE_PATH=${MODELS_ROOT}/k${K_FOLD_INDEX}_seed${SEED}
  echo "Running k=${K_FOLD_INDEX} on ${SPLIT} split using ${RESTORE_PATH}"
  # This saves the predictions for the K_FOLD_INDEXd'th model in the k-fold to
  # "config.experiment_kwargs.config.predictions_dir" for subsequent ensembling
  python "${SCRIPT_DIR}"/experiment.py \
      --jaxline_mode="eval" \
      --config="${SCRIPT_DIR}"/config.py  \
      --config.one_off_evaluate=True \
      --config.checkpoint_dir=${CHECKPOINT_DIR}/${K_FOLD_INDEX} \
      --config.restore_path=${RESTORE_PATH} \
      --config.experiment_kwargs.config.dataset_kwargs.data_root=${DATA_ROOT} \
      --config.experiment_kwargs.config.dataset_kwargs.k_fold_split_id=${K_FOLD_INDEX} \
      --config.experiment_kwargs.config.num_eval_iterations_to_ensemble=${EPOCHS_TO_ENSEMBLE} \
      --config.experiment_kwargs.config.predictions_dir=${OUTPUT_DIR}/${K_FOLD_INDEX} \
      --config.experiment_kwargs.config.eval.split=${SPLIT}

done


python "${SCRIPT_DIR}"/ensemble_predictions.py \
     --split=${SPLIT} \
     --data_root=${DATA_ROOT} \
     --predictions_path=${OUTPUT_DIR} \
     --output_path=${OUTPUT_DIR}


echo "Done"
