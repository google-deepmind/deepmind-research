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

# Generates test predictions from trained model checkpoints.

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

SPLIT="test"

DATA_ROOT=${TASK_ROOT}/data/  # For valid k-fold splits.
MODELS_ROOT=${TASK_ROOT}/models/
CACHED_CONFORMERS_DIR=${TASK_ROOT}/online_conformers/$SPLIT
CACHED_CONFORMERS_FILE=${CACHED_CONFORMERS_DIR}/smiles_to_conformers.pkl
OUTPUT_DIR=${TASK_ROOT}/predictions/

# First generate and cache the conformer feature data for the test split.
mkdir -p ${CACHED_CONFORMERS_DIR}
time python "${SCRIPT_DIR}"/generate_conformer_features.py \
  --splits=$SPLIT \
  --num_parallel_procs=32 \
  --output_file=${CACHED_CONFORMERS_FILE}

seed=42
# Share GPU between two runs by disabling XLA memory pre-allocation.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

for seed_group in 0 1; do
  for k in `seq 0 9`; do
    # Conformer
    predictions_dir="${OUTPUT_DIR}/predictions/${SPLIT}/conformer/k${k}_seed${seed}"
    time python "${SCRIPT_DIR}"/experiment.py \
      --jaxline_mode="eval" \
      --config="${SCRIPT_DIR}"/config.py \
      --config.experiment_kwargs.config.evaluation.split=$SPLIT \
      --config.restore_path=${MODELS_ROOT}/conformer/k${k}_seed${seed} \
      --config.experiment_kwargs.config.predictions_dir=${predictions_dir} \
      --config.experiment_kwargs.config.dataset_config.data_root=${DATA_ROOT} \
      --config.experiment_kwargs.config.dataset_config.k_fold_split_id=$k \
      --config.experiment_kwargs.config.dataset_config.cached_conformers_file=${CACHED_CONFORMERS_FILE} \
      --config.experiment_kwargs.config.dataset_config.filter_in_or_out_samples_with_nans_in_conformers="out" \
      --config.experiment_kwargs.config.model.latent_size=256 \
      --config.experiment_kwargs.config.model.mlp_hidden_size=1024 \
      --config.experiment_kwargs.config.model.num_message_passing_steps=32 \
      --config.one_off_evaluate &


    # Non-Conformer
    predictions_dir="${OUTPUT_DIR}/predictions/${SPLIT}/non_conformer/k${k}_seed${seed}"
    time python "${SCRIPT_DIR}"/experiment.py \
      --jaxline_mode="eval" \
      --config="${SCRIPT_DIR}"/config.py \
      --config.experiment_kwargs.config.evaluation.split=$SPLIT \
      --config.restore_path=${MODELS_ROOT}/non_conformer/k${k}_seed${seed} \
      --config.experiment_kwargs.config.predictions_dir=${predictions_dir} \
      --config.experiment_kwargs.config.dataset_config.data_root=${DATA_ROOT} \
      --config.experiment_kwargs.config.dataset_config.k_fold_split_id=$k \
      --config.experiment_kwargs.config.dataset_config.cached_conformers_file=${CACHED_CONFORMERS_FILE} \
      --config.experiment_kwargs.config.dataset_config.filter_in_or_out_samples_with_nans_in_conformers="in" \
      --config.experiment_kwargs.config.model.num_message_passing_steps=50 \
      --config.experiment_kwargs.config.model.add_relative_distance=false \
      --config.experiment_kwargs.config.model.add_relative_displacement=false \
      --config.one_off_evaluate &

    wait

    ((seed=seed+1))

  done
done

# Ensemble predictions.
time python "${SCRIPT_DIR}"/ensemble_predictions.py \
  --seed_start=42 \
  --split=$SPLIT \
  --conformer_path="${OUTPUT_DIR}/predictions/${SPLIT}/conformer" \
  --non_conformer_path="${OUTPUT_DIR}/predictions/${SPLIT}/non_conformer" \
  --output_path="${OUTPUT_DIR}/ensembled_predictions/"
