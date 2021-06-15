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
#!/bin/bash

set -e

while getopts ":i:o:" opt; do
  case ${opt} in
    i )
      INPUT_DIR=$OPTARG
      ;;
    o )
      TASK_ROOT=$OPTARG
      ;;
    \? )
      echo "Usage: organize_data.sh -i <Downloaded data dir> -o <Task root directory>"
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# Get this script's directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [[ -z "${INPUT_DIR}" ]]; then
  echo "Need INPUT_DIR argument (-i <INPUT_DIR>)"
  exit 1
fi

if [[ -z "${TASK_ROOT}" ]]; then
  echo "Need TASK_ROOT argument (-o <TASK_ROOT>)"
  exit 1
fi

DATA_ROOT="${TASK_ROOT}"/data

# Create raw directory to move all files to it.
mkdir "${INPUT_DIR}"/mag240m_kddcup2021/raw

mv "${INPUT_DIR}"/mag240m_kddcup2021/processed/paper/node_feat.npy \
   "${INPUT_DIR}"/mag240m_kddcup2021/processed/paper/node_label.npy \
   "${INPUT_DIR}"/mag240m_kddcup2021/processed/paper/node_year.npy \
   "${DATA_ROOT}"/raw
mv "${INPUT_DIR}"/mag240m_kddcup2021/processed/author___affiliated_with___institution/edge_index.npy \
   "${DATA_ROOT}"/raw/author_affiliated_with_institution_edges.npy
mv "${ROOT}"/mag240m_kddcup2021/processed/author___writes___paper/edge_index.npy \
   "${DATA_ROOT}"/raw/author_writes_paper_edges.npy
mv "${ROOT}"/mag240m_kddcup2021/processed/paper___cites___paper/edge_index.npy \
   "${DATA_ROOT}"/raw/paper_cites_paper_edges.npy

# Split and save the train/valid/test indices to the raw directory, with names
"train_idx.npy", "valid_idx.npy", "test_idx.npy":
python3 "${SCRIPT_DIR}"/split_and_save_indices.py --data_root=${DATA_ROOT}
