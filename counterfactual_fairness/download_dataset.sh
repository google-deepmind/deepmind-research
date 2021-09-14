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
#
# Usage:
#     sh download_dataset.sh ${OUTPUT_DIR}
# Example:
#     sh download_dataset.sh uci_adult

set -e

OUTPUT_DIR="${1}"

BASE_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"

mkdir -p ${OUTPUT_DIR}
for file in adult.data adult.test
do
wget -O "${OUTPUT_DIR}/${file}" "${BASE_URL}${file}"
done
