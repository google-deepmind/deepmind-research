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
#     sh download_dataset.sh ${DATASET_NAME} ${OUTPUT_DIR}
# Example:
#     sh download_dataset.sh flag_simple /tmp/

set -e

DATASET_NAME="${1}"
OUTPUT_DIR="${2}/${DATASET_NAME}"

# Validate inputs
if [ -z "${DATASET_NAME}" ] || [ -z "${2}" ]; then
    echo "Usage: sh download_dataset.sh DATASET_NAME OUTPUT_DIR"
    echo "Example: sh download_dataset.sh flag_simple /tmp/"
    echo "Available datasets: flag_simple, cylinder_flow, deforming_plate, sphere_simple"
    exit 1
fi

# Ensure no double slash in URL construction  
BASE_URL="https://storage.googleapis.com/dm-meshgraphnets"

echo "Downloading dataset: ${DATASET_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Base URL: ${BASE_URL}/${DATASET_NAME}/"

mkdir -p ${OUTPUT_DIR}
for file in meta.json train.tfrecord valid.tfrecord test.tfrecord
do
    DOWNLOAD_URL="${BASE_URL}/${DATASET_NAME}/${file}"
    echo "Downloading: ${DOWNLOAD_URL}"
    
    # Download with error handling
    if wget -O "${OUTPUT_DIR}/${file}" "${DOWNLOAD_URL}"; then
        echo "✓ Successfully downloaded: ${file}"
    else
        echo "✗ Failed to download: ${file}"
        echo "  URL: ${DOWNLOAD_URL}"
        echo "  Please check if the dataset name is correct."
        exit 1
    fi
done

echo "✅ Dataset download completed successfully!"
echo "📁 Files saved to: ${OUTPUT_DIR}"
