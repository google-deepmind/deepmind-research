#!/bin/sh
# Copyright 2019 Deepmind Technologies Limited.
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
set -e

echo "downloading checkpoints from GCP"
iodine/download_checkpoints.sh

python3 -m venv iodine_venv
source iodine_venv/bin/activate
pip3 install --upgrade setuptools wheel
pip3 install -r iodine/requirements.txt

# Get some fake data and put it where the real multi_objects_dataset files live.
mkdir -p iodine/multi_object_datasets
cp iodine/test_data/tetrominoes_mini.tfrecords iodine/multi_object_datasets/tetrominoes_train.tfrecords

# Run training with a cut down size.
python3 -m iodine.main \
  -f with tetrominoes \
  data.shuffle_buffer=2 \
  data.batch_size=2 \
  n_z=4 \
  num_components=3 \
  stop_after_steps=11
