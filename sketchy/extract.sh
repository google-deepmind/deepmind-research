#!/bin/bash

# Copyright 2020 DeepMind Technologies Limited.
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

# Use this script to extract the downloaded sketchy dataset.

if [ -z "$1" ]; then
  echo "Usage: $(basename "$0") download_folder" >&2
  exit 1
fi

DOWNLOAD_FOLDER="$1"
NUM_PARALLEL_WORKERS="$(grep processor /proc/cpuinfo | wc -l)"

cd "$DOWNLOAD_FOLDER"

find . -name '*.tar.bz2' -print0 \
  | xargs -0 -n1 -P"$NUM_PARALLEL_WORKERS" tar xf
