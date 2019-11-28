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

# Usage:
# user@host:/path/to/deepmind_research$ unrestricted_advx/run.sh

# Sets up virtual environment, install dependencies, and runs evaluation script
python3 -m venv unrestricted_venv
source unrestricted_venv/bin/activate
pip install -r unrestricted_advx/requirements.txt
git clone git@github.com:google/unrestricted-adversarial-examples.git
pip install -e unrestricted-adversarial-examples/bird-or-bicycle
pip install -e unrestricted-adversarial-examples/unrestricted-advex
