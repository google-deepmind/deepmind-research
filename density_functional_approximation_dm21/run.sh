#!/bin/bash
# Copyright 2021 DeepMind Technologies Limited.
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

# Install density_functional_approximation_dm21 in a clean virtualenv and run
# the tests.  This assumes the working directory is the top-level directory of
# the deepmind-research repository, i.e.:
# git clone git@github.com:deepmind/deepmind-research.git
# cd deepmind_research
# density_functional_approximation_dm21/run.sh

python3 -m venv /tmp/DM21
source /tmp/DM21/bin/activate
pip3 install -r density_functional_approximation_dm21/requirements.txt
pip install 'density_functional_approximation_dm21/[testing]'
py.test density_functional_approximation_dm21/
