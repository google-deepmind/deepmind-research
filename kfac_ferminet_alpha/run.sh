#!/bin/bash
# Copyright 2020 DeepMind Technologies Limited.
#
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

# This script installs kfac_ferminet_alpha in a clean virtualenv and runs an
# example training loop. It is designed to be run from the parent directory,
# e.g.:
#
# git clone git@github.com:deepmind/deepmind-research.git
# cd deepmind_research
# kfac_ferminet_alpha/run.sh

python3 -m venv /tmp/kfac_ferminet_alpha_example
source /tmp/kfac_ferminet_alpha_example/bin/activate
pip3 install -U pip
pip3 install -r kfac_ferminet_alpha/requirements.txt
# For a GPU you have to do:
# pip3 install --upgrade jax jaxlib==0.1.64+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip3 install jaxlib
pip3 install kfac_ferminet_alpha/
python3 kfac_ferminet_alpha/example.py
