#!/bin/bash
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
# WikiGraphs is licensed under the terms of the Creative Commons
# Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.
#
# WikiText-103 data (unchanged) is licensed by Salesforce.com, Inc. under the
# terms of the Creative Commons Attribution-ShareAlike 4.0 International
# (CC BY-SA 4.0) license. You can find details about CC BY-SA 4.0 at:
#
#     https://creativecommons.org/licenses/by-sa/4.0/legalcode
#
# Freebase data is licensed by Google LLC under the terms of the Creative
# Commons CC BY 4.0 license. You may obtain a copy of the License at:
#
#     https://creativecommons.org/licenses/by/4.0/legalcode
#
# ==============================================================================
BASE_DIR=/tmp/data

# wikitext-103
TARGET_DIR=${BASE_DIR}/wikitext-103
mkdir -p ${TARGET_DIR}
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -P ${TARGET_DIR}
unzip ${TARGET_DIR}/wikitext-103-v1.zip -d ${TARGET_DIR}
mv ${TARGET_DIR}/wikitext-103/* ${TARGET_DIR}
rm -rf ${TARGET_DIR}/wikitext-103 ${TARGET_DIR}/wikitext-103-v1.zip

# wikitext-103-raw
TARGET_DIR=${BASE_DIR}/wikitext-103-raw
mkdir -p ${TARGET_DIR}
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip -P ${TARGET_DIR}
unzip ${TARGET_DIR}/wikitext-103-raw-v1.zip -d ${TARGET_DIR}
mv ${TARGET_DIR}/wikitext-103-raw/* ${TARGET_DIR}
rm -rf ${TARGET_DIR}/wikitext-103-raw ${TARGET_DIR}/wikitext-103-raw-v1.zip


# processed freebase graphs
FREEBASE_TARGET_DIR=/tmp/data
mkdir -p ${FREEBASE_TARGET_DIR}/packaged/
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uuSS2o72dUCJrcLff6NBiLJuTgSU-uRo' -O ${FREEBASE_TARGET_DIR}/packaged/max256.tar
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nOfUq3RUoPEWNZa2QHXl2q-1gA5F6kYh' -O ${FREEBASE_TARGET_DIR}/packaged/max512.tar
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uuJwkocJXG1UcQ-RCH3JU96VsDvi7UD2' -O ${FREEBASE_TARGET_DIR}/packaged/max1024.tar

for version in max1024 max512 max256
do
  output_dir=${FREEBASE_TARGET_DIR}/freebase/${version}/
  mkdir -p ${output_dir}
  tar -xvf ${FREEBASE_TARGET_DIR}/packaged/${version}.tar -C ${output_dir}
done
rm -rf ${FREEBASE_TARGET_DIR}/packaged
