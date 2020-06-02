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

# Use this script to download the sketchy dataset.
#
# You will need to extract the archive files before using them.  Each archive
# (except for the last in each group) contains 100 episodes.

if [ -z "$1" ]; then
  echo "Usage: $(basename "$0") download_folder" >&2
  exit 1
fi

DOWNLOAD_FOLDER="$1"
NUM_PARALLEL_DOWNLOADS="4"  # set the number of download workers
DATA_URL="https://storage.googleapis.com/sketchy-data"

function download_shards {
  # Usage: download_shards prefix num_shards
  local PREFIX="$1"
  local LIMIT="$(printf "%05d" "$2")"
  # Avoid leading zeros or this will be interpreted as an octal number.
  local MAX="$(("$2"-1))"

  (
    for IDX in $(seq -f'%05.0f' 0 "$MAX"); do
      echo "${PREFIX}-${IDX}-of-${LIMIT}.tar.bz2"
    done
  ) | xargs -I{} -n1 -P"${NUM_PARALLEL_DOWNLOADS}" \
    curl "${DATA_URL}/{}" --output "${DOWNLOAD_FOLDER}/{}"
}

# This is the metadata. This file is small, you always want it.
curl "${DATA_URL}/metadata.sqlite" --output "${DOWNLOAD_FOLDER}/metadata.sqlite"

# Download these files if you want all and only the episodes with an associated
# reward sequence.
#
# sqlite3 metadata.sqlite <<EOF
# SELECT DISTINCT(Episodes.DataPath)
# FROM Episodes, RewardSequences
# WHERE Episodes.EpisodeId = RewardSequences.EpisodeId
# EOF
#
# If you are downloading the full dataset then you do not need these files. The
# episodes they contain are included in the other subsets.
#
# download_shards episodes_with_rewards 58

# These files contain a curated set of high quality demonstrations for the
# lift_green task.
#
# sqlite3 metadata.sqlite <<EOF
# SELECT Episodes.DataPath
# FROM Episodes, EpisodeTags
# WHERE EpisodeTags.Tag='lift_green__demos'
# AND Episodes.EpisodeId = EpisodeTags.EpisodeId
# EOF
#
download_shards lift_green__demos 2

# These files contain a broader set of episodes for the lift_green task.  If you
# download these you should also download the lift_green__demos files.
#
# sqlite3 metadata.sqlite <<EOF
# SELECT Episodes.DataPath
# FROM Episodes, EpisodeTags
# WHERE EpisodeTags.Tag='lift_green__episodes'
# AND Episodes.EpisodeId = EpisodeTags.EpisodeId
# AND EpisodeTags.EpisodeId NOT IN (
#   SELECT ET.EpisodeId
#   FROM EpisodeTags AS ET
#   WHERE ET.Tag IN ('lift_green__demos')
#   )
# EOF
#
download_shards lift_green__episodes 70

# These files contain a curated set of high quality demonstrations for the
# stack_green_on_red task.
#
# sqlite3 metadata.sqlite <<EOF
# SELECT Episodes.DataPath
# FROM Episodes, EpisodeTags
# WHERE EpisodeTags.Tag='stack_green_on_red__demos'
# AND Episodes.EpisodeId = EpisodeTags.EpisodeId
# AND EpisodeTags.EpisodeId NOT IN (
#   SELECT ET.EpisodeId
#   FROM EpisodeTags AS ET
#   WHERE ET.Tag IN ('lift_green__demos', 'lift_green__episodes')
#   )
# EOF
#
download_shards stack_green_on_red__demos 2

# These files contain a broader set of episodes for the stack_green_on_red task.
# If you download these you should also download the stack_green_on_red__demos
# files.
#
# sqlite3 metadata.sqlite <<EOF
# SELECT Episodes.DataPath
# FROM Episodes, EpisodeTags
# WHERE EpisodeTags.Tag='stack_green_on_red__episodes'
# AND Episodes.EpisodeId = EpisodeTags.EpisodeId
# AND EpisodeTags.EpisodeId NOT IN (
#   SELECT ET.EpisodeId
#   FROM EpisodeTags AS ET
#   WHERE ET.Tag IN (
#     'lift_green__demos',
#     'lift_green__episodes',
#     'stack_green_on_red__demos')
#   )
# EOF
#
download_shards stack_green_on_red__episodes 101

# These files contain a large variety of episodes using the same object set as
# the lift_green and stack_green_on_red tasks.  There are many tasks represented
# here, and the episode quality is highly variable.
#
# sqlite3 metadata.sqlite <<EOF
# SELECT Episodes.DataPath
# FROM Episodes, EpisodeTags
# WHERE EpisodeTags.Tag='rgb30__all'
# AND Episodes.EpisodeId = EpisodeTags.EpisodeId
# AND EpisodeTags.EpisodeId NOT IN (
#   SELECT ET.EpisodeId
#   FROM EpisodeTags AS ET
#   WHERE ET.Tag IN (
#     'lift_green__demos',
#     'lift_green__episodes',
#     'stack_green_on_red__demos',
#     'stack_green_on_red__episodes')
#   )
# EOF
#
download_shards rgb30__all 205

# These files contain a broad set of episodes for the pull_cloth_up task.
#
# sqlite3 metadata.sqlite <<EOF
# SELECT Episodes.DataPath
# FROM Episodes, EpisodeTags
# WHERE EpisodeTags.Tag='pull_cloth_up__episodes'
# AND Episodes.EpisodeId = EpisodeTags.EpisodeId
# EOF
#
download_shards pull_cloth_up__episodes 133

# These files contain a large variety of episodes using the same object set as
# the pull_cloth_up task.
#
# sqlite3 metadata.sqlite <<EOF
# SELECT Episodes.DataPath
# FROM Episodes, EpisodeTags
# WHERE EpisodeTags.Tag='deform8__all'
# AND Episodes.EpisodeId = EpisodeTags.EpisodeId
# AND EpisodeTags.EpisodeId NOT IN (
#   SELECT ET.EpisodeId
#   FROM EpisodeTags AS ET
#   WHERE ET.Tag IN ('pull_cloth_up__episodes')
#   )
# EOF
#
download_shards deform8__all 233
