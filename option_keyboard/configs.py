# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Environment configurations."""


def get_task_config():
  return dict(
      arena_size=11,
      num_channels=2,
      max_num_steps=50,  # 50 for the actual task.
      num_init_objects=10,
      object_priors=[0.5, 0.5],
      egocentric=True,
      rewarder="BalancedCollectionRewarder",
  )


def get_pretrain_config():
  return dict(
      arena_size=11,
      num_channels=2,
      max_num_steps=40,  # 40 for pretraining.
      num_init_objects=10,
      object_priors=[0.5, 0.5],
      egocentric=True,
      default_w=(1, 1),
  )


def get_fig4_task_config():
  return dict(
      arena_size=11,
      num_channels=2,
      max_num_steps=50,  # 50 for the actual task.
      num_init_objects=10,
      object_priors=[0.5, 0.5],
      egocentric=True,
      default_w=(1, -1),
  )


def get_fig5_task_config(default_w):
  return dict(
      arena_size=11,
      num_channels=2,
      max_num_steps=50,  # 50 for the actual task.
      num_init_objects=10,
      object_priors=[0.5, 0.5],
      egocentric=True,
      default_w=default_w,
  )
