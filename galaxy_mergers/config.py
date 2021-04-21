# Copyright 2021 DeepMind Technologies Limited.
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

"""Default config, focused on model evaluation."""

from ml_collections import config_dict


def get_config(filter_time_intervals=None):
  """Return config object for training."""
  config = config_dict.ConfigDict()
  config.eval_strategy = config_dict.ConfigDict()
  config.eval_strategy.class_name = 'OneDeviceConfig'
  config.eval_strategy.kwargs = config_dict.ConfigDict(
      dict(device_type='v100'))

  ## Experiment config.
  config.experiment_kwargs = config_dict.ConfigDict(dict(
      resnet_kwargs=dict(
          blocks_per_group_list=[3, 4, 6, 3],  # This choice is ResNet50.
          bn_config=dict(
              decay_rate=0.9,
              eps=1e-5),
          resnet_v2=False,
          additional_features_mode='mlp',
      ),
      optimizer_config=dict(
          class_name='Momentum',
          kwargs={'momentum': 0.9},
          # Set up the learning rate schedule.
          lr_init=0.025,
          lr_factor=0.1,
          lr_schedule=(50e3, 100e3, 150e3),
          gradient_clip=5.,
      ),
      l2_regularization=1e-4,
      total_train_batch_size=128,
      train_net_args={'is_training': True},
      eval_batch_size=128,
      eval_net_args={'is_training': True},
      data_config=dict(
          # dataset loading
          dataset_path=None,
          num_val_splits=10,
          val_split=0,

          # image cropping
          image_size=(80, 80, 7),
          train_crop_type='crop_fixed',
          test_crop_type='crop_fixed',
          n_crop_repeat=1,

          train_augmentations=dict(
              rotation_and_flip=True,
              rescaling=True,
              translation=True,
          ),

          test_augmentations=dict(
              rotation_and_flip=False,
              rescaling=False,
              translation=False,
          ),
          test_time_ensembling='sum',

          num_eval_buckets=5,
          eval_confidence_interval=95,

          task='grounded_unnormalized_regression',
          loss_config=dict(
              loss='mse',
              mse_normalize=False,
          ),
          model_uncertainty=True,
          additional_features='',
          time_filter_intervals=filter_time_intervals,
          class_boundaries={
              '0': [[-1., 0]],
              '1': [[0, 1.]]
          },
          frequencies_to_use='all',
      ),
      n_train_epochs=100
      ))

  return config
