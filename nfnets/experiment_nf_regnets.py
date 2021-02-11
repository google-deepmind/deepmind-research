# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
r"""ImageNet experiment with NF-RegNets."""

from ml_collections import config_dict
from nfnets import experiment


def get_config():
  """Return config object for training."""
  config = experiment.get_config()

  # Experiment config.
  train_batch_size = 1024  # Global batch size.
  images_per_epoch = 1281167
  num_epochs = 360
  steps_per_epoch = images_per_epoch / train_batch_size
  config.training_steps = ((images_per_epoch * num_epochs) // train_batch_size)
  config.random_seed = 0

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              lr=0.4,
              num_epochs=num_epochs,
              label_smoothing=0.1,
              model='NF_RegNet',
              image_size=224,
              use_ema=True,
              ema_decay=0.99999,  # Cinco nueves amigos
              ema_start=0,
              augment_name='mixup_cutmix',
              train_batch_size=train_batch_size,
              eval_batch_size=50,
              eval_subset='test',
              num_classes=1000,
              which_dataset='imagenet',
              which_loss='softmax_cross_entropy',  # One of softmax or sigmoid
              bfloat16=False,
              lr_schedule=dict(
                  name='WarmupCosineDecay',
                  kwargs=dict(num_steps=config.training_steps,
                              start_val=0,
                              min_val=0.001,
                              warmup_steps=5*steps_per_epoch),
                  ),
              lr_scale_by_bs=False,
              optimizer=dict(
                  name='SGD',
                  kwargs={'momentum': 0.9, 'nesterov': True,
                          'weight_decay': 5e-5,},
              ),
              model_kwargs=dict(
                  variant='B0',
                  width=0.75,
                  expansion=2.25,
                  se_ratio=0.5,
                  alpha=0.2,
                  stochdepth_rate=0.1,
                  drop_rate=None,
                  activation='silu',
                  ),

              )))

  # Set weight decay based on variant (scaled as 5e-5 + 1e-5 * level)
  variant = config.experiment_kwargs.config.model_kwargs.variant
  weight_decay = {'B0': 5e-5, 'B1': 6e-5, 'B2': 7e-5,
                  'B3': 8e-5, 'B4': 9e-5, 'B5': 1e-4}[variant]
  config.experiment_kwargs.config.optimizer.kwargs.weight_decay = weight_decay

  return config

Experiment = experiment.Experiment
