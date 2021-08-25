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
"""Config file for BYOL experiment."""

from byol.utils import dataset


# Preset values for certain number of training epochs.
_LR_PRESETS = {40: 0.45, 100: 0.45, 300: 0.3, 1000: 0.2}
_WD_PRESETS = {40: 1e-6, 100: 1e-6, 300: 1e-6, 1000: 1.5e-6}
_EMA_PRESETS = {40: 0.97, 100: 0.99, 300: 0.99, 1000: 0.996}


def get_config(num_epochs: int, batch_size: int):
  """Return config object, containing all hyperparameters for training."""
  train_images_per_epoch = dataset.Split.TRAIN_AND_VALID.num_examples

  assert num_epochs in [40, 100, 300, 1000]

  config = dict(
      random_seed=0,
      num_classes=1000,
      batch_size=batch_size,
      max_steps=num_epochs * train_images_per_epoch // batch_size,
      enable_double_transpose=True,
      base_target_ema=_EMA_PRESETS[num_epochs],
      network_config=dict(
          projector_hidden_size=4096,
          projector_output_size=256,
          predictor_hidden_size=4096,
          encoder_class='ResNet50',  # Should match a class in utils/networks.
          encoder_config=dict(
              resnet_v2=False,
              width_multiplier=1),
          bn_config={
              'decay_rate': .9,
              'eps': 1e-5,
              # Accumulate batchnorm statistics across devices.
              # This should be equal to the `axis_name` argument passed
              # to jax.pmap.
              'cross_replica_axis': 'i',
              'create_scale': True,
              'create_offset': True,
          }),
      optimizer_config=dict(
          weight_decay=_WD_PRESETS[num_epochs],
          eta=1e-3,
          momentum=.9,
      ),
      lr_schedule_config=dict(
          base_learning_rate=_LR_PRESETS[num_epochs],
          warmup_steps=10 * train_images_per_epoch // batch_size,
      ),
      evaluation_config=dict(
          subset='test',
          batch_size=100,
      ),
      checkpointing_config=dict(
          use_checkpointing=True,
          checkpoint_dir='/tmp/byol',
          save_checkpoint_interval=300,
          filename='pretrain.pkl'
      ),
  )

  return config
