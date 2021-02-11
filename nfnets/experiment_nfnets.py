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
r"""ImageNet experiment with NFNets."""

import haiku as hk
from ml_collections import config_dict
from nfnets import experiment
from nfnets import optim


def get_config():
  """Return config object for training."""
  config = experiment.get_config()

  # Experiment config.
  train_batch_size = 4096  # Global batch size.
  images_per_epoch = 1281167
  num_epochs = 360
  steps_per_epoch = images_per_epoch / train_batch_size
  config.training_steps = ((images_per_epoch * num_epochs) // train_batch_size)
  config.random_seed = 0

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              lr=0.1,
              num_epochs=num_epochs,
              label_smoothing=0.1,
              model='NFNet',
              image_size=224,
              use_ema=True,
              ema_decay=0.99999,
              ema_start=0,
              augment_name=None,
              augment_before_mix=False,
              eval_preproc='resize_crop_32',
              train_batch_size=train_batch_size,
              eval_batch_size=50,
              eval_subset='test',
              num_classes=1000,
              which_dataset='imagenet',
              which_loss='softmax_cross_entropy',  # One of softmax or sigmoid
              bfloat16=True,
              lr_schedule=dict(
                  name='WarmupCosineDecay',
                  kwargs=dict(num_steps=config.training_steps,
                              start_val=0,
                              min_val=0.0,
                              warmup_steps=5*steps_per_epoch),
                  ),
              lr_scale_by_bs=True,
              optimizer=dict(
                  name='SGD_AGC',
                  kwargs={'momentum': 0.9, 'nesterov': True,
                          'weight_decay': 2e-5,
                          'clipping': 0.01, 'eps': 1e-3},
              ),
              model_kwargs=dict(
                  variant='F0',
                  width=1.0,
                  se_ratio=0.5,
                  alpha=0.2,
                  stochdepth_rate=0.25,
                  drop_rate=None,  # Use native drop-rate
                  activation='gelu',
                  final_conv_mult=2,
                  final_conv_ch=None,
                  use_two_convs=True,
                  ),
              )))

  # Unlike NF-RegNets, use the same weight decay for all, but vary RA levels
  variant = config.experiment_kwargs.config.model_kwargs.variant
  # RandAugment levels (e.g. 405 = 4 layers, magnitude 5, 205 = 2 layers, mag 5)
  augment = {'F0': '405', 'F1': '410', 'F2': '410', 'F3': '415',
             'F4': '415', 'F5': '415', 'F6': '415', 'F7': '415'}[variant]
  aug_base_name = 'cutmix_mixup_randaugment'
  config.experiment_kwargs.config.augment_name = f'{aug_base_name}_{augment}'

  return config


class Experiment(experiment.Experiment):
  """Experiment with correct parameter filtering for applying AGC."""

  def _make_opt(self):
    # Separate conv params and gains/biases
    def pred_gb(mod, name, val):
      del mod, val
      return (name in ['scale', 'offset', 'b']
              or 'gain' in name or 'bias' in name)
    gains_biases, weights = hk.data_structures.partition(pred_gb, self._params)
    def pred_fc(mod, name, val):
      del name, val
      return 'linear' in mod and 'squeeze_excite' not in mod
    fc_weights, weights = hk.data_structures.partition(pred_fc, weights)
    # Lr schedule with batch-based LR scaling
    if self.config.lr_scale_by_bs:
      max_lr = (self.config.lr * self.config.train_batch_size) / 256
    else:
      max_lr = self.config.lr
    lr_sched_fn = getattr(optim, self.config.lr_schedule.name)
    lr_schedule = lr_sched_fn(max_val=max_lr, **self.config.lr_schedule.kwargs)
    # Optimizer; no need to broadcast!
    opt_kwargs = {key: val for key, val in self.config.optimizer.kwargs.items()}
    opt_kwargs['lr'] = lr_schedule
    opt_module = getattr(optim, self.config.optimizer.name)
    self.opt = opt_module([{'params': gains_biases, 'weight_decay': None,},
                           {'params': fc_weights, 'clipping': None},
                           {'params': weights}], **opt_kwargs)
    if self._opt_state is None:
      self._opt_state = self.opt.states()
    else:
      self.opt.plugin(self._opt_state)
