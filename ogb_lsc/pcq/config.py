# Copyright 2021 DeepMind Technologies Limited.
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

"""Experiment config for PCQM4M-LSC entry."""

from jaxline import base_config
from ml_collections import config_dict


def get_config(debug: bool = False) -> config_dict.ConfigDict:
  """Get Jaxline experiment config."""
  config = base_config.get_base_config()
  # E.g. '/data/pretrained_models/k0_seed100' (and set k_fold_split_id=0, below)
  config.restore_path = config_dict.placeholder(str)

  training_batch_size = 64
  eval_batch_size = 64

  ## Experiment config.
  loss_config_name = 'RegressionLossConfig'
  loss_kwargs = dict(
      exponent=1.,  # 2 for l2 loss, 1 for l1 loss, etc...
  )

  dataset_config = dict(
      data_root=config_dict.placeholder(str),
      augment_with_random_mirror_symmetry=True,
      k_fold_split_id=config_dict.placeholder(int),
      num_k_fold_splits=config_dict.placeholder(int),
      # Options: "in" or "out".
      # Filter=in would keep the samples with nans in the conformer features.
      # Filter=out would keep the samples with no NaNs anywhere in the conformer
      # features.
      filter_in_or_out_samples_with_nans_in_conformers=(
          config_dict.placeholder(str)),
      cached_conformers_file=config_dict.placeholder(str))

  model_config = dict(
      mlp_hidden_size=512,
      mlp_layers=2,
      latent_size=512,
      use_layer_norm=False,
      num_message_passing_steps=32,
      shared_message_passing_weights=False,
      mask_padding_graph_at_every_step=True,
      loss_config_name=loss_config_name,
      loss_kwargs=loss_kwargs,
      processor_mode='resnet',
      global_reducer='sum',
      node_reducer='sum',
      dropedge_rate=0.1,
      dropnode_rate=0.1,
      aux_multiplier=0.1,
      add_relative_distance=True,
      add_relative_displacement=True,
      add_absolute_positions=False,
      position_normalization=2.,
      relative_displacement_normalization=1.,
      ignore_globals=False,
      ignore_globals_from_final_layer_for_predictions=True,
  )

  if debug:
    # Make network smaller.
    model_config.update(dict(
        mlp_hidden_size=32,
        mlp_layers=1,
        latent_size=32,
        num_message_passing_steps=1))

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              debug=debug,
              predictions_dir=config_dict.placeholder(str),
              ema=True,
              ema_decay=0.9999,
              sample_random=0.05,
              optimizer=dict(
                  name='adam',
                  optimizer_kwargs=dict(b1=.9, b2=.95),
                  lr_schedule=dict(
                      warmup_steps=int(5e4),
                      decay_steps=int(5e5),
                      init_value=1e-5,
                      peak_value=1e-4,
                      end_value=0.,
                  ),
              ),
              model=model_config,
              dataset_config=dataset_config,
              # As a rule of thumb, use the following statistics:
              # Avg. # nodes in graph: 16.
              # Avg. # edges in graph: 40.
              training=dict(
                  dynamic_batch_size={
                      'n_node': 256 if debug else 16 * training_batch_size,
                      'n_edge': 512 if debug else 40 * training_batch_size,
                      'n_graph': 2 if debug else training_batch_size,
                  },),
              evaluation=dict(
                  split='valid',
                  dynamic_batch_size=dict(
                      n_node=256 if debug else 16 * eval_batch_size,
                      n_edge=512 if debug else 40 * eval_batch_size,
                      n_graph=2 if debug else eval_batch_size,
                  )))))

  ## Training loop config.
  config.training_steps = int(5e6)
  config.checkpoint_dir = '/tmp/checkpoint/pcq/'
  config.train_checkpoint_all_hosts = False
  config.save_checkpoint_interval = 300
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60
  config.best_model_eval_metric = 'mae'
  config.best_model_eval_metric_higher_is_better = False

  return config
