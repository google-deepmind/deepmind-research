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

"""Experiment config for MAG240M-LSC entry."""

from jaxline import base_config
from ml_collections import config_dict


def get_config(debug: bool = False) -> config_dict.ConfigDict:
  """Get Jaxline experiment config."""
  config = base_config.get_base_config()
  config.random_seed = 42
  # E.g. '/data/pretrained_models/k0_seed100' (and set k_fold_split_id=0, below)
  config.restore_path = config_dict.placeholder(str)
  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              debug=debug,
              predictions_dir=config_dict.placeholder(str),
              # 5 for model selection and early stopping, 50 for final eval.
              num_eval_iterations_to_ensemble=5,
              dataset_kwargs=dict(
                  data_root='/data/',
                  online_subsampling_kwargs=dict(
                      max_nb_neighbours_per_type=[
                          [[40, 20, 0, 40], [0, 0, 0, 0], [0, 0, 0, 0]],
                          [[40, 20, 0, 40], [40, 0, 10, 0], [0, 0, 0, 0]],
                      ],
                      remove_future_nodes=True,
                      deduplicate_nodes=True,
                  ),
                  ratio_unlabeled_data_to_labeled_data=10.0,
                  k_fold_split_id=config_dict.placeholder(int),
                  use_all_labels_when_not_training=False,
                  use_dummy_adjacencies=debug,
              ),
              optimizer=dict(
                  name='adamw',
                  kwargs=dict(weight_decay=1e-5, b1=0.9, b2=0.999),
                  learning_rate_schedule=dict(
                      use_schedule=True,
                      base_learning_rate=1e-2,
                      warmup_steps=50000,
                      total_steps=config.get_ref('training_steps'),
                  ),
              ),
              model_config=dict(
                  mlp_hidden_sizes=[32] if debug else [512],
                  latent_size=32 if debug else 256,
                  num_message_passing_steps=2 if debug else 4,
                  activation='relu',
                  dropout_rate=0.3,
                  dropedge_rate=0.25,
                  disable_edge_updates=True,
                  use_sent_edges=True,
                  normalization_type='layer_norm',
                  aggregation_function='sum',
              ),
              training=dict(
                  loss_config=dict(
                      bgrl_loss_config=dict(
                          stop_gradient_for_supervised_loss=False,
                          bgrl_loss_scale=1.0,
                          symmetrize=True,
                          first_graph_corruption_config=dict(
                              feature_drop_prob=0.4,
                              edge_drop_prob=0.2,
                          ),
                          second_graph_corruption_config=dict(
                              feature_drop_prob=0.4,
                              edge_drop_prob=0.2,
                          ),
                      ),
                  ),
                  # GPU memory may require reducing the `256`s below to `48`.
                  dynamic_batch_size_config=dict(
                      n_node=256 if debug else 340 * 256,
                      n_edge=512 if debug else 720 * 256,
                      n_graph=4 if debug else 256,
                  ),
              ),
              eval=dict(
                  split='valid',
                  ema_annealing_schedule=dict(
                      use_schedule=True,
                      base_rate=0.999,
                      total_steps=config.get_ref('training_steps')),
                  dynamic_batch_size_config=dict(
                      n_node=256 if debug else 340 * 128,
                      n_edge=512 if debug else 720 * 128,
                      n_graph=4 if debug else 128,
                  ),
              ))))

  ## Training loop config.
  config.training_steps = 500000
  config.checkpoint_dir = '/tmp/checkpoint/mag/'
  config.train_checkpoint_all_hosts = False
  config.log_train_data_interval = 10
  config.log_tensors_interval = 10
  config.save_checkpoint_interval = 30
  config.best_model_eval_metric = 'accuracy'
  config.best_model_eval_metric_higher_is_better = True

  return config
