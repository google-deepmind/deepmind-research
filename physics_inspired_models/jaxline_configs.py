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
"""Module containing all of the configurations for various models."""
import copy
import os
from jaxline import base_config
import ml_collections as collections

_DATASETS_PATH_VAR_NAME = "DM_HAMILTONIAN_DYNAMICS_SUITE_DATASETS"


def get_config(arg_string):
  """Return config object for training."""
  args = arg_string.split(",")
  if len(args) != 3:
    raise ValueError("You must provide exactly three arguments separated by a "
                     "comma - model_config_name,sweep_index,dataset_name.")
  model_config_name, sweep_index, dataset_name = args
  sweep_index = int(sweep_index)

  config = base_config.get_base_config()
  config.random_seed = 123109801
  config.eval_modes = ("eval", "eval_metric")

  # Get the model config and the sweeps
  if model_config_name not in globals():
    raise ValueError(f"The config name {model_config_name} does not exist in "
                     f"jaxline_configs.py")
  config_and_sweep_fn = globals()[model_config_name]
  model_config, sweeps = config_and_sweep_fn()

  if not os.environ.get(_DATASETS_PATH_VAR_NAME, None):
    raise ValueError(f"You need to set the {_DATASETS_PATH_VAR_NAME}")
  dm_hamiltonian_suite_path = os.environ[_DATASETS_PATH_VAR_NAME]
  dataset_folder = os.path.join(dm_hamiltonian_suite_path, dataset_name)

  # Experiment config. Note that batch_size is per device.
  # In the experiments we run on 4 GPUs, so the effective batch size was 128.
  config.experiment_kwargs = collections.ConfigDict(
      dict(
          config=dict(
              dataset_folder=dataset_folder,
              model_kwargs=model_config,
              num_extrapolation_steps=60,
              drop_stats_containing=("neg_log_p_x", "l2_over_time", "neg_elbo"),
              optimizer=dict(
                  name="adam",
                  kwargs=dict(
                      learning_rate=1.5e-4,
                      b1=0.9,
                      b2=0.999,
                  )
              ),
              training=dict(
                  batch_size=32,
                  burnin_steps=5,
                  num_epochs=None,
                  lagging_vae=False
              ),
              evaluation=dict(
                  batch_size=64,
              ),
              evaluation_metric=dict(
                  batch_size=5,
                  batch_n=20,
                  num_eval_metric_steps=60,
                  max_poly_order=5,
                  max_jacobian_score=1000,
                  rsq_threshold=0.9,
                  sym_threshold=0.05,
                  evaluation_point_n=10,
                  weight_tolerance=1e-03,
                  max_iter=1000,
                  cv=2,
                  alpha_min_logspace=-4,
                  alpha_max_logspace=-0.5,
                  alpha_step_n=10,
                  calculate_fully_after_steps=40000,
              ),
              evaluation_metric_mlp=dict(
                  batch_size=64,
                  batch_n=10000,
                  datapoint_param_multiplier=1000,
                  num_eval_metric_steps=60,
                  evaluation_point_n=10,
                  evaluation_trajectory_n=50,
                  rsq_threshold=0.9,
                  sym_threshold=0.05,
                  ridge_lambda=0.01,
                  model=dict(
                      num_units=4,
                      num_layers=4,
                      activation="tanh",
                  ),
                  optimizer=dict(
                      name="adam",
                      kwargs=dict(
                          learning_rate=1.5e-3,
                      )
                  ),
              ),
              evaluation_vpt=dict(
                  batch_size=5,
                  batch_n=2,
                  vpt_threshold=0.025,
              )
          )
      )
  )

  # Training loop config.
  config.training_steps = int(500000)
  config.interval_type = "steps"
  config.log_tensors_interval = 50
  config.log_train_data_interval = 50
  config.log_all_train_data = False

  config.save_checkpoint_interval = 100
  config.checkpoint_dir = "/tmp/physics_inspired_models/"
  config.train_checkpoint_all_hosts = False
  config.eval_specific_checkpoint_dir = ""

  config.update_from_flattened_dict(sweeps[sweep_index])
  return config


config_prefix = "experiment_kwargs.config."
model_prefix = config_prefix + "model_kwargs."

default_encoder_kwargs = collections.ConfigDict(dict(
    conv_channels=64,
    num_blocks=3,
    blocks_depth=2,
    activation="leaky_relu",
))

default_decoder_kwargs = collections.ConfigDict(dict(
    conv_channels=64,
    num_blocks=3,
    blocks_depth=2,
    activation="leaky_relu",
))

default_latent_system_net_kwargs = collections.ConfigDict(dict(
    conv_channels=64,
    num_units=250,
    num_layers=5,
    activation="swish",
))


default_latent_system_kwargs = collections.ConfigDict(dict(
    # Physics model arguments
    input_space=collections.config_dict.placeholder(str),
    simulation_space=collections.config_dict.placeholder(str),
    potential_func_form="separable_net",
    kinetic_func_form=collections.config_dict.placeholder(str),
    hgn_kinetic_func_form="separable_net",
    lgn_kinetic_func_form="matrix_dep_quad",
    parametrize_mass_matrix=collections.config_dict.placeholder(bool),
    hgn_parametrize_mass_matrix=False,
    lgn_parametrize_mass_matrix=True,
    mass_eps=1.0,
    # ODE model arguments
    integrator_method=collections.config_dict.placeholder(str),
    # RGN model arguments
    residual=collections.config_dict.placeholder(bool),
    # General arguments
    net_kwargs=default_latent_system_net_kwargs
))

default_config_dict = collections.ConfigDict(dict(
    name=collections.config_dict.placeholder(str),
    latent_system_dim=32,
    latent_system_net_type="mlp",
    latent_system_kwargs=default_latent_system_kwargs,
    encoder_aggregation_type="linear_projection",
    decoder_de_aggregation_type=collections.config_dict.placeholder(str),
    encoder_kwargs=default_encoder_kwargs,
    decoder_kwargs=default_decoder_kwargs,
    has_latent_transform=False,
    num_inference_steps=5,
    num_target_steps=60,
    latent_training_type="forward",
    # Choices: overlap_by_one, no_overlap, include_inference
    training_data_split="overlap_by_one",
    objective_type="ELBO",
    elbo_beta_delay=0,
    elbo_beta_final=1.0,
    geco_kappa=0.001,
    geco_alpha=0.0,
    dt=0.125,
))

hgn_paper_encoder_kwargs = collections.ConfigDict(dict(
    conv_channels=[[32, 64], [64, 64], [64]],
    num_blocks=3,
    blocks_depth=2,
    activation="relu",
    kernel_shapes=[2, 4],
    padding=["VALID", "SAME"],
))

hgn_paper_decoder_kwargs = collections.ConfigDict(dict(
    conv_channels=64,
    num_blocks=3,
    blocks_depth=2,
    activation="tf_leaky_relu",
))

hgn_paper_latent_net_kwargs = collections.ConfigDict(dict(
    conv_channels=[32, 64, 64, 64],
    num_units=250,
    num_layers=5,
    activation="softplus",
    kernel_shapes=[3, 2, 2, 2, 2],
    strides=[1, 2, 1, 2, 1],
    padding=["SAME", "VALID", "SAME", "VALID", "SAME"]
))

hgn_paper_latent_system_kwargs = collections.ConfigDict(dict(
    potential_func_form="separable_net",
    kinetic_func_form="separable_net",
    parametrize_mass_matrix=False,
    net_kwargs=hgn_paper_latent_net_kwargs
))

hgn_paper_latent_transform_kwargs = collections.ConfigDict(dict(
    num_layers=5,
    conv_channels=64,
    num_units=64,
    activation="relu",
))

hgn_paper_config = copy.deepcopy(default_config_dict)
hgn_paper_config.training_data_split = "include_inference"
hgn_paper_config.latent_system_net_type = "conv"
hgn_paper_config.encoder_aggregation_type = (collections.config_dict.
                                             placeholder(str))
hgn_paper_config.decoder_de_aggregation_type = (collections.config_dict.
                                                placeholder(str))
hgn_paper_config.latent_system_kwargs = hgn_paper_latent_system_kwargs
hgn_paper_config.encoder_kwargs = hgn_paper_encoder_kwargs
hgn_paper_config.decoder_kwargs = hgn_paper_decoder_kwargs
hgn_paper_config.has_latent_transform = True
hgn_paper_config.latent_transform_kwargs = hgn_paper_latent_transform_kwargs
hgn_paper_config.num_inference_steps = 31
hgn_paper_config.num_target_steps = 0
hgn_paper_config.objective_type = "GECO"


forward_overlap_by_one = {
    model_prefix + "latent_training_type": "forward",
    model_prefix + "training_data_split": "overlap_by_one",
}

forward_backward_include_inference = {
    model_prefix + "latent_training_type": "forward_backward",
    model_prefix + "training_data_split": "include_inference",
}

latent_training_sweep = [
    forward_overlap_by_one,
    forward_backward_include_inference,
]


def sym_metric_hgn_plus_plus_sweep():
  """HGN++ experimental sweep for the SyMetric paper."""
  model_config = copy.deepcopy(default_config_dict)
  model_config.name = "HGN"
  sweeps = list()
  for elbo_beta_final in [0.001, 0.1, 1.0, 2.0]:
    sweeps.append({
        config_prefix + "optimizer.kwargs.learning_rate": 1.5e-4,
        model_prefix + "latent_training_type": "forward",
        model_prefix + "training_data_split": "overlap_by_one",
        model_prefix + "elbo_beta_final": elbo_beta_final,
    })
  for elbo_beta_final in [0.001, 0.1, 1.0, 2.0]:
    sweeps.append({
        config_prefix + "optimizer.kwargs.learning_rate": 1.5e-4,
        model_prefix + "latent_training_type": "forward_backward",
        model_prefix + "training_data_split": "include_inference",
        model_prefix + "elbo_beta_final": elbo_beta_final,
    })

  return model_config, sweeps


def sym_metric_hgn_sweep():
  """HGN experimental sweep for the SyMetric paper."""
  model_config = copy.deepcopy(hgn_paper_config)
  model_config.name = "HGN"
  return model_config, list(dict())


def benchmark_hgn_overlap_sweep():
  """HGN++ sweep for the benchmark paper."""
  model_config = copy.deepcopy(default_config_dict)
  model_config.name = "HGN"

  sweeps = list()
  for elbo_beta_final in [0.001, 0.1, 1.0, 2.0]:
    for train_dict in latent_training_sweep:
      sweeps.append({
          config_prefix + "optimizer.kwargs.learning_rate": 1.5e-4,
          model_prefix + "elbo_beta_final": elbo_beta_final,
      })
      sweeps[-1].update(train_dict)

  return model_config, sweeps


def benchmark_lgn_sweep():
  """LGN sweep for the benchmark paper."""
  model_config = copy.deepcopy(default_config_dict)
  model_config.name = "LGN"

  sweeps = list()
  for elbo_beta_final in [0.001, 0.1, 1.0, 2.0]:
    for train_dict in latent_training_sweep:
      sweeps.append({
          config_prefix + "optimizer.kwargs.learning_rate": 1.5e-4,
          model_prefix + "latent_system_kwargs.kinetic_func_form":
              "matrix_dep_pure_quad",
          model_prefix + "elbo_beta_final": elbo_beta_final,
      })
      sweeps[-1].update(train_dict)

  return model_config, sweeps


def benchmark_ode_sweep():
  """Neural ODE sweep for the benchmark paper."""
  model_config = copy.deepcopy(default_config_dict)
  model_config.name = "ODE"

  sweeps = list()
  for elbo_beta_final in [0.001, 0.1, 1.0, 2.0]:
    for integrator in ("adaptive", "rk2"):
      for train_dict in latent_training_sweep:
        sweeps.append({
            config_prefix + "optimizer.kwargs.learning_rate": 1.5e-4,
            model_prefix + "integrator_method": integrator,
            model_prefix + "elbo_beta_final": elbo_beta_final,
        })
        sweeps[-1].update(train_dict)

  return model_config, sweeps


def benchmark_rgn_sweep():
  """RGN sweep for the benchmark paper."""
  model_config = copy.deepcopy(default_config_dict)
  model_config.name = "RGN"

  sweeps = list()
  for elbo_beta_final in [0.001, 0.1, 1.0, 2.0]:
    for residual in (True, False):
      sweeps.append({
          config_prefix + "optimizer.kwargs.learning_rate": 1.5e-4,
          model_prefix + "latent_system_kwargs.residual": residual,
          model_prefix + "elbo_beta_final": elbo_beta_final,
      })

  return model_config, sweeps


def benchmark_ar_sweep():
  """AR sweep for the benchmark paper."""
  model_config = copy.deepcopy(default_config_dict)
  model_config.name = "AR"
  model_config.latent_dynamics_type = "vanilla"

  sweeps = list()
  for elbo_beta_final in [0.001, 0.1, 1.0, 2.0]:
    for ar_type in ("vanilla", "lstm", "gru"):
      sweeps.append({
          config_prefix + "optimizer.kwargs.learning_rate": 1.5e-4,
          model_prefix + "latent_dynamics_type": ar_type,
          model_prefix + "elbo_beta_final": elbo_beta_final,
      })

  return model_config, sweeps
