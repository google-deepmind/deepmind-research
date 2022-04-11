# Copyright 2019 Deepmind Technologies Limited.
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
"""Configurations for IODINE."""
# pylint: disable=missing-docstring, unused-variable
import math


def clevr6():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "iodine/checkpoints/clevr6"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "iodine.modules.data.CLEVR",
      "batch_size": batch_size,
      "path": "multi_object_datasets/clevr_with_masks_train.tfrecords",
      "max_num_objects": 6,
  }

  model = {
      "constructor": "iodine.modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "iodine.modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "iodine.modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "iodine.modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "iodine.modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "iodine.modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "iodine.modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "iodine.modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "iodine.modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "iodine.modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          "constructor":
              "iodine.modules.factor_eval.FactorRegressor",
          "mapping": [
              ("color", 9, "categorical"),
              ("shape", 4, "categorical"),
              ("size", 3, "categorical"),
              ("position", 3, "scalar"),
          ],
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }


def multi_dsprites():
  n_z = 16  # number of latent dimensions
  num_components = 6  # number of components (K)
  num_iters = 5
  checkpoint_dir = "iodine/checkpoints/multi_dsprites"

  # For the paper we used 8 GPUs with a batch size of 16 each.
  # This means a total batch size of 128, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 16
  learn_rate = 0.0003 * math.sqrt(batch_size / 128)

  data = {
      "constructor":
          "iodine.modules.data.MultiDSprites",
      "batch_size":
          batch_size,
      "path":
          "multi_object_datasets/multi_dsprites_colored_on_grayscale.tfrecords",
      "dataset_variant":
          "colored_on_grayscale",
      "min_num_objs":
          3,
      "max_num_objs":
          3,
  }

  model = {
      "constructor": "iodine.modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "cos",
      "coord_freqs": 3,
      "decoder": {
          "constructor": "iodine.modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "iodine.modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [32, 32, 32, 32, None],
                  "kernel_shapes": [5],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "iodine.modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "iodine.modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [32, 32, 32],
                  "strides": [2],
                  "kernel_shapes": [5],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [128],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "iodine.modules.networks.LSTM",
              "hidden_sizes": [128],
          },
          "refinement_head": {
              "constructor": "iodine.modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "iodine.modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "iodine.modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "iodine.modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          "constructor":
              "iodine.modules.factor_eval.FactorRegressor",
          "mapping": [
              ("color", 3, "scalar"),
              ("shape", 4, "categorical"),
              ("scale", 1, "scalar"),
              ("x", 1, "scalar"),
              ("y", 1, "scalar"),
              ("orientation", 2, "angle"),
          ],
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }


def tetrominoes():
  n_z = 32  # number of latent dimensions
  num_components = 4  # number of components (K)
  num_iters = 5
  checkpoint_dir = "iodine/checkpoints/tetrominoes"

  # For the paper we used 8 GPUs with a batch size of 32 each.
  # This means a total batch size of 256, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 128
  learn_rate = 0.0003 * math.sqrt(batch_size / 256)

  data = {
      "constructor": "iodine.modules.data.Tetrominoes",
      "batch_size": batch_size,
      "path": "iodine/multi_object_datasets/tetrominoes_train.tfrecords",
  }

  model = {
      "constructor": "iodine.modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "cos",
      "coord_freqs": 3,
      "decoder": {
          "constructor": "iodine.modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "iodine.modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [32, 32, 32, 32, None],
                  "kernel_shapes": [5],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
              "coord_freqs": 3,
          },
      },
      "refinement_core": {
          "constructor": "iodine.modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "iodine.modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [32, 32, 32],
                  "strides": [2],
                  "kernel_shapes": [5],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [128],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "iodine.modules.networks.LSTM",
              "hidden_sizes": [],  # No recurrent layer used for this dataset
          },
          "refinement_head": {
              "constructor": "iodine.modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "iodine.modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "iodine.modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "iodine.modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          "constructor":
              "iodine.modules.factor_eval.FactorRegressor",
          "mapping": [
              ("position", 2, "scalar"),
              ("color", 3, "scalar"),
              ("shape", 20, "categorical"),
          ],
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }
