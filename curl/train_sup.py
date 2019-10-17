################################################################################
# Copyright 2019 DeepMind Technologies Limited
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
################################################################################
"""Runs the supervised CL benchmark experiments in the paper."""

from absl import app
from absl import flags

from curl import training

flags.DEFINE_enum('dataset', 'mnist', ['mnist', 'omniglot'], 'Dataset.')

FLAGS = flags.FLAGS


def main(unused_argv):
  training.run_training(
      dataset=FLAGS.dataset,
      output_type='bernoulli',
      n_y=10,
      n_y_active=10,
      training_data_type='sequential',
      n_concurrent_classes=2,
      lr_init=1e-3,
      lr_factor=1.,
      lr_schedule=[1],
      train_supervised=True,
      blend_classes=False,
      n_steps=100000,
      report_interval=10000,
      knn_values=[],
      random_seed=1,
      encoder_kwargs={
          'encoder_type': 'multi',
          'n_enc': [400, 400],
          'enc_strides': [1],
      },
      decoder_kwargs={
          'decoder_type': 'single',
          'n_dec': [400, 400],
          'dec_up_strides': None,
      },
      n_z=32,
      dynamic_expansion=False,
      ll_thresh=-10000.0,
      classify_with_samples=False,
      gen_replay_type='fixed',
      use_supervised_replay=False,
      )

if __name__ == '__main__':
  app.run(main)
