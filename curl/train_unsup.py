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
"""Runs the unsupervised i.i.d benchmark experiments in the paper."""

from absl import app
from absl import flags

from curl import training

flags.DEFINE_enum('dataset', 'mnist', ['mnist', 'omniglot'], 'Dataset.')

FLAGS = flags.FLAGS


def main(unused_argv):
  if FLAGS.dataset == 'mnist':
    n_y = 25
    n_y_active = 1
    n_z = 50
  else:  # omniglot
    n_y = 100
    n_y_active = 1
    n_z = 100

  training.run_training(
      dataset=FLAGS.dataset,
      n_y=n_y,
      n_y_active=n_y_active,
      n_z=n_z,
      output_type='bernoulli',
      training_data_type='iid',
      n_concurrent_classes=1,
      lr_init=5e-4,
      lr_factor=1.,
      lr_schedule=[1],
      blend_classes=False,
      train_supervised=False,
      n_steps=100000,
      report_interval=10000,
      knn_values=[3],
      random_seed=1,
      encoder_kwargs={
          'encoder_type': 'multi',
          'n_enc': [500, 500],
          'enc_strides': [1],
      },
      decoder_kwargs={
          'decoder_type': 'single',
          'n_dec': [500],
          'dec_up_strides': None,
      },
      dynamic_expansion=True,
      ll_thresh=-200.0,
      classify_with_samples=True,
      gen_replay_type=None,
      use_supervised_replay=False,
      )

if __name__ == '__main__':
  app.run(main)
