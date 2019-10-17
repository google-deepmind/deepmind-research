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
"""Training file to run most of the experiments in the paper.

The default parameters corresponding to the first set of experiments in Section
4.2.

For the expansion ablation, run with different ll_thresh values as in the paper.
Note that n_y_active represents the number of *active* components at the
start, and should be set to 1, while n_y represents the maximum number of
components allowed, and should be set sufficiently high (eg. n_y = 100).

For the MGR ablation, setting use_sup_replay = True switches to using SMGR,
and the gen_replay_type flag can switch between fixed and dynamic replay. The
generative snapshot period is set automatically in the train_curl.py file based
on these settings (ie. the data_period variable), so the 0.1T runs can be
reproduced by dividing this value by 10.
"""

from absl import app
from absl import flags

from curl import training

flags.DEFINE_enum('dataset', 'mnist', ['mnist', 'omniglot'], 'Dataset.')

FLAGS = flags.FLAGS


def main(unused_argv):
  training.run_training(
      dataset=FLAGS.dataset,
      output_type='bernoulli',
      n_y=30,
      n_y_active=1,
      training_data_type='sequential',
      n_concurrent_classes=1,
      lr_init=1e-3,
      lr_factor=1.,
      lr_schedule=[1],
      blend_classes=False,
      train_supervised=False,
      n_steps=100000,
      report_interval=10000,
      knn_values=[10],
      random_seed=1,
      encoder_kwargs={
          'encoder_type': 'multi',
          'n_enc': [1200, 600, 300, 150],
          'enc_strides': [1],
      },
      decoder_kwargs={
          'decoder_type': 'single',
          'n_dec': [500, 500],
          'dec_up_strides': None,
      },
      n_z=32,
      dynamic_expansion=True,
      ll_thresh=-200.0,
      classify_with_samples=False,
      gen_replay_type='fixed',
      use_supervised_replay=False,
      )

if __name__ == '__main__':
  app.run(main)
