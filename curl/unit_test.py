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
"""Tests for curl."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from curl import training


class TrainingTest(absltest.TestCase):

  def testRunTraining(self):

    training.run_training(
        dataset='mnist',
        output_type='bernoulli',
        n_y=10,
        n_y_active=1,
        training_data_type='sequential',
        n_concurrent_classes=1,
        lr_init=1e-3,
        lr_factor=1.,
        lr_schedule=[1],
        blend_classes=False,
        train_supervised=False,
        n_steps=1000,
        report_interval=1000,
        knn_values=[3],
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
  absltest.main()
