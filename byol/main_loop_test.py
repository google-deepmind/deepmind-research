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
"""Tests for BYOL's main training loop."""

from absl import flags
from absl.testing import absltest
import tensorflow_datasets as tfds

from byol import byol_experiment
from byol import eval_experiment
from byol import main_loop
from byol.configs import byol as byol_config
from byol.configs import eval as eval_config


FLAGS = flags.FLAGS


class MainLoopTest(absltest.TestCase):

  def test_pretrain(self):
    config = byol_config.get_config(num_epochs=40, batch_size=4)
    temp_dir = self.create_tempdir().full_path

    # Override some config fields to make test lighter.
    config['network_config']['encoder_class'] = 'TinyResNet'
    config['network_config']['projector_hidden_size'] = 256
    config['network_config']['predictor_hidden_size'] = 256
    config['checkpointing_config']['checkpoint_dir'] = temp_dir
    config['evaluation_config']['batch_size'] = 16
    config['max_steps'] = 16

    with tfds.testing.mock_data(num_examples=64):
      experiment_class = byol_experiment.ByolExperiment
      main_loop.train_loop(experiment_class, config)
      main_loop.eval_loop(experiment_class, config)

  def test_linear_eval(self):
    config = eval_config.get_config(checkpoint_to_evaluate=None, batch_size=4)
    temp_dir = self.create_tempdir().full_path

    # Override some config fields to make test lighter.
    config['network_config']['encoder_class'] = 'TinyResNet'
    config['allow_train_from_scratch'] = True
    config['checkpointing_config']['checkpoint_dir'] = temp_dir
    config['evaluation_config']['batch_size'] = 16
    config['max_steps'] = 16

    with tfds.testing.mock_data(num_examples=64):
      experiment_class = eval_experiment.EvalExperiment
      main_loop.train_loop(experiment_class, config)
      main_loop.eval_loop(experiment_class, config)

  def test_pipeline(self):
    b_config = byol_config.get_config(num_epochs=40, batch_size=4)
    temp_dir = self.create_tempdir().full_path

    # Override some config fields to make test lighter.
    b_config['network_config']['encoder_class'] = 'TinyResNet'
    b_config['network_config']['projector_hidden_size'] = 256
    b_config['network_config']['predictor_hidden_size'] = 256
    b_config['checkpointing_config']['checkpoint_dir'] = temp_dir
    b_config['evaluation_config']['batch_size'] = 16
    b_config['max_steps'] = 16

    with tfds.testing.mock_data(num_examples=64):
      main_loop.train_loop(byol_experiment.ByolExperiment, b_config)

    e_config = eval_config.get_config(
        checkpoint_to_evaluate=f'{temp_dir}/pretrain.pkl',
        batch_size=4)

    # Override some config fields to make test lighter.
    e_config['network_config']['encoder_class'] = 'TinyResNet'
    e_config['allow_train_from_scratch'] = True
    e_config['checkpointing_config']['checkpoint_dir'] = temp_dir
    e_config['evaluation_config']['batch_size'] = 16
    e_config['max_steps'] = 16

    with tfds.testing.mock_data(num_examples=64):
      main_loop.train_loop(eval_experiment.EvalExperiment, e_config)


if __name__ == '__main__':
  absltest.main()
