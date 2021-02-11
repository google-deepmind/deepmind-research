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
"""Quick script to test that experiment can import and run."""
import jax
import jax.numpy as jnp
from nfnets import experiment
from nfnets import experiment_nfnets


def test_experiment():
  """Tests the main experiment."""
  config = experiment.get_config()
  exp_config = config.experiment_kwargs.config
  exp_config.train_batch_size = 2
  exp_config.eval_batch_size = 2
  exp_config.lr = 0.1
  exp_config.fake_data = True
  exp_config.model_kwargs.width = 2
  print(exp_config.model_kwargs)

  xp = experiment.Experiment('train', exp_config, jax.random.PRNGKey(0))
  bcast = jax.pmap(lambda x: x)
  global_step = bcast(jnp.zeros(jax.local_device_count()))
  rng = bcast(jnp.stack([jax.random.PRNGKey(0)] * jax.local_device_count()))
  print('Taking a single experiment step for test purposes!')
  result = xp.step(global_step, rng)
  print(f'Step successfully taken, resulting metrics are {result}')


def test_nfnet_experiment():
  """Tests the NFNet experiment."""
  config = experiment_nfnets.get_config()
  exp_config = config.experiment_kwargs.config
  exp_config.train_batch_size = 2
  exp_config.eval_batch_size = 2
  exp_config.lr = 0.1
  exp_config.fake_data = True
  exp_config.model_kwargs.width = 2
  print(exp_config.model_kwargs)

  xp = experiment_nfnets.Experiment('train', exp_config, jax.random.PRNGKey(0))
  bcast = jax.pmap(lambda x: x)
  global_step = bcast(jnp.zeros(jax.local_device_count()))
  rng = bcast(jnp.stack([jax.random.PRNGKey(0)] * jax.local_device_count()))
  print('Taking a single NFNet experiment step for test purposes!')
  result = xp.step(global_step, rng)
  print(f'NFNet Step successfully taken, resulting metrics are {result}')

test_experiment()
test_nfnet_experiment()
