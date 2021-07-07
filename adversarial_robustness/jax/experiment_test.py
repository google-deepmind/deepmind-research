# Copyright 2021 Deepmind Technologies Limited.
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

"""Quick script to test that experiment can import and run."""

from absl import app
import jax
import jax.numpy as jnp
from jaxline import utils as jl_utils
from adversarial_robustness.jax import experiment


@jl_utils.disable_pmap_jit
def test_experiment(unused_argv):
  """Tests the main experiment."""
  config = experiment.get_config()
  exp_config = config.experiment_kwargs.config
  exp_config.dry_run = True
  exp_config.emulated_workers = 0
  exp_config.training.batch_size = 2
  exp_config.evaluation.batch_size = 2
  exp_config.model.kwargs.depth = 10
  exp_config.model.kwargs.width = 1

  xp = experiment.Experiment('train', exp_config, jax.random.PRNGKey(0))
  bcast = jax.pmap(lambda x: x)
  global_step = bcast(jnp.zeros(jax.local_device_count()))
  rng = bcast(jnp.stack([jax.random.PRNGKey(0)] * jax.local_device_count()))
  print('Taking a single experiment step for test purposes!')
  result = xp.step(global_step, rng)
  print(f'Step successfully taken, resulting metrics are {result}')


if __name__ == '__main__':
  app.run(test_experiment)
