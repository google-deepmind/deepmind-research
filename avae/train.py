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

"""VAE style training."""

from typing import Any, Callable, Iterator, Sequence, Mapping, Tuple, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from avae import checkpointer
from avae import types


def train(
    train_data_iterator: Iterator[types.LabelledData],
    test_data_iterator: Iterator[types.LabelledData],
    elbo_fun: hk.Transformed,
    learning_rate: float,
    checkpoint_dir: str,
    checkpoint_filename: str,
    checkpoint_every: int,
    test_every: int,
    iterations: int,
    rng_seed: int,
    test_functions: Optional[Sequence[Callable[[Mapping[str, jnp.ndarray]],
                                               Tuple[str, float]]]] = None,
    extra_checkpoint_info: Optional[Mapping[str, Any]] = None):
  """Train VAE with given data iterator and elbo definition.

  Args:
   train_data_iterator: Iterator of batched training data.
   test_data_iterator: Iterator of batched testing data.
   elbo_fun: Haiku transfomed function returning elbo.
   learning_rate: Learning rate to be used with optimizer.
   checkpoint_dir: Path of the checkpoint directory.
   checkpoint_filename: Filename of the checkpoint.
   checkpoint_every: Checkpoint every N iterations.
   test_every: Test and log results every N iterations.
   iterations: Number of training iterations to perform.
   rng_seed: Seed for random number generator.
   test_functions: Test function iterable, each function takes test data and
    outputs extra info to print at test and log time.
   extra_checkpoint_info: Extra info to put inside saved checkpoint.
  """
  rng_seq = hk.PRNGSequence(jax.random.PRNGKey(rng_seed))

  opt_init, opt_update = optax.chain(
      # Set the parameters of Adam. Note the learning_rate is not here.
      optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
      # Put a minus sign to *minimise* the loss.
      optax.scale(-learning_rate))

  @jax.jit
  def loss(params, key, data):
    elbo_outputs = elbo_fun.apply(params, key, data)
    return -jnp.mean(elbo_outputs.elbo)

  @jax.jit
  def loss_test(params, key, data):
    elbo_output = elbo_fun.apply(params, key, data)
    return (-jnp.mean(elbo_output.elbo), jnp.mean(elbo_output.data_fidelity),
            jnp.mean(elbo_output.kl))

  @jax.jit
  def update_step(params, key, data, opt_state):
    grads = jax.grad(loss, has_aux=False)(params, key, data)
    updates, opt_state = opt_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

  exp_checkpointer = checkpointer.Checkpointer(
      checkpoint_dir, checkpoint_filename)
  experiment_data = exp_checkpointer.load_checkpoint()

  if experiment_data is not None:
    start = experiment_data['step']
    params = experiment_data['experiment_state']
    opt_state = experiment_data['opt_state']
  else:
    start = 0
    params = elbo_fun.init(
        next(rng_seq), next(train_data_iterator).data)
    opt_state = opt_init(params)

  for step in range(start, iterations, 1):
    if step % test_every == 0:
      test_loss, ll, kl = loss_test(params, next(rng_seq),
                                    next(test_data_iterator).data)
      output_message = (f'Step {step} elbo {-test_loss:0.2f} LL {ll:0.2f} '
                        f'KL {kl:0.2f}')
      if test_functions:
        for test_function in test_functions:
          name, test_output = test_function(params)
          output_message += f' {name}: {test_output:0.2f}'
      print(output_message)

    params, opt_state = update_step(params, next(rng_seq),
                                    next(train_data_iterator).data, opt_state)

    if step % checkpoint_every == 0:
      exp_checkpointer.save_checkpoint(
          params, opt_state, step, extra_checkpoint_info)
