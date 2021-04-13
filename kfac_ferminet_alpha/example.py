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
"""Example of running KFAC."""
from absl import app
from absl import flags
import jax
import jax.numpy as jnp

import numpy as np
import kfac_ferminet_alpha as kfac_ferminet_alpha
from kfac_ferminet_alpha import utils


TRAINING_STEPS = flags.DEFINE_integer(
    name="training_steps",
    default=100,
    help="Number of training steps to perform")
BATCH_SIZE = flags.DEFINE_integer(
    name="batch_size", default=128, help="Batch size")
LEARNING_RATE = flags.DEFINE_float(
    name="learning_rate", default=1e-3, help="Learning rate")
L2_REG = flags.DEFINE_float(
    name="l2_reg", default=1e-3, help="L2 regularization coefficient")
MOMENTUM = flags.DEFINE_float(
    name="momentum", default=0.8, help="Momentum coefficient")
DAMPING = flags.DEFINE_float(
    name="damping", default=1e-2, help="Damping coefficient")
MULTI_DEVICE = flags.DEFINE_bool(
    name="multi_device",
    default=False,
    help="Whether the computation should be replicated across multiple devices")
SEED = flags.DEFINE_integer(name="seed", default=12412321, help="JAX RNG seed")


def glorot_uniform(shape, key):
  dim_in = np.prod(shape[:-1])
  dim_out = shape[-1]
  c = jnp.sqrt(6 / (dim_in + dim_out))
  return jax.random.uniform(key, shape=shape, minval=-c, maxval=c)


def fully_connected_layer(params, x):
  w, b = params
  return jnp.matmul(x, w) + b[None]


def model_init(rng_key, batch, encoder_sizes=(1000, 500, 250, 30)):
  """Initialize the standard autoencoder."""
  x_size = batch.shape[-1]
  decoder_sizes = encoder_sizes[len(encoder_sizes) - 2::-1]
  sizes = (x_size,) + encoder_sizes + decoder_sizes + (x_size,)
  keys = jax.random.split(rng_key, len(sizes) - 1)
  params = []
  for rng_key, dim_in, dim_out in zip(keys, sizes, sizes[1:]):
    # Glorot uniform initialization
    w = glorot_uniform((dim_in, dim_out), rng_key)
    b = jnp.zeros([dim_out])
    params.append((w, b))
  return params, None


def model_loss(params, inputs, l2_reg):
  """Evaluate the standard autoencoder."""
  h = inputs.reshape([inputs.shape[0], -1])
  for i, layer_params in enumerate(params):
    h = fully_connected_layer(layer_params, h)
    # Last layer does not have a nonlinearity
    if i % 4 != 3:
      h = jnp.tanh(h)
  l2_value = 0.5 * sum(jnp.square(p).sum() for p in jax.tree_leaves(params))
  error = jax.nn.sigmoid(h) - inputs.reshape([inputs.shape[0], -1])
  mean_squared_error = jnp.mean(jnp.sum(error * error, axis=1), axis=0)
  regularized_loss = mean_squared_error + l2_reg * l2_value

  return regularized_loss, dict(mean_squared_error=mean_squared_error)


def random_data(multi_device, batch_shape, rng):
  if multi_device:
    shape = (multi_device,) + tuple(batch_shape)
  else:
    shape = tuple(batch_shape)
  while True:
    rng, key = jax.random.split(rng)
    yield jax.random.normal(key, shape)


def main(argv):
  del argv  # Unused.

  learning_rate = jnp.asarray([LEARNING_RATE.value])
  momentum = jnp.asarray([MOMENTUM.value])
  damping = jnp.asarray([DAMPING.value])

  # RNG keys
  global_step = jnp.zeros([])
  rng = jax.random.PRNGKey(SEED.value)
  params_key, opt_key, step_key, data_key = jax.random.split(rng, 4)
  dataset = random_data(MULTI_DEVICE.value, (BATCH_SIZE.value, 20), data_key)
  example_batch = next(dataset)

  if MULTI_DEVICE.value:
    global_step = utils.replicate_all_local_devices(global_step)
    learning_rate = utils.replicate_all_local_devices(learning_rate)
    momentum = utils.replicate_all_local_devices(momentum)
    damping = utils.replicate_all_local_devices(damping)
    params_key, opt_key = utils.replicate_all_local_devices(
        (params_key, opt_key))
    step_key = utils.make_different_rng_key_on_all_devices(step_key)
    split_key = jax.pmap(lambda x: tuple(jax.random.split(x)))
    jit_init_parameters_func = jax.pmap(model_init)
  else:
    split_key = jax.random.split
    jit_init_parameters_func = jax.jit(model_init)

  # Initialize or load parameters
  params, func_state = jit_init_parameters_func(params_key, example_batch)

  # Make optimizer
  optim = kfac_ferminet_alpha.Optimizer(
      value_and_grad_func=jax.value_and_grad(
          lambda p, x: model_loss(p, x, L2_REG.value), has_aux=True),
      l2_reg=L2_REG.value,
      value_func_has_aux=True,
      value_func_has_state=False,
      value_func_has_rng=False,
      learning_rate_schedule=None,
      momentum_schedule=None,
      damping_schedule=None,
      norm_constraint=1.0,
      num_burnin_steps=10,
  )

  # Initialize optimizer
  opt_state = optim.init(params, opt_key, example_batch, func_state)

  for t in range(TRAINING_STEPS.value):
    step_key, key_t = split_key(step_key)
    params, opt_state, stats = optim.step(
        params,
        opt_state,
        key_t,
        dataset,
        learning_rate=learning_rate,
        momentum=momentum,
        damping=damping)
    global_step = global_step + 1

    # Log any of the statistics
    print(f"iteration: {t}")
    print(f"mini-batch loss = {stats['loss']}")
    if "aux" in stats:
      for k, v in stats["aux"].items():
        print(f"{k} = {v}")
    print("----")


if __name__ == "__main__":
  app.run(main)
