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
"""Utilities functions for Jax."""
import collections
import functools
from typing import Any, Callable, Dict, Mapping, Union

import distrax
import jax
from jax import core
from jax import lax
from jax import nn
import jax.numpy as jnp
from jax.tree_util import register_pytree_node
from jaxline import utils
import numpy as np

HaikuParams = Mapping[str, Mapping[str, jnp.ndarray]]
Params = Union[Mapping[str, jnp.ndarray], HaikuParams, jnp.ndarray]
_Activation = Callable[[jnp.ndarray], jnp.ndarray]

tf_leaky_relu = functools.partial(nn.leaky_relu, negative_slope=0.2)


def filter_only_scalar_stats(stats):
  return {k: v for k, v in stats.items() if v.size == 1}


def to_numpy(obj):
  return jax.tree_map(np.array, obj)


@jax.custom_gradient
def geco_lagrange_product(lagrange_multiplier, constraint_ema, constraint_t):
  """Modifies the gradients so that they work as described in GECO.

  The evaluation gives:
    lagrange * C_ema
  The gradient w.r.t lagrange:
    - g * C_t
  The gradient w.r.t constraint_ema:
    0.0
  The gradient w.r.t constraint_t:
    g * lagrange

  Note that if you pass the same value for `constraint_ema` and `constraint_t`
  this would only flip the gradient for the lagrange multiplier.

  Args:
    lagrange_multiplier: The lagrange multiplier
    constraint_ema: The moving average of the constraint
    constraint_t: The current constraint

  Returns:

  """
  def grad(gradient):
    return (- gradient * constraint_t,
            jnp.zeros_like(constraint_ema),
            gradient * lagrange_multiplier)
  return lagrange_multiplier * constraint_ema, grad


def bcast_if(x, t, n):
  return [x] * n if isinstance(x, t) else x


def stack_time_into_channels(
    images: jnp.ndarray,
    data_format: str
) -> jnp.ndarray:
  axis = data_format.index("C")
  list_of_time = [jnp.squeeze(v, axis=1) for v in
                  jnp.split(images, images.shape[1], axis=1)]
  return jnp.concatenate(list_of_time, axis)


def stack_device_dim_into_batch(obj):
  return jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), obj)


def nearest_neighbour_upsampling(x, scale, data_format="NHWC"):
  """Performs nearest-neighbour upsampling."""

  if data_format == "NCHW":
    b, c, h, w = x.shape
    x = jnp.reshape(x, [b, c, h, 1, w, 1])
    ones = jnp.ones([1, 1, 1, scale, 1, scale], dtype=x.dtype)
    return jnp.reshape(x * ones, [b, c, scale * h, scale * w])
  elif data_format == "NHWC":
    b, h, w, c = x.shape
    x = jnp.reshape(x, [b, h, 1, w, 1, c])
    ones = jnp.ones([1, 1, scale, 1, scale, 1], dtype=x.dtype)
    return jnp.reshape(x * ones, [b, scale * h, scale * w, c])
  else:
    raise ValueError(f"Unrecognized data_format={data_format}.")


def get_activation(arg: Union[_Activation, str]) -> _Activation:
  """Returns an activation from provided string."""
  if isinstance(arg, str):
    # Try fetch in order - [this module, jax.nn, jax.numpy]
    if arg in globals():
      return globals()[arg]
    if hasattr(nn, arg):
      return getattr(nn, arg)
    elif hasattr(jnp, arg):
      return getattr(jnp, arg)
    else:
      raise ValueError(f"Unrecognized activation with name {arg}.")
  if not callable(arg):
    raise ValueError(f"Expected a callable, but got {type(arg)}")
  return arg


def merge_first_dims(x: jnp.ndarray, num_dims_to_merge: int = 2) -> jnp.ndarray:
  return x.reshape((-1,) + x.shape[num_dims_to_merge:])


def extract_image(
    inputs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]]
) -> jnp.ndarray:
  """Extracts a tensor with key `image` or `x_image` if it is a dict, otherwise returns the inputs."""
  if isinstance(inputs, dict):
    if "image" in inputs:
      return inputs["image"]
    else:
      return inputs["x_image"]
  elif isinstance(inputs, jnp.ndarray):
    return inputs
  raise NotImplementedError(f"Not implemented of inputs of type"
                            f" {type(inputs)}.")


def extract_gt_state(inputs: Any) -> jnp.ndarray:
  if isinstance(inputs, dict):
    return inputs["x"]
  elif not isinstance(inputs, jnp.ndarray):
    raise NotImplementedError(f"Not implemented of inputs of type"
                              f" {type(inputs)}.")
  return inputs


def reshape_latents_conv_to_flat(conv_latents, axis_n_to_keep=1):
  q, p = jnp.split(conv_latents, 2, axis=-1)
  q = jax.tree_map(lambda x: x.reshape(x.shape[:axis_n_to_keep] + (-1,)), q)
  p = jax.tree_map(lambda x: x.reshape(x.shape[:axis_n_to_keep] + (-1,)), p)
  flat_latents = jnp.concatenate([q, p], axis=-1)

  return flat_latents


def triu_matrix_from_v(x, ndim):
  assert x.shape[-1] == (ndim * (ndim + 1)) // 2
  matrix = jnp.zeros(x.shape[:-1] + (ndim, ndim))
  idx = jnp.triu_indices(ndim)
  index_update = lambda x, idx, y: x.at[idx].set(y)
  for _ in range(x.ndim - 1):
    index_update = jax.vmap(index_update, in_axes=(0, None, 0))
  return index_update(matrix, idx, x)


def flatten_dict(d, parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
  items = []
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.MutableMapping):
      items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)


def convert_to_pytype(target, reference):
  """Makes target the same pytype as reference, by jax.tree_flatten."""
  _, pytree = jax.tree_flatten(reference)
  leaves, _ = jax.tree_flatten(target)
  return jax.tree_unflatten(pytree, leaves)


def func_if_not_scalar(func):
  """Makes a function that uses func only on non-scalar values."""
  @functools.wraps(func)
  def wrapped(array, axis=0):
    if array.ndim == 0:
      return array
    return func(array, axis=axis)
  return wrapped


mean_if_not_scalar = func_if_not_scalar(jnp.mean)


class MultiBatchAccumulator(object):
  """Class for abstracting statistics accumulation over multiple batches."""

  def __init__(self):
    self._obj = None
    self._obj_max = None
    self._obj_min = None
    self._num_samples = None

  def add(self, averaged_values, num_samples):
    """Adds an element to the moving average and the max."""
    if self._obj is None:
      self._obj_max = jax.tree_map(lambda y: y * 1.0, averaged_values)
      self._obj_min = jax.tree_map(lambda y: y * 1.0, averaged_values)
      self._obj = jax.tree_map(lambda y: y * num_samples, averaged_values)
      self._num_samples = num_samples
    else:
      self._obj_max = jax.tree_multimap(jnp.maximum, self._obj_max,
                                        averaged_values)
      self._obj_min = jax.tree_multimap(jnp.minimum, self._obj_min,
                                        averaged_values)
      self._obj = jax.tree_multimap(lambda x, y: x + y * num_samples, self._obj,
                                    averaged_values)
      self._num_samples += num_samples

  def value(self):
    return jax.tree_map(lambda x: x / self._num_samples, self._obj)

  def max(self):
    return jax.tree_map(float, self._obj_max)

  def min(self):
    return jax.tree_map(float, self._obj_min)

  def sum(self):
    return self._obj


register_pytree_node(
    distrax.Normal,
    lambda instance: ([instance.loc, instance.scale], None),
    lambda _, args: distrax.Normal(*args)
)


def inner_product(x: Any, y: Any) -> jnp.ndarray:
  products = jax.tree_multimap(lambda x_, y_: jnp.sum(x_ * y_), x, y)
  return sum(jax.tree_leaves(products))


get_first = utils.get_first
bcast_local_devices = utils.bcast_local_devices
py_prefetch = utils.py_prefetch
p_split = jax.pmap(lambda x, num: list(jax.random.split(x, num)),
                   static_broadcasted_argnums=1)


def wrap_if_pmap(p_func):
  def p_func_if_pmap(obj, axis_name):
    try:
      core.axis_frame(axis_name)
      return p_func(obj, axis_name)
    except NameError:
      return obj
  return p_func_if_pmap


pmean_if_pmap = wrap_if_pmap(lax.pmean)
psum_if_pmap = wrap_if_pmap(lax.psum)
