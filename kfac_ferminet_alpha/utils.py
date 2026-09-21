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
"""Utilities related to multi-device operations."""
import collections
from typing import Any, Mapping, Optional, Sequence, Tuple, TypeVar, Union
import dataclasses
import jax
from jax import core
from jax import lax
import jax.numpy as jnp
from jax.scipy import linalg
import jax.tree_util as tree_util

T = TypeVar("T")


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
compute_mean = jax.pmap(lambda x: lax.pmean(x, "i"), axis_name="i")
compute_sum = jax.pmap(lambda x: lax.psum(x, "i"), axis_name="i")


def get_first(obj: T) -> T:
  return jax.tree_map(lambda x: x[0], obj)


def get_mean(obj: T) -> T:
  return get_first(compute_mean(obj))


def get_sum(obj: T) -> T:
  return get_first(compute_sum(obj))


broadcast_all_local_devices = jax.pmap(lambda x: x)


def replicate_all_local_devices(obj: T) -> T:
  n = jax.local_device_count()
  obj_stacked = jax.tree_map(lambda x: jnp.stack([x] * n, axis=0), obj)
  return broadcast_all_local_devices(obj_stacked)


def make_different_rng_key_on_all_devices(rng: jnp.ndarray) -> jnp.ndarray:
  rng = jax.random.fold_in(rng, jax.host_id())
  rng = jax.random.split(rng, jax.local_device_count())
  return broadcast_all_local_devices(rng)


p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


def scalar_mul(obj: T, scalar: Union[float, jnp.ndarray]) -> T:
  return jax.tree_map(lambda x: x * scalar, obj)


def scalar_div(obj: T, scalar: Union[float, jnp.ndarray]) -> T:
  return jax.tree_map(lambda x: x / scalar, obj)


def make_func_args(params, func_state, rng, batch, has_state: bool,
                   has_rng: bool):
  """Correctly puts all arguments to the function together."""
  func_args = (params,)
  if has_state:
    if func_state is None:
      raise ValueError("The `func_state` is None, but the argument `has_state` "
                       "is True.")
    func_args += (func_state,)
  if has_rng:
    if rng is None:
      raise ValueError("The `rng` is None, but the argument `has_rng` is True.")
    func_args += (rng,)
  func_args += (batch,)
  return func_args


def extract_func_outputs(
    raw_outputs: Any,
    has_aux: bool,
    has_state: bool,
) -> Tuple[jnp.ndarray, Any, Any]:
  """Given the function output returns separately the loss, func_state, aux."""
  if not has_aux and not has_state:
    return raw_outputs, None, None
  loss, other = raw_outputs
  if has_aux and has_state:
    func_state, aux = other
  elif has_aux:
    func_state, aux = None, other
  else:
    func_state, aux = other, None
  return loss, func_state, aux


def inner_product(obj1: T, obj2: T) -> jnp.ndarray:
  if jax.tree_structure(obj1) != jax.tree_structure(obj2):
    raise ValueError("The two structures are not identical.")
  elements_product = jax.tree_map(lambda x, y: jnp.sum(x * y), obj1, obj2)
  return sum(jax.tree_flatten(elements_product)[0])


def psd_inv_cholesky(matrix: jnp.ndarray, damping: jnp.ndarray) -> jnp.ndarray:
  assert matrix.ndim == 2
  identity = jnp.eye(matrix.shape[0])
  matrix = matrix + damping * identity
  return linalg.solve(matrix, identity, sym_pos=True)


def solve_maybe_small(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
  """Computes a^-1 b more efficiently for small matrices."""
  assert a.shape[-1] == a.shape[-2] == b.shape[-1]
  d = a.shape[-1]
  if d == 0:
    return a
  elif d == 1:
    return b / a[..., 0]
  elif d == 2:
    det = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    b_0 = a[..., 1, 1] * b[..., 0] - a[..., 0, 1] * b[..., 1]
    b_1 = a[..., 0, 0] * b[..., 1] - a[..., 1, 0] * b[..., 0]
    return jnp.stack([b_0, b_1], axis=-1) / det
  elif d == 3:
    raise NotImplementedError()
  return jnp.linalg.solve(a, b)


def pi_adjusted_inverse(
    factor_0: jnp.ndarray,
    factor_1: jnp.ndarray,
    damping: jnp.ndarray,
    pmap_axis_name: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Performs inversion with pi-adjusted damping."""
  # Compute the norms of each factor
  norm_0 = jnp.trace(factor_0)
  norm_1 = jnp.trace(factor_1)

  # We need to sync the norms here, because reduction can be non-deterministic.
  # They specifically are on GPUs by default for better performance.
  # Hence although factor_0 and factor_1 are synced, the trace operation above
  # can still produce different answers on different devices.
  norm_0, norm_1 = pmean_if_pmap((norm_0, norm_1), axis_name=pmap_axis_name)

  # Compute the overall scale
  scale = norm_0 * norm_1

  def regular_inverse(
      operand: Sequence[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    factor0, factor1, norm0, norm1, s, d = operand
    # Special cases with one or two scalar factors
    if factor0.size == 1 and factor1.size == 1:
      value = jnp.ones_like(factor0) / jnp.sqrt(s)
      return value, value
    if factor0.size == 1:
      factor1_normed = factor1 / norm1
      damping1 = d / norm1
      factor1_inv = psd_inv_cholesky(factor1_normed, damping1)
      return jnp.full((1, 1), s), factor1_inv
    if factor1.size == 1:
      factor0_normed = factor0 / norm0
      damping0 = d / norm0
      factor0_inv = psd_inv_cholesky(factor0_normed, damping0)
      return factor0_inv, jnp.full((1, 1), s)

    # Invert first factor
    factor0_normed = factor0 / norm0
    damping0 = jnp.sqrt(d * factor1.shape[0] / (s * factor0.shape[0]))
    factor0_inv = psd_inv_cholesky(factor0_normed, damping0) / jnp.sqrt(s)

    # Invert second factor
    factor1_normed = factor1 / norm1
    damping1 = jnp.sqrt(d * factor0.shape[0] / (s * factor1.shape[0]))
    factor1_inv = psd_inv_cholesky(factor1_normed, damping1) / jnp.sqrt(s)
    return factor0_inv, factor1_inv

  def zero_inverse(
      operand: Sequence[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return (jnp.eye(factor_0.shape[0]) / jnp.sqrt(operand[-1]),
            jnp.eye(factor_1.shape[0]) / jnp.sqrt(operand[-1]))

  # In the special case where for some reason one of the factors is zero, then
  # the correct inverse of `(0 kron A + lambda I)` is
  # `(I/sqrt(lambda) kron (I/sqrt(lambda)`. However, because one of the norms is
  # zero, then `pi` and `1/pi` would be 0 and infinity leading to NaN values.
  # Hence, we need to make this check explicitly.
  return lax.cond(
      jnp.greater(scale, 0.0),
      regular_inverse,
      zero_inverse,
      operand=(factor_0, factor_1, norm_0, norm_1, scale, damping))


def convert_value_and_grad_to_value_func(
    value_and_grad_func,
    has_aux: bool = False,
):
  """Converts a value_and_grad function to value_func only."""

  def value_func(*args, **kwargs):
    out, _ = value_and_grad_func(*args, **kwargs)
    if has_aux:
      return out[0]
    else:
      return out

  return value_func


def check_structure_shapes_and_dtype(obj1: T, obj2: T) -> None:
  """Verifies that the two objects have the same pytree structure."""
  assert jax.tree_structure(obj1) == jax.tree_structure(obj2)
  for v1, v2 in zip(jax.tree_flatten(obj1)[0], jax.tree_flatten(obj2)[0]):
    assert v1.shape == v2.shape
    assert v1.dtype == v2.dtype


def check_first_dim_is_batch_size(batch_size: int, *args: jnp.ndarray) -> None:
  for i, arg in enumerate(args):
    if arg.shape[0] != batch_size:
      raise ValueError(f"Expecting first dimension of arg[{i}] with shape "
                       f"{arg.shape} to be equal to the batch size "
                       f"{batch_size}.")


def py_tree_registered_dataclass(cls, *args, **kwargs):
  """Creates a new dataclass type and registers it as a pytree node."""
  dcls = dataclasses.dataclass(cls, *args, **kwargs)
  tree_util.register_pytree_node(
      dcls,
      lambda instance: (  # pylint: disable=g-long-lambda
          [getattr(instance, f.name)
           for f in dataclasses.fields(instance)], None),
      lambda _, instance_args: dcls(*instance_args))
  return dcls


class WeightedMovingAverage:
  """A wrapped class for a variable for which we keep exponential moving average."""

  def __init__(self, weight: jnp.ndarray, array: jnp.ndarray):
    self._weight = weight
    self._array = array

  @staticmethod
  def zero(shape: Sequence[int]) -> "WeightedMovingAverage":
    return WeightedMovingAverage(weight=jnp.zeros([]), array=jnp.zeros(shape))

  @property
  def weight(self) -> jnp.ndarray:
    return self._weight

  @property
  def value(self) -> jnp.ndarray:
    return self._array / self._weight

  @property
  def raw_value(self) -> jnp.ndarray:
    return self._array

  def update(self, value: jnp.ndarray, old_weight_multiplier: float,
             new_weight: float) -> None:
    self._weight = old_weight_multiplier * self._weight + new_weight
    self._array = old_weight_multiplier * self._array + new_weight * value

  def sync(self, pmap_axis_name: str) -> None:
    self._array = pmean_if_pmap(self._array, pmap_axis_name)

  def __str__(self) -> str:
    return (f"ExponentialMovingAverage(weight={self._weight}, "
            f"array={self._array})")

  def __repr__(self) -> str:
    return self.__str__()


tree_util.register_pytree_node(
    WeightedMovingAverage,
    lambda instance: ((instance.weight, instance.raw_value), None),
    lambda _, instance_args: WeightedMovingAverage(*instance_args),
)


class Stateful:
  """A class for stateful objects."""

  def __init__(self, stateful_fields_names: Optional[Sequence[str]] = ()):
    self.__stateful_fields_names = stateful_fields_names

  def _add_stateful_fields_names(self, value: Sequence[str]) -> None:
    self.__stateful_fields_names += tuple(value)

  def get_state(self) -> Mapping[str, Any]:
    """Returns the state of the object."""
    state = dict()
    for name in self.__stateful_fields_names:
      state[name] = Stateful._get_state_from_instance(getattr(self, name))
    return state

  def set_state(self, value):
    """Sets the state of the object with the provided value and returns the object."""
    assert isinstance(value, dict)
    for name in self.__stateful_fields_names:
      setattr(self, name,
              Stateful._set_state_to_instance(getattr(self, name), value[name]))
    return self

  def clear_state(self) -> None:
    """Clears the state of the object."""
    for name in self.__stateful_fields_names:
      setattr(self, name,
              Stateful._clear_state_from_instance(getattr(self, name)))

  def pop_state(self) -> Mapping[str, Any]:
    """Returns the current state of the object, while simultaneously clearing it."""
    state = self.get_state()
    self.clear_state()
    return state

  @staticmethod
  def _get_state_from_instance(obj):
    """Recursively gets the state of the object and returns it."""
    if isinstance(obj, Stateful):
      return obj.get_state()
    if isinstance(obj, list):
      return [Stateful._get_state_from_instance(i) for i in obj]
    if isinstance(obj, tuple):
      return tuple(Stateful._get_state_from_instance(i) for i in obj)
    if isinstance(obj, collections.OrderedDict):
      return collections.OrderedDict(
          (k, Stateful._get_state_from_instance(v)) for k, v in obj.items())
    if isinstance(obj, dict):
      return dict(
          (k, Stateful._get_state_from_instance(v)) for k, v in obj.items())
    return obj

  @staticmethod
  def _set_state_to_instance(obj, value):
    """Recursively sets the state of the object and returns it."""
    if isinstance(obj, Stateful):
      obj.set_state(value)
      return obj
    if isinstance(value, list):
      if obj is None:
        obj = [None] * len(value)
      return [
          Stateful._set_state_to_instance(obj_i, value_i)
          for obj_i, value_i in zip(obj, value)
      ]
    if isinstance(value, tuple):
      if obj is None:
        obj = [None] * len(value)
      return tuple(
          Stateful._set_state_to_instance(obj_i, value_i)
          for obj_i, value_i in zip(obj, value))
    if isinstance(value, collections.OrderedDict):
      if obj is None:
        obj = dict((k, None) for k in value)
      return collections.OrderedDict(
          (k, Stateful._set_state_to_instance(obj[k], value[k])) for k in obj)
    if isinstance(value, dict):
      obj = dict((k, None) for k in value)
      return dict(
          (k, Stateful._set_state_to_instance(obj[k], value[k])) for k in obj)
    return value

  @staticmethod
  def _clear_state_from_instance(obj):
    """Recursively clears the state of the object and returns it."""
    if isinstance(obj, Stateful):
      obj.clear_state()
      return obj
    if isinstance(obj, list):
      return [Stateful._clear_state_from_instance(obj_i) for obj_i in obj]
    if isinstance(obj, tuple):
      return tuple(Stateful._clear_state_from_instance(obj_i) for obj_i in obj)
    if isinstance(obj, collections.OrderedDict):
      return collections.OrderedDict(
          (k, Stateful._clear_state_from_instance(obj[k])) for k in obj)
    if isinstance(obj, dict):
      return dict((k, Stateful._clear_state_from_instance(obj[k])) for k in obj)
    return None

  @staticmethod
  def infer_class_state(class_type):
    """Infers a stateful class state attributes from class annotations."""
    if not issubclass(class_type, Stateful):
      raise ValueError(
          f"In order to annotate a class as stateful it must inherit "
          f"{Stateful!r}")

    class_type = dataclasses.dataclass(
        class_type, init=False, repr=False, eq=False)  # pytype: disable=wrong-keyword-args
    fields_names = tuple(field.name for field in dataclasses.fields(class_type))
    original_init = getattr(class_type, "__init__", None)
    if original_init is None:

      def injected_init(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)  # pylint: disable=bad-super-call
        Stateful._add_stateful_fields_names(self, fields_names)
        for field_name in fields_names:
          if getattr(self, field_name, None) is None:
            setattr(self, field_name, None)

      setattr(class_type, "__init__", injected_init)
    else:

      def injected_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        Stateful._add_stateful_fields_names(self, fields_names)
        for field_name in fields_names:
          if getattr(self, field_name, None) is None:
            setattr(self, field_name, None)

      setattr(class_type, "__init__", injected_init)
    return class_type


def compute_sq_norm_relative_abs_diff(obj, pmap_axis_name):
  sq_norm = inner_product(obj, obj)
  synced_sq_norm = psum_if_pmap(sq_norm, pmap_axis_name)
  synced_sq_norm = (synced_sq_norm - sq_norm) / (jax.device_count() - 1.0)
  sq_norm_abs_diff = jnp.abs(sq_norm - synced_sq_norm)
  return sq_norm_abs_diff / sq_norm


def product(iterable_object):
  x = 1
  for element in iterable_object:
    x *= element
  return x
