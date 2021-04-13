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
"""A module for registering already known functions for tagging patterns."""
import functools

from typing import Sequence, Tuple, TypeVar

import jax
from jax import core as jax_core
from jax import lax
from jax import lib as jax_lib
from jax.interpreters import batching as jax_batching
import jax.numpy as jnp

_T = TypeVar("_T")


class LossTag(jax_core.Primitive):
  """A tagging primitive specifically for losses."""
  multiple_results = True

  def __init__(self, cls, num_inputs: int, num_targets: int = 1):
    super().__init__(cls.__name__ + "_tag")
    self._cls = cls
    self._num_inputs = num_inputs
    self._num_targets = num_targets
    jax.xla.translations[self] = self.xla_translation
    jax.ad.primitive_jvps[self] = self.jvp
    # This line defines how does the tag behave under vmap. It is required for
    # any primitive that can be used inside a vmap. The reason why we want to
    # allow this is two fold - one to not break user code when the tags are not
    # used at all, and two - to be able to define a network with code for a
    # single example which is the vmap-ed for a batch.
    jax_batching.primitive_batchers[self] = self.batching

  @property
  def num_inputs(self) -> int:
    return self._num_inputs

  @property
  def num_targets(self) -> int:
    return self._num_targets

  def loss(self, *args, weight: float = 1.0, **kwargs):
    return self._cls(*args, weight=weight, **kwargs)

  def loss_evaluate(self, *args, weight: float = 1.0, **kwargs):
    return self.loss(*args, weight=weight, **kwargs).evaluate()

  def get_outputs(self, *args, weight: float, return_loss: bool, **kwargs):
    if len(args) < self.num_inputs:
      raise ValueError("Inputs to the tag are not enough.")
    if len(args) < self.num_inputs + self.num_targets:
      if len(args) != self.num_inputs:
        raise ValueError("Inputs to the tag are not quite enough.")
      if return_loss:
        raise ValueError("Can not have return_loss=True when there are no "
                         "targets.")
      return args
    if len(args) > self.num_inputs + self.num_targets:
      raise ValueError("Inputs to the tag are too many.")
    if return_loss:
      return self.loss(*args, weight=weight, **kwargs).evaluate()
    else:
      return args

  def impl(self, *args, weight: float, return_loss: bool, **kwargs):
    return self.get_outputs(*args, weight=weight, return_loss=return_loss)

  def abstract_eval(self, *args, weight: float, return_loss: bool, **kwargs):
    return self.get_outputs(*args, weight=weight, return_loss=return_loss)

  def xla_translation(
      self,
      c,
      *args,
      weight: float = 1.0,
      return_loss: bool = False,
      **kwargs,
  ):
    outputs = self.get_outputs(
        *args, weight=weight, return_loss=return_loss, **kwargs)
    if isinstance(outputs, tuple):
      return jax_lib.xla_client.ops.Tuple(c, outputs)
    return outputs

  def jvp(
      self,
      arg_values,
      arg_tangents,
      weight: float,
      return_loss: bool,
      **kwargs,
  ):
    if len(arg_values) != len(arg_tangents):
      raise ValueError("Values and tangents are not the same length.")
    primal_output = self.bind(
        *arg_values, weight=weight, return_loss=return_loss, **kwargs)
    if len(arg_values) == self.num_inputs:
      tangents_out = self.get_outputs(
          *arg_tangents, weight=weight, return_loss=return_loss, **kwargs)
    elif return_loss:
      tangents_out = jax.jvp(
          functools.partial(self.loss_evaluate, weight=weight, **kwargs),
          arg_tangents, arg_tangents)[1]
    else:
      tangents_out = arg_tangents
    return primal_output, tangents_out

  def batching(self, batched_args, batched_dims, **kwargs):
    return self.bind(*batched_args, **kwargs), batched_dims[0]


class LayerTag(jax_core.Primitive):
  """A tagging primitive that is used to mark/tag computation."""

  def __init__(self, name: str, num_inputs: int, num_outputs: int):
    super().__init__(name)
    if num_outputs > 1:
      raise NotImplementedError(
          f"Only single outputs are supported, got: num_outputs={num_outputs}")
    self._num_outputs = num_outputs
    self._num_inputs = num_inputs
    jax.xla.translations[self] = self.xla_translation
    jax.ad.deflinear(self, self.transpose)
    jax.ad.primitive_transposes[self] = self.transpose
    # This line defines how does the tag behave under vmap. It is required for
    # any primitive that can be used inside a vmap. The reason why we want to
    # allow this is two fold - one to not break user code when the tags are not
    # used at all, and two - to be able to define a network with code for a
    # single example which is the vmap-ed for a batch.
    jax_batching.primitive_batchers[self] = self.batching

  @property
  def num_outputs(self) -> int:
    return self._num_outputs

  @property
  def num_inputs(self) -> int:
    return self._num_inputs

  def split_all_inputs(
      self,
      all_inputs: Sequence[_T],
  ) -> Tuple[Sequence[_T], Sequence[_T], Sequence[_T]]:
    outputs = tuple(all_inputs[:self.num_outputs])
    inputs = tuple(all_inputs[self.num_outputs:self.num_outputs +
                              self.num_inputs])
    params = tuple(all_inputs[self.num_outputs + self.num_inputs:])
    return outputs, inputs, params

  def get_outputs(self, *operands: _T, **kwargs) -> _T:
    assert self.num_outputs == 1
    return operands[0]

  def xla_translation(self, c, *operands: _T, **kwargs) -> _T:
    return self.get_outputs(*operands, **kwargs)

  @staticmethod
  def transpose(cotangent, *operands, **kwargs):
    return (cotangent,) + (None,) * (len(operands) - 1)

  def impl(self, *operands, **kwargs):
    return self.get_outputs(*operands, **kwargs)

  def abstract_eval(self, *abstract_operands, **kwargs):
    return self.get_outputs(*abstract_operands, **kwargs)

  def batching(self, batched_operands, batched_dims, **kwargs):
    return self.bind(*batched_operands, **kwargs), batched_dims[0]


#   _____                      _
#  / ____|                    (_)
# | |  __  ___ _ __   ___ _ __ _  ___
# | | |_ |/ _ \ '_ \ / _ \ '__| |/ __|
# | |__| |  __/ | | |  __/ |  | | (__
#  \_____|\___|_| |_|\___|_|  |_|\___|
#
#

generic_tag = LayerTag(name="generic_tag", num_inputs=0, num_outputs=1)


def register_generic(parameter: _T) -> _T:
  return generic_tag.bind(parameter)


# _____
# |  __ \
# | |  | | ___ _ __  ___  ___
# | |  | |/ _ \ '_ \/ __|/ _ \
# | |__| |  __/ | | \__ \  __/
# |_____/ \___|_| |_|___/\___|
#

dense_tag = LayerTag(name="dense_tag", num_inputs=1, num_outputs=1)


def register_dense(y, x, w, b=None):
  if b is None:
    return dense_tag.bind(y, x, w)
  return dense_tag.bind(y, x, w, b)


def dense_func(x, params):
  """Example of a dense layer function."""
  w = params[0]
  y = jnp.matmul(x, w)
  if len(params) == 1:
    # No bias
    return y
  # Add bias
  return y + params[1]


def dense_tagging(jaxpr, inverse_map, values_map):
  """Correctly registers a dense layer pattern."""
  del inverse_map
  in_values = [values_map[v] for v in jaxpr.invars]
  out_values = [values_map[v] for v in jaxpr.outvars]
  return register_dense(out_values[0], *in_values)


#  ___  _____     _____                      _       _   _
# |__ \|  __ \   / ____|                    | |     | | (_)
#    ) | |  | | | |     ___  _ ____   _____ | |_   _| |_ _  ___  _ __
#   / /| |  | | | |    / _ \| '_ \ \ / / _ \| | | | | __| |/ _ \| "_ \
#  / /_| |__| | | |___| (_) | | | \ V / (_) | | |_| | |_| | (_) | | | |
# |____|_____/   \_____\___/|_| |_|\_/ \___/|_|\__,_|\__|_|\___/|_| |_|
#

conv2d_tag = LayerTag(name="conv2d_tag", num_inputs=1, num_outputs=1)


def register_conv2d(y, x, w, b=None, **kwargs):
  if b is None:
    return conv2d_tag.bind(y, x, w, **kwargs)
  return conv2d_tag.bind(y, x, w, b, **kwargs)


def conv2d_func(x, params):
  """Example of a conv2d layer function."""
  w = params[0]
  y = lax.conv_general_dilated(
      x,
      w,
      window_strides=(2, 2),
      padding="SAME",
      dimension_numbers=("NHWC", "HWIO", "NHWC"))
  if len(params) == 1:
    # No bias
    return y
  # Add bias
  return y + params[1][None, None, None]


def conv2d_tagging(jaxpr, inverse_map, values_map):
  """Correctly registers a conv2d layer pattern."""
  in_values = [values_map[v] for v in jaxpr.invars]
  out_values = [values_map[v] for v in jaxpr.outvars]
  keys = [k for k in inverse_map.keys() if isinstance(k, str)]
  keys = [k for k in keys if k.startswith("conv_general_dilated")]
  if len(keys) != 1:
    raise ValueError("Did not find any conv_general_dilated!")
  kwargs = inverse_map[keys[0]].params
  return register_conv2d(out_values[0], *in_values, **kwargs)


#   _____           _                        _    _____ _     _  __ _
#  / ____|         | |                      | |  / ____| |   (_)/ _| |
# | (___   ___ __ _| | ___    __ _ _ __   __| | | (___ | |__  _| |_| |_
#  \___ \ / __/ _` | |/ _ \  / _` | '_ \ / _` |  \___ \| '_ \| |  _| __|
#  ____) | (_| (_| | |  __/ | (_| | | | | (_| |  ____) | | | | | | | |_
# |_____/ \___\__,_|_|\___|  \__,_|_| |_|\__,_| |_____/|_| |_|_|_|  \__|
#

scale_and_shift_tag = LayerTag(
    name="scale_and_shift_tag", num_inputs=1, num_outputs=1)


def register_scale_and_shift(y, args, has_scale: bool, has_shift: bool):
  assert has_scale or has_shift
  x, args = args[0], args[1:]
  return scale_and_shift_tag.bind(
      y, x, *args, has_scale=has_scale, has_shift=has_shift)


def scale_and_shift_func(x, params, has_scale: bool, has_shift: bool):
  """Example of a scale and shift function."""
  if has_scale and has_shift:
    scale, shift = params
    return x * scale + shift
  elif has_scale:
    return x * params[0]
  elif has_shift:
    return x + params[0]
  else:
    raise ValueError()


def scale_and_shift_tagging(
    jaxpr,
    inverse_map,
    values_map,
    has_scale: bool,
    has_shift: bool,
):
  """Correctly registers a scale and shift layer pattern."""
  del inverse_map
  in_values = [values_map[v] for v in jaxpr.invars]
  out_values = [values_map[v] for v in jaxpr.outvars]
  return register_scale_and_shift(out_values[0], in_values, has_scale,
                                  has_shift)


def batch_norm_func(
    inputs: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
  """Example of batch norm as is defined in Haiku."""
  x, y = inputs
  scale, shift = params
  inv = scale * y
  return x * inv + shift


def batch_norm_tagging_func(
    jaxpr,
    inverse_map,
    values_map,
    has_scale: bool,
    has_shift: bool,
):
  """Correctly registers a batch norm layer pattern as is defined in Haiku."""
  del inverse_map
  in_values = [values_map[v] for v in jaxpr.invars]
  out_values = [values_map[v] for v in jaxpr.outvars]
  # The first two are both multipliers with the scale so we merge them
  in_values = [in_values[0] * in_values[1]] + in_values[2:]
  return register_scale_and_shift(out_values[0], in_values, has_scale,
                                  has_shift)
