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
"""Module for all of the different curvature blocks."""
import abc
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Union
import jax
from jax import core
import jax.numpy as jnp

from kfac_ferminet_alpha import tag_graph_matcher as tgm
from kfac_ferminet_alpha import utils

_Arrays = Sequence[jnp.ndarray]
_BlockInfo = Mapping[str, Any]


class CurvatureBlock(utils.Stateful, abc.ABC):
  """Top level class."""

  def __init__(self, layer_tag_eq: tgm.jax_core.JaxprEqn):
    super(CurvatureBlock, self).__init__()
    self._layer_tag_eq = layer_tag_eq

  @property
  def layer_tag_primitive(self) -> tgm.tags.LayerTag:
    assert isinstance(self._layer_tag_eq.primitive, tgm.tags.LayerTag)
    return self._layer_tag_eq.primitive

  @property
  def outputs_shapes(self) -> Sequence[Sequence[int]]:
    output_vars = self.layer_tag_primitive.split_all_inputs(
        self._layer_tag_eq.invars)[0]
    return jax.tree_map(lambda x: x.aval.shape, output_vars)

  @property
  def inputs_shapes(self) -> Sequence[Sequence[int]]:
    input_vars = self.layer_tag_primitive.split_all_inputs(
        self._layer_tag_eq.invars)[1]
    return jax.tree_map(lambda x: x.aval.shape, input_vars)

  @property
  def params_shapes(self) -> Sequence[Sequence[int]]:
    params_vars = self.layer_tag_primitive.split_all_inputs(
        self._layer_tag_eq.invars)[2]
    return jax.tree_map(lambda x: x.aval.shape, params_vars)

  @abc.abstractmethod
  def init(self, rng: jnp.ndarray) -> MutableMapping[str, Any]:
    """This initializes/creates all of the arrays for the state of the block.

    Usually this would include the arrays used for storing the curvature
    approximation, as well as the arrays for storing any approximate
    inverses/powers of the curvature block.

    Args:
      rng: The Jax PRNG key to use if any of the state is supposed to be
      initialized randomly.
    Returns:
      A mutable mapping of the state.
    """

  @abc.abstractmethod
  def update_curvature_matrix_estimate(
      self,
      info: _BlockInfo,
      batch_size: int,
      ema_old: Union[float, jnp.ndarray],
      ema_new: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    pass

  @abc.abstractmethod
  def update_curvature_inverse_estimate(
      self,
      diagonal_weight: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    pass

  @abc.abstractmethod
  def multiply_matpower(
      self,
      vec: _Arrays,
      exp: Union[float, int],
      diagonal_weight: Union[float, jnp.ndarray]
  ) -> _Arrays:
    pass


CurvatureBlockCtor = Callable[[core.JaxprEqn], CurvatureBlock]


@utils.Stateful.infer_class_state
class NaiveDiagonal(CurvatureBlock):
  """The naively estimated diagonal block."""
  diagonal_factor: utils.WeightedMovingAverage

  def init(self, rng: jnp.ndarray) -> Dict[str, Any]:
    del rng
    return dict(
        diagonal_factor=utils.WeightedMovingAverage.zero(
            self.outputs_shapes[0])
    )

  def update_curvature_matrix_estimate(
      self,
      info: _BlockInfo,
      batch_size: int,
      ema_old: Union[float, jnp.ndarray],
      ema_new: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    dw, = info["outputs_tangent"]
    diagonal_update = dw * dw / batch_size
    self.diagonal_factor.update(diagonal_update, ema_old, ema_new)
    self.diagonal_factor.sync(pmap_axis_name)

  def update_curvature_inverse_estimate(
      self,
      diagonal_weight: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    pass

  def multiply_matpower(
      self,
      vec: _Arrays,
      exp: Union[float, int],
      diagonal_weight: Union[float, jnp.ndarray]
  ) -> _Arrays:
    w, = vec
    if exp == 1:
      return w * (self.diagonal_factor.value + diagonal_weight),
    elif exp == -1:
      return w / (self.diagonal_factor.value + diagonal_weight),
    else:
      raise NotImplementedError()


@utils.Stateful.infer_class_state
class TwoKroneckerFactored(CurvatureBlock, abc.ABC):
  """A factor that is the Kronecker product of two matrices."""
  inputs_factor: utils.WeightedMovingAverage
  inputs_factor_inverse: jnp.ndarray
  outputs_factor: utils.WeightedMovingAverage
  outputs_factor_inverse: jnp.ndarray
  extra_scale: Optional[Union[int, float, jnp.ndarray]]

  @property
  def has_bias(self) -> bool:
    return len(self._layer_tag_eq.invars) == 4

  @abc.abstractmethod
  def input_size(self) -> int:
    pass

  @abc.abstractmethod
  def output_size(self) -> int:
    pass

  def compute_extra_scale(self) -> Optional[Union[int, float, jnp.ndarray]]:
    return 1

  def init(self, rng: jnp.ndarray) -> Dict[str, Any]:
    # The extra scale is technically a constant, but in general it could be
    # useful for anyone examining the state to know it explicitly,
    # hence we actually keep it as part of the state.
    d_in = self.input_size()
    d_out = self.output_size()
    return dict(
        inputs_factor=utils.WeightedMovingAverage.zero([d_in, d_in]),
        inputs_factor_inverse=jnp.zeros([d_in, d_in]),
        outputs_factor=utils.WeightedMovingAverage.zero([d_out, d_out]),
        outputs_factor_inverse=jnp.zeros([d_out, d_out]),
        extra_scale=self.compute_extra_scale()
    )

  def update_curvature_inverse_estimate(
      self,
      diagonal_weight: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    self.inputs_factor.sync(pmap_axis_name)
    self.outputs_factor.sync(pmap_axis_name)

    # This computes the approximate inverse factor using the pi-adjusted
    # inversion from the original KFAC paper.
    # Note that the damping is divided by extra_scale since:
    # (s * A kron B + lambda I)^-1 = s^-1 (A kron B + s^-1 * lambda I)^-1
    # And the extra division by the scale is included in `multiply_matpower`.
    (self.inputs_factor_inverse,
     self.outputs_factor_inverse) = utils.pi_adjusted_inverse(
         factor_0=self.inputs_factor.value,
         factor_1=self.outputs_factor.value,
         damping=diagonal_weight / self.extra_scale,
         pmap_axis_name=pmap_axis_name)

  def multiply_matpower(
      self,
      vec: _Arrays,
      exp: Union[float, int],
      diagonal_weight: Union[float, jnp.ndarray]
  ) -> _Arrays:
    if self.has_bias:
      w, b = vec
      vec = jnp.concatenate([w.reshape([-1, w.shape[-1]]), b[None]], axis=0)
    else:
      w, = vec
      vec = w.reshape([-1, w.shape[-1]])
    if exp == 1:
      inputs_factor, outputs_factor = (self.inputs_factor.value,
                                       self.outputs_factor.value)
      scale = self.extra_scale
    elif exp == -1:
      inputs_factor, outputs_factor = (self.inputs_factor_inverse,
                                       self.outputs_factor_inverse)
      scale = 1.0 / self.extra_scale
      diagonal_weight = 0
    else:
      raise NotImplementedError()

    result = jnp.matmul(inputs_factor, vec)
    result = jnp.matmul(result, outputs_factor)
    result = result * scale + diagonal_weight * vec

    if self.has_bias:
      w_new, b_new = result[:-1], result[-1]
      return w_new.reshape(w.shape), b_new
    else:
      return result.reshape(w.shape),


class DenseTwoKroneckerFactored(TwoKroneckerFactored):
  """Factor for a standard dense layer."""

  def input_size(self) -> int:
    if self.has_bias:
      return self.params_shapes[0][0] + 1
    else:
      return self.params_shapes[0][0]

  def output_size(self) -> int:
    return self.params_shapes[0][1]

  def update_curvature_matrix_estimate(
      self,
      info: _BlockInfo,
      batch_size: int,
      ema_old: Union[float, jnp.ndarray],
      ema_new: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    del pmap_axis_name
    (x,), (dy,) = info["inputs"], info["outputs_tangent"]
    utils.check_first_dim_is_batch_size(batch_size, x, dy)

    if self.has_bias:
      x_one = jnp.ones_like(x[:, :1])
      x = jnp.concatenate([x, x_one], axis=1)
    input_stats = jnp.matmul(x.T, x) / batch_size
    output_stats = jnp.matmul(dy.T, dy) / batch_size
    self.inputs_factor.update(input_stats, ema_old, ema_new)
    self.outputs_factor.update(output_stats, ema_old, ema_new)


@utils.Stateful.infer_class_state
class ScaleAndShiftDiagonal(CurvatureBlock):
  """A scale and shift block with a diagonal approximation to the curvature."""
  scale_factor: Optional[utils.WeightedMovingAverage]
  shift_factor: Optional[utils.WeightedMovingAverage]

  @property
  def has_scale(self) -> bool:
    return self._layer_tag_eq.params["has_scale"]

  @property
  def has_shift(self) -> bool:
    return self._layer_tag_eq.params["has_shift"]

  def init(self, rng: jnp.ndarray) -> Dict[str, Any]:
    del rng
    if self.has_scale and self.has_shift:
      return dict(
          scale_factor=utils.WeightedMovingAverage.zero(
              self.params_shapes[0]
          ),
          shift_factor=utils.WeightedMovingAverage.zero(
              self.params_shapes[1]
          )
      )
    elif self.has_scale:
      return dict(
          scale_factor=utils.WeightedMovingAverage.zero(
              self.params_shapes[0]
          ),
          shift_factor=None
      )
    elif self.has_shift:
      return dict(
          scale_factor=None,
          shift_factor=utils.WeightedMovingAverage.zero(
              self.params_shapes[0]
          ),
      )
    else:
      raise ValueError("Neither `has_scale` nor `has_shift`.")

  def update_curvature_matrix_estimate(
      self,
      info: _BlockInfo,
      batch_size: int,
      ema_old: Union[float, jnp.ndarray],
      ema_new: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    (x,), (dy,) = info["inputs"], info["outputs_tangent"]
    utils.check_first_dim_is_batch_size(batch_size, x, dy)

    if self.has_scale:
      assert self.scale_factor is not None
      scale_shape = info["params"][0].shape
      full_scale_shape = (1,) * (len(x.shape) - len(scale_shape)) + scale_shape
      axis = [i for i, s in enumerate(full_scale_shape) if s == 1 and i != 0]
      d_scale = jnp.sum(x * dy, axis=axis)
      scale_diag_update = jnp.sum(d_scale * d_scale, axis=0) / batch_size
      self.scale_factor.update(scale_diag_update, ema_old, ema_new)  # pytype: disable=attribute-error  # trace-all-classes
      self.scale_factor.sync(pmap_axis_name)  # pytype: disable=attribute-error  # trace-all-classes

    if self.has_shift:
      assert self.shift_factor is not None
      shift_shape = info["params"][1].shape
      full_shift_shape = (1,) * (len(x.shape) - len(shift_shape)) + shift_shape
      axis = [i for i, s in enumerate(full_shift_shape) if s == 1 and i != 0]
      d_shift = jnp.sum(dy, axis=axis)
      shift_diag_update = jnp.sum(d_shift * d_shift, axis=0) / batch_size
      self.shift_factor.update(shift_diag_update, ema_old, ema_new)  # pytype: disable=attribute-error  # trace-all-classes
      self.shift_factor.sync(pmap_axis_name)  # pytype: disable=attribute-error  # trace-all-classes

  def update_curvature_inverse_estimate(
      self,
      diagonal_weight: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    pass

  def multiply_matpower(
      self,
      vec: _Arrays,
      exp: Union[float, int],
      diagonal_weight: Union[float, jnp.ndarray]
  ) -> _Arrays:
    if self.has_scale and self.has_shift:
      factors = (self.scale_factor.value, self.shift_factor.value)  # pytype: disable=attribute-error  # trace-all-classes
    elif self.has_scale:
      factors = (self.scale_factor.value,)  # pytype: disable=attribute-error  # trace-all-classes
    elif self.has_shift:
      factors = (self.shift_factor.value,)  # pytype: disable=attribute-error  # trace-all-classes
    else:
      raise ValueError("Neither `has_scale` nor `has_shift`.")
    factors = jax.tree_map(lambda x: x + diagonal_weight, factors)
    if exp == 1:
      return jax.tree_map(jnp.multiply, vec, factors)
    elif exp == -1:
      return jax.tree_map(jnp.divide, vec, factors)
    else:
      raise NotImplementedError()


@utils.Stateful.infer_class_state
class ScaleAndShiftFull(CurvatureBlock):
  """A scale and shift block with full approximation to the curvature."""
  factor: utils.WeightedMovingAverage
  inverse_factor: jnp.ndarray

  @property
  def _has_scale(self) -> bool:
    return self._layer_tag_eq.params["has_scale"]

  @property
  def _has_shift(self) -> bool:
    return self._layer_tag_eq.params["has_shift"]

  def init(self, rng: jnp.ndarray) -> Dict[str, Any]:
    del rng
    dims = sum(utils.product(shape) for shape in self.params_shapes)
    return dict(
        factor=utils.WeightedMovingAverage.zero([dims, dims]),
        inverse_factor=jnp.zeros([dims, dims])
    )

  def update_curvature_matrix_estimate(
      self,
      info: _BlockInfo,
      batch_size: int,
      ema_old: Union[float, jnp.ndarray],
      ema_new: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    del pmap_axis_name
    (x,), (dy,) = info["inputs"], info["outputs_tangent"]
    utils.check_first_dim_is_batch_size(batch_size, x, dy)

    grads = list()
    if self._has_scale:
      # Scale gradients
      scale_shape = info["params"][0].shape
      full_scale_shape = (1,) * (len(x.shape) - len(scale_shape)) + scale_shape
      axis = [i for i, s in enumerate(full_scale_shape) if s == 1 and i != 0]
      d_scale = jnp.sum(x * dy, axis=axis)
      d_scale = d_scale.reshape([batch_size, -1])
      grads.append(d_scale)

    if self._has_shift:
      # Shift gradients
      shift_shape = info["params"][1].shape
      full_shift_shape = (1,) * (len(x.shape) - len(shift_shape)) + shift_shape
      axis = [i for i, s in enumerate(full_shift_shape) if s == 1 and i != 0]
      d_shift = jnp.sum(dy, axis=axis)
      d_shift = d_shift.reshape([batch_size, -1])
      grads.append(d_shift)

    grads = jnp.concatenate(grads, axis=1)
    factor_update = jnp.matmul(grads.T, grads) / batch_size
    self.factor.update(factor_update, ema_old, ema_new)

  def update_curvature_inverse_estimate(
      self,
      diagonal_weight: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    self.factor.sync(pmap_axis_name)
    self.inverse_factor = utils.psd_inv_cholesky(self.factor.value,
                                                 diagonal_weight)

  def multiply_matpower(
      self,
      vec: _Arrays,
      exp: Union[float, int],
      diagonal_weight: Union[float, jnp.ndarray]
  ) -> _Arrays:
    # Remember the vector is a tuple of all parameters
    if self._has_scale and self._has_shift:
      flat_vec = jnp.concatenate([v.flatten() for v in vec])
    else:
      flat_vec = vec[0].flatten()

    if exp == 1:
      flat_result = (
          jnp.matmul(self.factor.value, flat_vec) + diagonal_weight * flat_vec)
    elif exp == -1:
      flat_result = jnp.matmul(self.inverse_factor, flat_vec)
    else:
      raise NotImplementedError()

    if self._has_scale and self._has_shift:
      scale_dims = int(vec[0].size)
      scale_result = flat_result[:scale_dims].reshape(vec[0].shape)
      shift_result = flat_result[scale_dims:].reshape(vec[1].shape)
      return scale_result, shift_result
    else:
      return flat_vec.reshape(vec[0].shape),


_default_tag_to_block: MutableMapping[str, CurvatureBlockCtor] = dict(
    dense_tag=DenseTwoKroneckerFactored,
    generic_tag=NaiveDiagonal,
    scale_and_shift_tag=ScaleAndShiftDiagonal,
)


def copy_default_tag_to_block() -> MutableMapping[str, CurvatureBlockCtor]:
  return dict(_default_tag_to_block)


def get_default_tag_to_block(tag_name: str) -> CurvatureBlockCtor:
  return _default_tag_to_block[tag_name]


def set_default_tag_to_block(
    tag_name: str,
    block_class: CurvatureBlockCtor,
) -> None:
  _default_tag_to_block[tag_name] = block_class
