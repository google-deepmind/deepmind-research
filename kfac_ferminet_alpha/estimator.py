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
"""Defines the high-level Fisher estimator class."""
import collections
from typing import Any, Callable, Mapping, Optional, Sequence, Union, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jnr
import numpy as np

from kfac_ferminet_alpha import curvature_blocks
from kfac_ferminet_alpha import tracer
from kfac_ferminet_alpha import utils

_CurvatureBlock = curvature_blocks.CurvatureBlock
TagMapping = Mapping[str, curvature_blocks.CurvatureBlockCtor]
BlockVector = Sequence[jnp.ndarray]

_StructureT = TypeVar("_StructureT")
_OptionalStateT = TypeVar("_OptionalStateT", bound=Optional[Mapping[str, Any]])


@utils.Stateful.infer_class_state
class CurvatureEstimator(utils.Stateful):
  """Curvature estimator class supporting various curvature approximations."""
  blocks: "collections.OrderedDict[str, _CurvatureBlock]"
  damping: Optional[jnp.ndarray]

  def __init__(self,
               tagged_func: Callable[[Any], jnp.ndarray],
               func_args: Sequence[Any],
               l2_reg: Union[float, jnp.ndarray],
               estimation_mode: str = "fisher_gradients",
               params_index: int = 0,
               layer_tag_to_block_cls: Optional[TagMapping] = None):
    """Create a FisherEstimator object.

    Args:
      tagged_func: The function which evaluates the model, in which layer and
        loss tags has already been registered.
      func_args: Arguments to trace the function for layer and loss tags.
      l2_reg: Scalar. The L2 regularization coefficient, which represents
          the following regularization function: `coefficient/2 ||theta||^2`.
      estimation_mode: The type of curvature estimator to use. One of: *
        'fisher_gradients' - the basic estimation approach from the original
        K-FAC paper. (Default) * 'fisher_curvature_prop' - method which
        estimates the Fisher using self-products of random 1/-1 vectors times
        "half-factors" of the
              Fisher, as described here: https://arxiv.org/abs/1206.6464 *
                'fisher_exact' - is the obvious generalization of Curvature
                Propagation to compute the exact Fisher (modulo any additional
                diagonal or Kronecker approximations) by looping over one-hot
                vectors for each coordinate of the output instead of using 1/-1
                vectors. It is more expensive to compute than the other three
                options by a factor equal to the output dimension, roughly
                speaking. * 'fisher_empirical' - computes the 'empirical' Fisher
                information matrix (which uses the data's distribution for the
                targets, as opposed to the true Fisher which uses the model's
                distribution) and requires that each registered loss have
                specified targets. * 'ggn_curvature_prop' - Analogous to
                fisher_curvature_prop, but estimates the Generalized
                Gauss-Newton matrix (GGN). * 'ggn_exact'- Analogous to
                fisher_exact, but estimates the Generalized Gauss-Newton matrix
                (GGN).
      params_index: The index of the arguments accepted by `func` which
        correspond to parameters.
      layer_tag_to_block_cls: An optional dict mapping tags to specific classes
        of block approximations, which to override the default ones.
    """
    if estimation_mode not in ("fisher_gradients", "fisher_empirical",
                               "fisher_exact", "fisher_curvature_prop",
                               "ggn_exact", "ggn_curvature_prop"):
      raise ValueError(f"Unrecognised estimation_mode={estimation_mode}.")
    super().__init__()
    self.tagged_func = tagged_func
    self.l2_reg = l2_reg
    self.estimation_mode = estimation_mode
    self.params_index = params_index
    self.vjp = tracer.trace_estimator_vjp(self.tagged_func)

    # Figure out the mapping from layer
    self.layer_tag_to_block_cls = curvature_blocks.copy_default_tag_to_block()
    if layer_tag_to_block_cls is None:
      layer_tag_to_block_cls = dict()
    layer_tag_to_block_cls = dict(**layer_tag_to_block_cls)
    self.layer_tag_to_block_cls.update(layer_tag_to_block_cls)

    # Create the blocks
    self._in_tree = jax.tree_structure(func_args)
    self._jaxpr = jax.make_jaxpr(self.tagged_func)(*func_args).jaxpr
    self._layer_tags, self._loss_tags = tracer.extract_tags(self._jaxpr)
    self.blocks = collections.OrderedDict()
    counters = dict()
    for eqn in self._layer_tags:
      cls = self.layer_tag_to_block_cls[eqn.primitive.name]
      c = counters.get(cls.__name__, 0)
      self.blocks[cls.__name__ + "_" + str(c)] = cls(eqn)
      counters[cls.__name__] = c + 1

  @property
  def diagonal_weight(self) -> jnp.ndarray:
    return self.l2_reg + self.damping

  def vectors_to_blocks(
      self,
      parameter_structured_vector: Any,
  ) -> Sequence[BlockVector]:
    """Splits the parameters to values for the corresponding blocks."""
    in_vars = jax.tree_unflatten(self._in_tree, self._jaxpr.invars)
    params_vars = in_vars[self.params_index]
    params_vars_flat = jax.tree_flatten(params_vars)[0]
    params_values_flat = jax.tree_flatten(parameter_structured_vector)[0]
    assert len(params_vars_flat) == len(params_values_flat)
    params_dict = dict(zip(params_vars_flat, params_values_flat))
    per_block_vectors = []
    for eqn in self._layer_tags:
      if eqn.primitive.name == "generic_tag":
        block_vars = eqn.invars
      else:
        block_vars = eqn.primitive.split_all_inputs(eqn.invars)[2]  # pytype: disable=attribute-error  # trace-all-classes
      per_block_vectors.append(tuple(params_dict.pop(v) for v in block_vars))
    if params_dict:
      raise ValueError(f"From the parameters the following structure is not "
                       f"assigned to any block: {params_dict}. Most likely "
                       f"this part of the parameters is not part of the graph "
                       f"reaching the losses.")
    return tuple(per_block_vectors)

  def blocks_to_vectors(self, per_block_vectors: Sequence[BlockVector]) -> Any:
    """Reverses the function self.vectors_to_blocks."""
    in_vars = jax.tree_unflatten(self._in_tree, self._jaxpr.invars)
    params_vars = in_vars[self.params_index]
    assigned_dict = dict()
    for eqn, block_values in zip(self._layer_tags, per_block_vectors):
      if eqn.primitive.name == "generic_tag":
        block_params = eqn.invars
      else:
        block_params = eqn.primitive.split_all_inputs(eqn.invars)[2]  # pytype: disable=attribute-error  # trace-all-classes
      assigned_dict.update(zip(block_params, block_values))
    params_vars_flat, params_tree = jax.tree_flatten(params_vars)
    params_values_flat = [assigned_dict[v] for v in params_vars_flat]
    assert len(params_vars_flat) == len(params_values_flat)
    return jax.tree_unflatten(params_tree, params_values_flat)

  def init(
      self,
      rng: jnp.ndarray,
      init_damping: Optional[jnp.ndarray],
  ) -> Mapping[str, Any]:
    """Returns an initialized variables for the curvature approximations and the inverses.."""
    return dict(
        blocks=collections.OrderedDict(
            (name, block.init(block_rng))  #
            for (name, block), block_rng  #
            in zip(self.blocks.items(), jnr.split(rng, len(self.blocks)))),
        damping=init_damping)

  @property
  def mat_type(self) -> str:
    return self.estimation_mode.split("_")[0]

  def vec_block_apply(
      self,
      func: Callable[[_CurvatureBlock, BlockVector], BlockVector],
      parameter_structured_vector: Any,
  ) -> Any:
    """Executes func for each approximation block on vectors."""
    per_block_vectors = self.vectors_to_blocks(parameter_structured_vector)
    assert len(per_block_vectors) == len(self.blocks)
    results = jax.tree_map(func, tuple(self.blocks.values()),
                                per_block_vectors)
    parameter_structured_result = self.blocks_to_vectors(results)
    utils.check_structure_shapes_and_dtype(parameter_structured_vector,
                                           parameter_structured_result)
    return parameter_structured_result

  def multiply_inverse(self, parameter_structured_vector: Any) -> Any:
    """Multiplies the vectors by the corresponding (damped) inverses of the blocks.

    Args:
      parameter_structured_vector: Structure equivalent to the parameters of the
        model.

    Returns:
      A structured identical to `vectors` containing the product.
    """
    return self.multiply_matpower(parameter_structured_vector, -1)

  def multiply(self, parameter_structured_vector: Any) -> Any:
    """Multiplies the vectors by the corresponding (damped) blocks.

    Args:
      parameter_structured_vector: A vector in the same structure as the
        parameters of the model.

    Returns:
      A structured identical to `vectors` containing the product.
    """
    return self.multiply_matpower(parameter_structured_vector, 1)

  def multiply_matpower(
      self,
      parameter_structured_vector: _StructureT,
      exp: int,
  ) -> _StructureT:
    """Multiplies the vectors by the corresponding matrix powers of the blocks.

    Args:
      parameter_structured_vector: A vector in the same structure as the
        parameters of the model.
      exp: A float representing the power to raise the blocks by before
        multiplying it by the vector.

    Returns:
      A structured identical to `vectors` containing the product.
    """

    def func(block: _CurvatureBlock, vec: BlockVector) -> BlockVector:
      return block.multiply_matpower(vec, exp, self.diagonal_weight)

    return self.vec_block_apply(func, parameter_structured_vector)

  def update_curvature_matrix_estimate(
      self,
      ema_old: Union[float, jnp.ndarray],
      ema_new: Union[float, jnp.ndarray],
      batch_size: int,
      rng: jnp.ndarray,
      func_args: Sequence[Any],
      pmap_axis_name: str,
  ) -> None:
    """Updates the curvature estimate."""

    # Compute the losses and the VJP function from the function inputs
    losses, losses_vjp = self.vjp(func_args)

    # Helper function that updates the blocks given a vjp vector
    def _update_blocks(vjp_vec_, ema_old_, ema_new_):
      blocks_info_ = losses_vjp(vjp_vec_)
      for block_, block_info_ in zip(self.blocks.values(), blocks_info_):
        block_.update_curvature_matrix_estimate(
            info=block_info_,
            batch_size=batch_size,
            ema_old=ema_old_,
            ema_new=ema_new_,
            pmap_axis_name=pmap_axis_name)

    if self.estimation_mode == "fisher_gradients":
      keys = jnr.split(rng, len(losses)) if len(losses) > 1 else [rng]
      vjp_vec = tuple(
          loss.grad_of_evaluate_on_sample(key, coefficient_mode="sqrt")
          for loss, key in zip(losses, keys))
      _update_blocks(vjp_vec, ema_old, ema_new)

    elif self.estimation_mode in ("fisher_curvature_prop",
                                  "ggn_curvature_prop"):
      keys = jnr.split(rng, len(losses)) if len(losses) > 1 else [rng]
      vjp_vec = []
      for loss, key in zip(losses, keys):
        if self.estimation_mode == "fisher_curvature_prop":
          random_b = jnr.bernoulli(key, shape=loss.fisher_factor_inner_shape())
          vjp_vec.append(loss.multiply_fisher_factor(random_b * 2.0 - 1.0))
        else:
          random_b = jnr.bernoulli(key, shape=loss.ggn_factor_inner_shape())
          vjp_vec.append(loss.multiply_ggn_factor(random_b * 2.0 - 1.0))
      _update_blocks(tuple(vjp_vec), ema_old, ema_new)

    elif self.estimation_mode in ("fisher_exact", "ggn_exact"):
      # We use the following trick to simulate summation. The equation is:
      #   estimate = ema_old * estimate + ema_new * (sum_i estimate_index_i)
      #   weight = ema_old * weight + ema_new
      # Instead we update the estimate n times with the following updates:
      #   for k = 1
      #     estimate_k = ema_old * estimate + (ema_new/n) * (n*estimate_index_k)
      #     weight_k = ema_old * weight + (ema_new/n)
      #   for k > 1:
      #     estimate_k = 1.0 * estimate_k-1 + (ema_new/n) * (n*estimate_index_k)
      #     weight_k = 1.0 * weight_k-1 + (ema_new/n)
      # Which is mathematically equivalent to the original version.
      zero_tangents = jax.tree_map(jnp.zeros_like,
                                   list(loss.inputs for loss in losses))
      if self.estimation_mode == "fisher_exact":
        num_indices = [
            (l, int(np.prod(l.fisher_factor_inner_shape[1:]))) for l in losses
        ]
      else:
        num_indices = [
            (l, int(np.prod(l.ggn_factor_inner_shape()))) for l in losses
        ]
      total_num_indices = sum(n for _, n in num_indices)
      for i, (loss, loss_num_indices) in enumerate(num_indices):
        for index in range(loss_num_indices):
          vjp_vec = zero_tangents.copy()
          if self.estimation_mode == "fisher_exact":
            vjp_vec[i] = loss.multiply_fisher_factor_replicated_one_hot([index])
          else:
            vjp_vec[i] = loss.multiply_ggn_factor_replicated_one_hot([index])
          if isinstance(vjp_vec[i], jnp.ndarray):
            # In the special case of only one parameter, it still needs to be a
            # tuple for the tangents.
            vjp_vec[i] = (vjp_vec[i],)
          vjp_vec[i] = jax.tree_map(lambda x: x * total_num_indices, vjp_vec[i])
          _update_blocks(tuple(vjp_vec), ema_old, ema_new / total_num_indices)
          ema_old = 1.0

    elif self.estimation_mode == "fisher_empirical":
      raise NotImplementedError()
    else:
      raise ValueError(f"Unrecognised estimation_mode={self.estimation_mode}")

  def update_curvature_estimate_inverse(
      self,
      pmap_axis_name: str,
      state: _OptionalStateT,
  ) -> _OptionalStateT:
    if state is not None:
      old_state = self.get_state()
      self.set_state(state)
    for block in self.blocks.values():
      block.update_curvature_inverse_estimate(self.diagonal_weight,
                                              pmap_axis_name)
    if state is None:
      return None
    else:
      state = self.pop_state()
      self.set_state(old_state)
      return state
