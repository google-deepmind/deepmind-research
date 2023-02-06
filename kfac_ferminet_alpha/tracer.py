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
"""Module for the Jax tracer functionality for tags."""
import functools
from typing import Any, Callable, Sequence, Tuple

import jax
from jax import core
from jax import util as jax_util
import jax.numpy as jnp

from kfac_ferminet_alpha import layers_and_loss_tags as tags
from kfac_ferminet_alpha import tag_graph_matcher as tgm
from kfac_ferminet_alpha import utils

_Function = Callable[[Any], Any]
_Loss = tags.LossTag


def extract_tags(
    jaxpr: core.Jaxpr
) -> Tuple[Sequence[core.JaxprEqn], Sequence[core.JaxprEqn]]:
  """Extracts all of the tag equations."""
  # Loop through equations and evaluate primitives using `bind`
  layer_tags = []
  loss_tags = []
  for eqn in jaxpr.eqns:
    if isinstance(eqn.primitive, tags.LossTag):
      loss_tags.append(eqn)
    elif isinstance(eqn.primitive, tags.LayerTag):
      layer_tags.append(eqn)
  return tuple(layer_tags), tuple(loss_tags)


def construct_compute_losses_inputs(
    jaxpr: core.Jaxpr,
    consts: Tuple[Any],
    num_losses: int,
    primals: Any,
    params_index: int) -> Callable[[Any], Sequence[Sequence[jnp.ndarray]]]:
  """Constructs a function that computes all of the inputs to all losses."""
  primals_ = list(primals)

  def forward_compute_losses(
      params_primals: Any,
  ) -> Sequence[Sequence[jnp.ndarray]]:
    primals_[params_index] = params_primals
    flat_args = jax.tree_flatten(primals_)[0]
    # Mapping from variable -> value
    env = dict()
    read = functools.partial(tgm.read_env, env)
    write = functools.partial(tgm.write_env, env)

    # Bind args and consts to environment
    jax_util.safe_map(write, jaxpr.invars, flat_args)
    jax_util.safe_map(write, jaxpr.constvars, consts)

    # Loop through equations and evaluate primitives using `bind`
    losses_so_far = 0
    loss_tags = []
    for eqn in jaxpr.eqns:
      tgm.evaluate_eqn(eqn, jax_util.safe_map(read, eqn.invars), write)
      if isinstance(eqn.primitive, tags.LossTag):
        loss_tags.append(eqn)
        losses_so_far += 1
      if num_losses is not None and losses_so_far == num_losses:
        break
    return tuple(tuple(read(v) for v in tag.invars) for tag in loss_tags)
    # return tuple(jax_util.safe_map(read, tag.invars) for tag in loss_tags)
  return forward_compute_losses


# We know when `.primitive` will be either a `LossTag` or a `LayerTag`, however
# pytype cannot infer its subclass, so we need to unbox it.


def _unbox_loss_tag(jaxpr_eqn: core.JaxprEqn) -> tags.LossTag:
  assert isinstance(jaxpr_eqn.primitive, tags.LossTag)
  return jaxpr_eqn.primitive


def _unbox_layer_tag(jaxpr_eqn: core.JaxprEqn) -> tags.LayerTag:
  assert isinstance(jaxpr_eqn.primitive, tags.LayerTag)
  return jaxpr_eqn.primitive


def trace_losses_matrix_vector_vjp(tagged_func: _Function,
                                   params_index: int = 0):
  """Returns the Jacobian-transposed vector product (backward mode) function in equivalent form to jax.vjp."""
  def vjp(*primals):
    typed_jaxpr = jax.make_jaxpr(tagged_func)(*primals)
    jaxpr, consts = typed_jaxpr.jaxpr, typed_jaxpr.literals
    _, loss_jaxpr_eqns = extract_tags(jaxpr)
    n = len(loss_jaxpr_eqns)
    losses_func = construct_compute_losses_inputs(
        jaxpr, consts, n, primals, params_index)
    losses_inputs, full_vjp_func = jax.vjp(losses_func, primals[params_index])
    losses = []
    for jaxpr_eqn, inputs in zip(loss_jaxpr_eqns, losses_inputs):
      loss_tag = _unbox_loss_tag(jaxpr_eqn)
      losses.append(loss_tag.loss(*inputs, weight=jaxpr_eqn.params["weight"]))
    losses = tuple(losses)

    def vjp_func(tangents):
      flat_tangents = jax.tree_flatten(tangents)[0]
      loss_invars = []
      loss_targets = []
      for jaxpr_eqn, inputs in zip(loss_jaxpr_eqns, losses_inputs):
        num_inputs = _unbox_loss_tag(jaxpr_eqn).num_inputs
        loss_invars.append(tuple(jaxpr_eqn.invars[:num_inputs]))
        loss_targets.append(inputs[num_inputs:])
      treedef = jax.tree_structure(loss_invars)
      tangents = jax.tree_unflatten(treedef, flat_tangents)
      # Since the losses could also take and targets as inputs and we don't want
      # this function to computes vjp w.r.t to those (e.g. the user should not
      # be providing tangent vectors for the targets, only for inputs) we have
      # to manually fill in these "extra" tangents with zeros.
      targets_tangents = jax.tree_map(jnp.zeros_like, loss_targets)
      tangents = tuple(ti + tti for ti, tti in zip(tangents, targets_tangents))
      input_tangents = full_vjp_func(tangents)[0]
      return input_tangents,
    return losses, vjp_func
  return vjp


def trace_losses_matrix_vector_jvp(
    tagged_func: _Function,
    params_index: int = 0):
  """Returns the Jacobian vector product (forward mode) function in equivalent form to jax.jvp."""
  def jvp(primals, params_tangents):
    typed_jaxpr = jax.make_jaxpr(tagged_func)(*primals)
    jaxpr, consts = typed_jaxpr.jaxpr, typed_jaxpr.literals
    _, loss_tags = extract_tags(jaxpr)
    n = len(loss_tags)
    losses_func = construct_compute_losses_inputs(jaxpr, consts, n,
                                                  primals, params_index)
    primals = (primals[params_index],)
    tangents = (params_tangents,)
    (primals_out, tangents_out) = jax.jvp(losses_func, primals, tangents)
    tangents_out = tuple(tuple(t[:tag.primitive.num_inputs])
                         for t, tag in zip(tangents_out, loss_tags))
    losses = tuple(tag.primitive.loss(*inputs, weight=tag.params["weight"])
                   for tag, inputs in zip(loss_tags, primals_out))
    return losses, tangents_out
  return jvp


def trace_losses_matrix_vector_hvp(tagged_func, params_index=0):
  """Returns the Hessian vector product function of **the tagged losses**, rather than the output value of `tagged_func`."""
  # The function uses backward-over-forward mode.

  def hvp(primals, params_tangents):
    typed_jaxpr = jax.make_jaxpr(tagged_func)(*primals)
    jaxpr, consts = typed_jaxpr.jaxpr, typed_jaxpr.literals
    _, loss_tags = extract_tags(jaxpr)
    n = len(loss_tags)
    losses_func = construct_compute_losses_inputs(
        jaxpr, consts, n, primals, params_index)

    def losses_sum(param_primals):
      loss_inputs = losses_func(param_primals)
      losses = [
          _unbox_loss_tag(jaxpr_eqn).loss(
              *inputs, weight=jaxpr_eqn.params["weight"])
          for jaxpr_eqn, inputs in zip(loss_tags, loss_inputs)
      ]
      # This computes the sum of losses evaluated. Makes it easier as we can
      # now use jax.grad rather than jax.vjp for taking derivatives.
      return sum(jnp.sum(loss.evaluate(None)) for loss in losses)

    def grads_times_tangents(params_primals):
      grads = jax.grad(losses_sum)(params_primals)
      return utils.inner_product(grads, params_tangents)

    return jax.grad(grads_times_tangents)(primals[params_index])
  return hvp


def trace_estimator_vjp(tagged_func: _Function) -> _Function:
  """Creates the function needed for an estimator of curvature matrices.

  Args:
    tagged_func: An function that has been annotated with tags both for layers
      and losses.

  Returns:
    A function with the same signatures as `tagged_func`, which when provided
    with inputs returns two things:
    1. The instances of all losses objected that are tagged.
    2. A second function, which when provide with tangent vectors for each
      of the loss instances' parameters, returns for every tagged layer a
      dictionary containing the following elements:
        inputs - The primal values of the inputs to the layer.
        outputs - The primal values of the outputs to the layer.
        params - The primal values of the layer.
        inputs_tangent - The tangent value of layer, given the provided
          tangents of the losses.
        inputs_tangent - The tangent value of layer, given the provided
          tangents of the losses.
        inputs_tangent - The tangent value of layer, given the provided
          tangents of the losses.
  """
  def full_vjp_func(func_args):
    # Trace the tagged function
    typed_jaxpr = jax.make_jaxpr(tagged_func)(*func_args)
    jaxpr, consts = typed_jaxpr.jaxpr, typed_jaxpr.literals
    layer_tags, loss_tags = extract_tags(jaxpr)

    layer_vars_flat = jax.tree_flatten([tag.invars for tag in layer_tags])[0]
    layer_input_vars = tuple(set(layer_vars_flat))

    def forward():
      own_func_args = func_args
      # Mapping from variable -> value
      env = dict()
      read = functools.partial(tgm.read_env, env)
      write = functools.partial(tgm.write_env, env)

      # Bind args and consts to environment
      jax_util.safe_map(write, jaxpr.invars, jax.tree_flatten(own_func_args)[0])
      jax_util.safe_map(write, jaxpr.constvars, consts)

      # Loop through equations and evaluate primitives using `bind`
      num_losses_passed = 0
      for eqn in jaxpr.eqns:
        tgm.evaluate_eqn(eqn, jax_util.safe_map(read, eqn.invars), write)
        if isinstance(eqn.primitive, tags.LossTag):
          num_losses_passed += 1
          if num_losses_passed == len(loss_tags):
            break
      if num_losses_passed != len(loss_tags):
        raise ValueError("This should be unreachable.")

      return jax_util.safe_map(read, layer_input_vars)

    def forward_aux(aux):
      own_func_args = func_args
      # Mapping from variable -> value
      env = dict()
      read = functools.partial(tgm.read_env, env)
      def write(var, val):
        if not isinstance(var, jax.core.Literal):
          val = val + aux[var] if var in aux else val
        env[var] = val

      # Bind args and consts to environment
      jax_util.safe_map(write, jaxpr.invars, jax.tree_flatten(own_func_args)[0])
      jax_util.safe_map(write, jaxpr.constvars, consts)

      # Loop through equations and evaluate primitives using `bind`
      num_losses_passed = 0
      losses_inputs_values = []
      losses_kwargs_values = []
      for eqn in jaxpr.eqns:
        input_values = jax_util.safe_map(read, eqn.invars)
        tgm.evaluate_eqn(eqn, input_values, write)
        if isinstance(eqn.primitive, tags.LossTag):
          loss = eqn.primitive.loss(*input_values, weight=eqn.params["weight"])
          losses_inputs_values.append(loss.inputs)
          losses_kwargs_values.append(dict(
              targets=loss.targets,
              weight=eqn.params["weight"]
          ))
          num_losses_passed += 1
          if num_losses_passed == len(loss_tags):
            break
      if num_losses_passed != len(loss_tags):
        raise ValueError("This should be unreachable.")
      # Read the inputs to the loss functions, but also return the target values
      return tuple(losses_inputs_values), tuple(losses_kwargs_values)

    layer_input_values = forward()
    primals_dict = dict(zip(layer_input_vars, layer_input_values))
    primals_dict.update(zip(jaxpr.invars, jax.tree_flatten(func_args)[0]))
    aux_values = jax.tree_map(jnp.zeros_like, layer_input_values)
    aux_dict = dict(zip(layer_input_vars, aux_values))

    losses_args, aux_vjp, losses_kwargs = jax.vjp(forward_aux, aux_dict,
                                                  has_aux=True)
    losses = tuple(tag.primitive.loss(*inputs, **kwargs)
                   for tag, inputs, kwargs in
                   zip(loss_tags, losses_args, losses_kwargs))

    def vjp_func(tangents):
      all_tangents = aux_vjp(tangents)
      tangents_dict, inputs_tangents = all_tangents[0], all_tangents[1:]
      inputs_tangents = jax.tree_flatten(inputs_tangents)[0]
      tangents_dict.update(zip(jaxpr.invars, inputs_tangents))

      read_primals = functools.partial(tgm.read_env, primals_dict)
      read_tangents = functools.partial(tgm.read_env, tangents_dict)
      layers_info = []
      for jaxpr_eqn in layer_tags:
        layer_tag = _unbox_layer_tag(jaxpr_eqn)
        info = dict()
        primals = jax_util.safe_map(read_primals, tuple(jaxpr_eqn.invars))
        (
            info["outputs"],
            info["inputs"],
            info["params"],
        ) = layer_tag.split_all_inputs(primals)
        tangents = jax_util.safe_map(read_tangents, tuple(jaxpr_eqn.invars))
        (
            info["outputs_tangent"],
            info["inputs_tangent"],
            info["params_tangent"],
        ) = layer_tag.split_all_inputs(tangents)
        layers_info.append(info)
      return tuple(layers_info)

    return losses, vjp_func
  return full_vjp_func
