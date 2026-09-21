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
"""A module for the main curvature optimizer class."""
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jnr

from kfac_ferminet_alpha import estimator
from kfac_ferminet_alpha import tag_graph_matcher as tgm
from kfac_ferminet_alpha import utils

ScheduleType = Callable[[jnp.ndarray], Optional[jnp.ndarray]]
Parameters = Any
Batch = Any
FuncState = Any
State = Mapping[str, Any]


@utils.Stateful.infer_class_state
class Optimizer(utils.Stateful):
  """The default optimizer class."""
  velocities: Parameters
  estimator: estimator.CurvatureEstimator
  step_counter: jnp.ndarray

  def __init__(
      self,
      value_and_grad_func,
      l2_reg: Union[float, jnp.ndarray],
      value_func_has_aux: bool = False,
      value_func_has_state: bool = False,
      value_func_has_rng: bool = False,
      learning_rate_schedule: Optional[ScheduleType] = None,
      momentum_schedule: Optional[ScheduleType] = None,
      damping_schedule: Optional[ScheduleType] = None,
      min_damping: Union[float, jnp.ndarray] = 1e-8,
      max_damping: Union[float, jnp.ndarray] = jnp.inf,
      norm_constraint: Optional[Union[float, jnp.ndarray]] = None,
      num_burnin_steps: int = 10,
      estimation_mode: str = "fisher_gradients",
      curvature_ema: Union[float, jnp.ndarray] = 0.95,
      inverse_update_period: int = 5,
      register_only_generic: bool = False,
      layer_tag_to_block_cls: Optional[estimator.TagMapping] = None,
      patterns_to_skip: Sequence[str] = (),
      donate_parameters: bool = False,
      donate_optimizer_state: bool = False,
      donate_batch_inputs: bool = False,
      donate_func_state: bool = False,
      batch_process_func: Optional[Callable[[Any], Any]] = None,
      multi_device: bool = False,
      use_jax_cond: bool = True,
      debug: bool = False,
      pmap_axis_name="kfac_axis",
  ):
    """Initializes the K-FAC optimizer with the given settings.

    Args:
      value_and_grad_func: Python callable. The function should return the value
        of the loss to be optimized and its gradients. If the argument
          `value_func_has_aux` is `False` then the interface should be: loss,
            loss_grads = value_and_grad_func(params, batch)
          If `value_func_has_aux` is `True` then the interface should be: (loss,
            aux), loss_grads = value_and_grad_func(params, batch)
      l2_reg: Scalar. Set this value to tell the optimizer what L2
        regularization coefficient you are using (if any). Note the coefficient
        appears in the regularizer as coeff / 2 * sum(param**2). Note that the
        user is still responsible for adding regularization to the loss.
      value_func_has_aux: Boolean. Specifies whether the provided callable
        `value_and_grad_func` returns the loss value only, or also some
          auxiliary data. (Default: False)
      value_func_has_state: Boolean. Specifies whether the provided callable
        `value_and_grad_func` has a persistent state that is inputed and
          it also outputs an update version of it. (Default: False)
      value_func_has_rng: Boolean. Specifies whether the provided callable
        `value_and_grad_func` additionally takes as input an rng key.
          (Default: False)
      learning_rate_schedule: Callable. A schedule for the learning rate. This
        should take as input the current step number and return a single
          `jnp.ndarray` that represents the learning rate. (Default: None)
      momentum_schedule: Callable. A schedule for the momentum. This should take
        as input the current step number and return a single `jnp.ndarray`
          that represents the momentum. (Default: None)
      damping_schedule: Callable. A schedule for the damping. This should take
        as input the current step number and return a single `jnp.ndarray`
          that represents the learning rate. (Default: None)
      min_damping: Scalar. Minimum value the damping parameter can take. Note
        that the default value of 1e-8 is quite arbitrary, and you may have to
        adjust this up or down for your particular problem. If you are using a
        non-zero value of l2_reg you *may* be able to set this to
          zero. (Default: 1e-8)
      max_damping: Scalar. Maximum value the damping parameter can take.
          (Default: Infinity)
      norm_constraint: Scalar. If specified, the update is scaled down so that
        its approximate squared Fisher norm `v^T F v` is at most the specified
        value.(Note that here `F` is the approximate curvature matrix, not the
          exact.) (Default: None)
      num_burnin_steps: Int. At the start of optimization, e.g. the first step,
        before performing the actual step the optimizer will perform this many
        times updates to the curvature approximation without updating the
          actual parameters. (Default: 10)
      estimation_mode: String. The type of estimator to use for the curvature
          matrix. Can be one of: * fisher_empirical * fisher_exact *
            fisher_gradients * fisher_curvature_prop * ggn_exact *
            ggn_curvature_prop See the doc-string for CurvatureEstimator (in
            estimator.py) for a more
          detailed description of these options. (Default: 'fisher_gradients').
      curvature_ema: The decay factor used when calculating the covariance
          estimate moving averages. (Default: 0.95)
      inverse_update_period: Int. The number of steps in between updating the
          the computation of the inverse curvature approximation. (Default: 5)
      register_only_generic: Boolean. Whether when running the auto-tagger to
        register only generic parameters, or allow it to use the graph matcher
          to automatically pick up any kind of layer tags. (Default: False)
      layer_tag_to_block_cls: Dictionary. A mapping from layer tags to block
        classes which to override the default choices of block approximation for
        that specific tag. See the doc-string for CurvatureEstimator (in
        estimator.py) for a more detailed description of this.
      patterns_to_skip: Tuple. A list of any patterns that should be skipped by
        the graph matcher when auto-tagging.
      donate_parameters: Boolean. Whether to use jax's `donate_argnums` to
        donate the parameter values of each call to `step`. Note that this
        implies that you will not be able to access the old parameter values'
        buffers after calling into `step`.
      donate_optimizer_state: Boolean. Whether to use jax's `donate_argnums` to
        donate the optimizer state of each call to `step`. Note that this
        implies that you will not be able to access the old optimizer state
        values' buffers after calling into `step`.
      donate_batch_inputs: Boolean. Whether to use jax's `donate_argnums` to
        donate the batch values of each call to `step`. Note that this implies
        that you will not be able to access the old batch values' buffers after
        calling into `step`.
      donate_func_state: Boolean. Whether to use jax's `donate_argnums` to
        donate the persistent function state of each call to `step`. Note that
        this implies that you will not be able to access the old function state
        values' buffers after calling into `step`.
      batch_process_func: Callable. A function which to be called on each batch
        before feeding to the KFAC on device. This could be useful for specific
        device input optimizations.
      multi_device: Boolean. Whether to use `pmap` and run the optimizer on
          multiple devices. (Default: False)
      use_jax_cond: Not used for the moment.
      debug: Boolean. If non of the step or init functions would be jitted. Note
        that this also overrides `multi_device` and prevents using `pmap`.
          (Default: False)
      pmap_axis_name: String. The name of the `pmap` axis to use when
          `multi_device` is set to True. (Default: curvature_axis)
    """
    super().__init__()
    self.value_and_grad_func = value_and_grad_func
    self.value_func_has_aux = value_func_has_aux
    self.value_func_has_state = value_func_has_state
    self.value_func_has_rng = value_func_has_rng
    self.value_func = utils.convert_value_and_grad_to_value_func(
        value_and_grad_func, has_aux=value_func_has_aux)
    self.l2_reg = l2_reg
    self.learning_rate_schedule = learning_rate_schedule
    if momentum_schedule is not None:

      def schedule_with_first_step_zero(global_step: jnp.ndarray):
        value = momentum_schedule(global_step)
        check = jnp.equal(global_step, 0)
        return check * jnp.zeros_like(value) + (1 - check) * value

      self.momentum_schedule = schedule_with_first_step_zero
    else:
      self.momentum_schedule = None
    self.damping_schedule = damping_schedule
    self.min_damping = min_damping
    self.max_damping = max_damping
    self.norm_constraint = norm_constraint
    self.num_burnin_steps = num_burnin_steps
    self.estimation_mode = estimation_mode
    self.curvature_ema = curvature_ema
    self.inverse_update_period = inverse_update_period
    self.register_only_generic = register_only_generic
    self.layer_tag_to_block_cls = layer_tag_to_block_cls
    self.patterns_to_skip = patterns_to_skip
    self.donate_parameters = donate_parameters
    self.donate_optimizer_state = donate_optimizer_state
    self.donate_batch_inputs = donate_batch_inputs
    self.donate_func_state = donate_func_state
    self.batch_process_func = batch_process_func or (lambda x: x)
    self.multi_device = multi_device
    self.use_jax_cond = use_jax_cond
    self.debug = debug
    self.pmap_axis_name = pmap_axis_name if multi_device else None
    self._rng_split = utils.p_split if multi_device else jnr.split

    # Attributes filled in during self.init()
    self.finalized = False
    self.tagged_func = None
    self.flat_params_shapes = None
    self.params_treedef = None
    # Special attributes related to jitting/pmap
    self._jit_init = None
    self._jit_burnin = None
    self._jit_step = None

  def finalize(
      self,
      params: Parameters,
      rng: jnp.ndarray,
      batch: Batch,
      func_state: Optional[FuncState] = None,
  ) -> None:
    """Finalizes the optimizer by tracing the model function with the params and batch."""
    if self.finalized:
      raise ValueError("Optimizer has already been finalized.")
    if self.multi_device:
      # We assume that the parameters and batch are replicated, while tracing
      # must happen with parameters for a single device call
      params, rng, batch = jax.tree_map(lambda x: x[0], (params, rng, batch))
      if func_state is not None:
        func_state = jax.tree_map(lambda x: x[0], func_state)
    batch = self.batch_process_func(batch)
    # These are all tracing operations and we can run them with abstract values
    func_args = utils.make_func_args(params, func_state, rng, batch,
                                     self.value_func_has_state,
                                     self.value_func_has_rng)
    # Run all tracing with abstract values so no computation is done
    flat_params, self.params_treedef = jax.tree_flatten(params)
    self.flat_params_shapes = tuple(p.shape for p in flat_params)
    self.tagged_func = tgm.auto_register_tags(
        func=self.value_func,
        func_args=func_args,
        params_index=0,
        register_only_generic=self.register_only_generic,
        patterns_to_skip=self.patterns_to_skip)
    self.estimator = estimator.CurvatureEstimator(
        self.tagged_func,
        func_args,
        self.l2_reg,
        self.estimation_mode,
        layer_tag_to_block_cls=self.layer_tag_to_block_cls)
    # Arguments: params, opt_state, rng, batch, func_state
    donate_argnums = []
    if self.donate_parameters:
      donate_argnums.append(0)
    if self.donate_optimizer_state:
      donate_argnums.append(1)
    if self.donate_batch_inputs:
      donate_argnums.append(3)
    if self.donate_func_state and self.value_func_has_state:
      donate_argnums.append(4)
    donate_argnums = tuple(donate_argnums)

    if self.debug:
      self._jit_init = self._init
      self._jit_burnin = self._burnin
      self._jit_step = self._step
    elif self.multi_device:
      self._jit_init = jax.pmap(
          self._init, axis_name=self.pmap_axis_name, donate_argnums=[0])
      # batch size is static argnum and is at index 5
      self._jit_burnin = jax.pmap(
          self._burnin,
          axis_name=self.pmap_axis_name,
          static_broadcasted_argnums=[5])
      self._jit_step = jax.pmap(
          self._step,
          axis_name=self.pmap_axis_name,
          donate_argnums=donate_argnums,
          static_broadcasted_argnums=[5])
    else:
      self._jit_init = jax.jit(self._init, donate_argnums=[0])
      # batch size is static argnum and is at index 5
      self._jit_burnin = jax.jit(self._burnin, static_argnums=[5])
      self._jit_step = jax.jit(
          self._step, donate_argnums=donate_argnums, static_argnums=[5])
    self.finalized = True

  def _init(self, rng: jnp.ndarray) -> State:
    """This is the non-jitted version of initializing the state."""
    flat_velocities = [jnp.zeros(shape) for shape in self.flat_params_shapes]
    return dict(
        velocities=jax.tree_unflatten(self.params_treedef, flat_velocities),
        estimator=self.estimator.init(rng, None),
        step_counter=jnp.asarray(0))

  def verify_args_and_get_step_counter(
      self,
      params: Parameters,
      state: State,
      rng: jnp.ndarray,
      data_iterator: Iterator[Batch],
      func_state: Optional[FuncState] = None,
      learning_rate: Optional[jnp.ndarray] = None,
      momentum: Optional[jnp.ndarray] = None,
      damping: Optional[jnp.ndarray] = None,
      global_step_int: Optional[int] = None,
  ) -> int:
    """Verifies that the arguments passed to `Optimizer.step` are correct."""
    if not self.finalized:
      rng, rng_finalize = self._rng_split(rng)
      self.finalize(params, rng_finalize, next(data_iterator), func_state)
    # Verify correct arguments invocation
    if self.learning_rate_schedule is not None and learning_rate is not None:
      raise ValueError("When you have passed a `learning_rate_schedule` you "
                       "should not pass a value to the step function.")
    if self.momentum_schedule is not None and momentum is not None:
      raise ValueError("When you have passed a `momentum_schedule` you should "
                       "not pass a value to the step function.")
    if self.damping_schedule is not None and damping is not None:
      raise ValueError("When you have passed a `damping_schedule` you should "
                       "not pass a value to the step function.")
    # Do a bunrnin on the first iteration
    if global_step_int is None:
      if self.multi_device:
        return int(utils.get_first(state["step_counter"]))
      else:
        return int(state["step_counter"])
    return global_step_int

  def _burnin(
      self,
      params: Parameters,
      state: State,
      rng: jnp.ndarray,
      batch: Batch,
      func_state: Optional[FuncState],
      batch_size: Optional[int],
  ) -> Tuple[State, Optional[FuncState]]:
    """This is the non-jitted version of a single burnin step."""
    self.set_state(state)
    batch = self.batch_process_func(batch)
    rng, func_rng = jnr.split(rng) if self.value_func_has_rng else (rng, None)
    func_args = utils.make_func_args(params, func_state, func_rng, batch,
                                     self.value_func_has_state,
                                     self.value_func_has_rng)

    # Compute batch size
    if batch_size is None:
      batch_size = jax.tree_flatten(batch)[0][0].shape[0]

    # Update curvature estimate
    ema_old, ema_new = 1.0, 1.0 / self.num_burnin_steps
    self.estimator.update_curvature_matrix_estimate(ema_old, ema_new,
                                                    batch_size, rng, func_args,
                                                    self.pmap_axis_name)

    if func_state is not None:
      out, _ = self.value_and_grad_func(*func_args)
      _, func_state, _ = utils.extract_func_outputs(out,
                                                    self.value_func_has_aux,
                                                    self.value_func_has_state)

    return self.pop_state(), func_state

  def _step(
      self,
      params: Parameters,
      state: State,
      rng: jnp.ndarray,
      batch: Batch,
      func_state: Optional[FuncState],
      batch_size: Optional[int],
      learning_rate: Optional[jnp.ndarray],
      momentum: Optional[jnp.ndarray],
      damping: Optional[jnp.ndarray],
  ) -> Union[Tuple[Parameters, State, FuncState, Mapping[str, jnp.ndarray]],
             Tuple[Parameters, State, Mapping[str, jnp.ndarray]]]:
    """This is the non-jitted version of a single step."""
    # Unpack and set the state
    self.set_state(state)
    if damping is not None:
      assert self.estimator.damping is None
      self.estimator.damping = damping
    else:
      assert self.estimator.damping is not None

    # Preprocess the batch and construct correctly the function arguments
    batch = self.batch_process_func(batch)
    rng, func_rng = jnr.split(rng) if self.value_func_has_rng else (rng, None)
    func_args = utils.make_func_args(params, func_state, func_rng, batch,
                                     self.value_func_has_state,
                                     self.value_func_has_rng)

    # Compute the batch size
    if batch_size is None:
      batch_size = jax.tree_flatten(batch)[0][0].shape[0]

    # Compute schedules if applicable
    if self.learning_rate_schedule is not None:
      assert learning_rate is None
      learning_rate = self.learning_rate_schedule(self.step_counter)
    else:
      assert learning_rate is not None
    if self.momentum_schedule is not None:
      assert momentum is None
      momentum = self.momentum_schedule(self.step_counter)
    else:
      assert momentum is not None
    if self.damping_schedule is not None:
      assert damping is None
      damping = self.damping_schedule(self.step_counter)
    else:
      assert damping is not None

    # Compute current loss and gradients
    out, grads = self.value_and_grad_func(*func_args)
    loss, new_func_state, aux = utils.extract_func_outputs(
        out, self.value_func_has_aux, self.value_func_has_state)
    # Sync loss and grads
    loss, grads = utils.pmean_if_pmap((loss, grads), self.pmap_axis_name)

    # Update curvature estimate
    self.estimator.update_curvature_matrix_estimate(
        self.curvature_ema,
        1.0,
        batch_size,
        rng,
        func_args,
        self.pmap_axis_name,
    )

    # Optionally update the inverse estimate
    self.estimator.set_state(
        lax.cond(
            self.step_counter % self.inverse_update_period == 0,
            lambda s: self.estimator.update_curvature_estimate_inverse(  # pylint: disable=g-long-lambda
                self.pmap_axis_name, s),
            lambda s: s,
            self.estimator.pop_state()))

    # Compute proposed directions
    vectors = self.propose_directions(
        grads,
        self.velocities,
        learning_rate,
        momentum,
    )

    # The learning rate is defined as the negative of the coefficient by which
    # we multiply the gradients, while the momentum is the coefficient by
    # which we multiply the velocities.
    neg_learning_rate = -learning_rate  # pytype: disable=unsupported-operands  # trace-all-classes
    # Compute the coefficients of the update vectors
    assert neg_learning_rate is not None and momentum is not None
    coefficients = (neg_learning_rate, momentum)

    # Update velocities and compute new delta
    self.velocities, delta = self.velocities_and_delta(
        self.velocities,
        vectors,
        coefficients,
    )

    # Update parameters: params = params + delta
    params = jax.tree_map(jnp.add, params, delta)

    # Optionally compute the reduction ratio and update the damping
    self.estimator.damping = None
    rho = jnp.nan

    # Statistics with useful information
    stats = dict()
    stats["step"] = self.step_counter
    stats["loss"] = loss
    stats["learning_rate"] = -coefficients[0]
    stats["momentum"] = coefficients[1]
    stats["damping"] = damping
    stats["rho"] = rho
    if self.value_func_has_aux:
      stats["aux"] = aux
    self.step_counter = self.step_counter + 1

    if self.value_func_has_state:
      return params, self.pop_state(), new_func_state, stats  # pytype: disable=bad-return-type  # jax-ndarray
    else:
      assert new_func_state is None
      return params, self.pop_state(), stats  # pytype: disable=bad-return-type  # jax-ndarray

  def init(
      self,
      params: Parameters,
      rng: jnp.ndarray,
      batch: Batch,
      func_state: Optional[FuncState] = None,
  ) -> State:
    """Initializes the optimizer and returns the appropriate optimizer state."""
    if not self.finalized:
      self.finalize(params, rng, batch, func_state)
    return self._jit_init(rng)

  def step(
      self,
      params: Parameters,
      state: Mapping[str, Any],
      rng: jnp.ndarray,
      data_iterator: Iterator[Any],
      func_state: Any = None,
      learning_rate: Optional[jnp.ndarray] = None,
      momentum: Optional[jnp.ndarray] = None,
      damping: Optional[jnp.ndarray] = None,
      batch_size: Optional[int] = None,
      global_step_int: Optional[int] = None,
  ) -> Union[Tuple[Parameters, State, FuncState, Mapping[str, jnp.ndarray]],
             Tuple[Parameters, State, Mapping[str, jnp.ndarray]]]:
    """Performs a single update step using the optimizer.

    Args:
      params: The parameters of the model.
      state: The state of the optimizer.
      rng: A Jax PRNG key.
      data_iterator: An iterator that returns a batch of data.
      func_state: Any function state that gets passed in and returned.
      learning_rate: This must be provided when
        `use_adaptive_learning_rate=False` and `learning_rate_schedule=None`.
      momentum: This must be provided when
        `use_adaptive_momentum=False` and `momentum_schedule=None`.
      damping: This must be provided when
        `use_adaptive_damping=False` and `damping_schedule=None`.
      batch_size: The batch size to use for KFAC. The default behaviour when it
        is None is to use the leading dimension of the first data array.
      global_step_int: The global step as a python int. Note that this must
        match the step inte  rnal to the optimizer that is part of its state.

    Returns:
      (params, state, stats)
      where:
          params: The updated model parameters.
          state: The updated optimizer state.
          stats: A dictionary of key statistics provided to be logged.
    """
    step_counter_int = self.verify_args_and_get_step_counter(
        params=params,
        state=state,
        rng=rng,
        data_iterator=data_iterator,
        func_state=func_state,
        learning_rate=learning_rate,
        momentum=momentum,
        damping=damping,
        global_step_int=global_step_int)

    if step_counter_int == 0:
      for _ in range(self.num_burnin_steps):
        rng, rng_burn = self._rng_split(rng)
        batch = next(data_iterator)
        state, func_state = self._jit_burnin(params, state, rng_burn, batch,
                                             func_state, batch_size)

      # On the first step we always treat the momentum as 0.0
      if self.momentum_schedule is None:
        momentum = jnp.zeros([])
        if self.multi_device:
          momentum = utils.replicate_all_local_devices(momentum)

    batch = next(data_iterator)
    return self._jit_step(params, state, rng, batch, func_state, batch_size,
                          learning_rate, momentum, damping)

  def propose_directions(
      self,
      grads: Parameters,
      velocities: Parameters,
      learning_rate: Optional[jnp.ndarray],
      momentum: Optional[jnp.ndarray],
  ) -> Tuple[Parameters, Parameters]:
    """Computes the vector proposals for the next step."""
    del momentum  # not used in this, but could be used in subclasses
    preconditioned_grads = self.estimator.multiply_matpower(grads, -1)

    if self.norm_constraint is not None:
      assert learning_rate is not None
      sq_norm_grads = utils.inner_product(preconditioned_grads, grads)
      sq_norm_scaled_grads = sq_norm_grads * learning_rate**2

      # We need to sync the norms here, because reduction can be
      # non-deterministic. They specifically are on GPUs by default for better
      # performance. Hence although grads and preconditioned_grads are synced,
      # the inner_product operation can still produce different answers on
      # different devices.
      sq_norm_scaled_grads = utils.pmean_if_pmap(sq_norm_scaled_grads,
                                                 self.pmap_axis_name)

      max_coefficient = jnp.sqrt(self.norm_constraint / sq_norm_scaled_grads)
      coefficient = jnp.minimum(max_coefficient, 1)
      preconditioned_grads = utils.scalar_mul(preconditioned_grads, coefficient)

    return preconditioned_grads, velocities

  def velocities_and_delta(
      self,
      velocities: Parameters,
      vectors: Sequence[Parameters],
      coefficients: Sequence[jnp.ndarray],
  ) -> Sequence[Parameters]:
    """Computes the new velocities and delta (update to parameters)."""
    del velocities
    assert len(vectors) == len(coefficients)
    delta = utils.scalar_mul(vectors[0], coefficients[0])
    for vi, wi in zip(vectors[1:], coefficients[1:]):
      delta = jax.tree_map(jnp.add, delta, utils.scalar_mul(vi, wi))
    return delta, delta
