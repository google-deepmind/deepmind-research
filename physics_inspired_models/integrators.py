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
"""Module containing the implementations of the various numerical integrators.

Higher order methods mostly taken from [1].

References:
  [1] Leimkuhler, Benedict and Sebastian Reich. Simulating hamiltonian dynamics.
    Vol. 14. Cambridge university press, 2004.
  [2] Forest, Etienne and Ronald D. Ruth. Fourth-order symplectic integration.
    Physica D: Nonlinear Phenomena 43.1 (1990): 105-117.
  [3] Blanes, Sergio and Per Christian Moan. Practical symplectic partitioned
    Runge–Kutta and Runge–Kutta–Nyström methods. Journal of Computational and
    Applied Mathematics 142.2 (2002): 313-330.
  [4] McLachlan, Robert I. On the numerical integration of ordinary differential
    equations by symmetric composition methods. SIAM Journal on Scientific
    Computing 16.1 (1995): 151-168.
  [5] Yoshida, Haruo. Construction of higher order symplectic integrators.
    Physics letters A 150.5-7 (1990): 262-268.
  [6] Süli, Endre; Mayers, David (2003), An Introduction to Numerical Analysis,
    Cambridge University Press, ISBN 0-521-00794-1.
  [7] Hairer, Ernst; Nørsett, Syvert Paul; Wanner, Gerhard (1993), Solving
    ordinary differential equations I: Nonstiff problems, Berlin, New York:
    Springer-Verlag, ISBN 978-3-540-56670-0.
"""
from typing import Callable, Dict, Optional, Sequence, Tuple, TypeVar, Union

from dm_hamiltonian_dynamics_suite.hamiltonian_systems import phase_space
import jax
from jax import lax
from jax.experimental import ode
import jax.numpy as jnp
import numpy as np

M = TypeVar("M")
TM = TypeVar("TM")
TimeInterval = Union[jnp.ndarray, Tuple[float, float]]

#    _____                           _
#   / ____|                         | |
#  | |  __  ___ _ __   ___ _ __ __ _| |
#  | | |_ |/ _ \ '_ \ / _ \ '__/ _` | |
#  | |__| |  __/ | | |  __/ | | (_| | |
#   \_____|\___|_| |_|\___|_|  \__,_|_|
#   _____       _                       _   _
#  |_   _|     | |                     | | (_)
#    | |  _ __ | |_ ___  __ _ _ __ __ _| |_ _  ___  _ __
#    | | | '_ \| __/ _ \/ _` | '__/ _` | __| |/ _ \| '_ \
#   _| |_| | | | ||  __/ (_| | | | (_| | |_| | (_) | | | |
#  |_____|_| |_|\__\___|\__, |_|  \__,_|\__|_|\___/|_| |_|
#                        __/ |
#                       |___/


GeneralTangentFunction = Callable[
    [
        Optional[Union[float, jnp.ndarray]],  # t
        M  # y
    ],
    TM  # dy_dt
]

GeneralIntegrator = Callable[
    [
        GeneralTangentFunction,
        Optional[Union[float, jnp.ndarray]],  # t
        M,  # y
        jnp.ndarray,  # dt
    ],
    M  # y_next
]


def solve_ivp_dt(
    fun: GeneralTangentFunction,
    y0: M,
    t0: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    method: Union[str, GeneralIntegrator],
    num_steps: Optional[int] = None,
    steps_per_dt: int = 1,
    use_scan: bool = True,
    ode_int_kwargs: Optional[Dict[str, Union[float, int]]] = None
) -> Tuple[jnp.ndarray, M]:
  """Solve an initial value problem for a system of ODEs using explicit method.

  This function numerically integrates a system of ordinary differential
  equations given an initial value::
      dy / dt = f(t, y)
      y(t0) = y0
  Here t is a one-dimensional independent variable (time), y(t) is an
  n-dimensional vector-valued function (state), and an n-dimensional
  vector-valued function f(t, y) determines the differential equations.
  The goal is to find y(t) approximately satisfying the differential
  equations, given an initial value y(t0)=y0.

  All of the solvers supported here are explicit and non-adaptive. This makes
  them easy to run with a fixed amount of computation and ensures solutions are
  easily differentiable.

  Args:
    fun: callable
      Right-hand side of the system. The calling signature is ``fun(t, y)``.
      Here `t` is a scalar representing the time instance. `y` can be any
      type `M`, including a flat array, that is registered as a
      pytree. In addition, there is a type denoted as `TM` that represents
      the tangent space to `M`. It is assumed that any element of `TM` can be
      multiplied by arrays and scalars, can be added to other `TM` instances
      as well as they can be right added to an element of `M`, that is
      add(M, TM) exists. The function should return an element of `TM` that
      defines the time derivative of `y`.
    y0: an instance of `M`
      Initial state at `t_span[0]`.
    t0: float or array.
      The initial time point of integration.
    dt: array
      Array containing all consecutive increments in time, at which the integral
      to be evaluated. The size of this array along axis 0 defines the number of
      steps that the integrator would do.
    method: string or `GeneralIntegrator`
      The integrator method to use. Possible values for string are:
        * general_euler - see `GeneralEuler`
        * rk2 - see `RungaKutta2`
        * rk4 - see `RungaKutta4`
        * rk38 - see `RungaKutta38`
    num_steps: Optional int.
      If provided the `dt` will be treated as the same per step time interval,
      applied for this many steps. In other words setting this argument is
      equivalent to replicating `dt` num_steps times and stacking over axis=0.
    steps_per_dt: int
      This determines the overall step size. Between any two values of t_eval
      the step size is `dt = (t_eval[i+1] - t_eval[i]) / steps_per_dt.
    use_scan: bool
      Whether for the loop to use `lax.scan` or a python loop
    ode_int_kwargs: dict
      Extra arguments to be passed to `ode.odeint` when method="adaptive"

  Returns:
    t: array
      Time points at which the solution is evaluated.
    y : an instance of M
      Values of the solution at `t`.
  """
  if method == "adaptive":
    ndim = y0.q.ndim if isinstance(y0, phase_space.PhaseSpace) else y0.ndim
    signs = jnp.asarray(jnp.sign(dt))
    signs = signs.reshape([-1] + [1] * (ndim - 1))
    if isinstance(dt, float) or dt.ndim == 0:
      true_t_eval = t0 + dt * np.arange(1, num_steps + 1)
    else:
      true_t_eval = t0 + dt[None] * np.arange(1, num_steps + 1)[:, None]
    if isinstance(dt, float):
      dt = np.asarray(dt)
    if isinstance(dt, np.ndarray) and dt.ndim > 0:
      if np.all(np.abs(dt) != np.abs(dt[0])):
        raise ValueError("Not all values of `dt` where the same.")
    elif isinstance(dt, jnp.ndarray) and dt.ndim > 0:
      raise ValueError("The code here works only when `dy_dt` is time "
                       "independent and `np.abs(dt)` is the same. For this we "
                       "allow calling this only with numpy (not jax.numpy) "
                       "arrays.")
    dt: jnp.ndarray = jnp.abs(jnp.asarray(dt))
    dt = dt.reshape([-1])[0]
    t_eval = t0 + dt * np.arange(num_steps + 1)

    outputs = ode.odeint(
        func=lambda y_, t_: fun(None, y_) * signs,
        y0=y0,
        t=jnp.abs(t_eval - t0),
        **(ode_int_kwargs or dict())
    )
    # Note that we do not return the initial point
    return true_t_eval, jax.tree_map(lambda x: x[1:], outputs)

  method = get_integrator(method)
  if num_steps is not None:
    dt = jnp.repeat(jnp.asarray(dt)[None], repeats=num_steps, axis=0)
  t_eval = t0 + jnp.cumsum(dt, axis=0)
  t0 = jnp.ones_like(t_eval[..., :1]) * t0
  t = jnp.concatenate([t0, t_eval[..., :-1]], axis=-1)
  def loop_body(y_: M, t_dt: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[M, M]:
    t_, dt_ = t_dt
    dt_: jnp.ndarray = dt_ / steps_per_dt
    for _ in range(steps_per_dt):
      y_ = method(fun, t_, y_, dt_)
      t_ = t_ + dt_
    return y_, y_
  if use_scan:
    return t_eval, lax.scan(loop_body, init=y0, xs=(t, dt))[1]
  else:
    y = [y0]
    for t_and_dt_i in zip(t, dt):
      y.append(loop_body(y[-1], t_and_dt_i)[0])
    # Note that we do not return the initial point
    return t_eval, jax.tree_multimap(lambda *args: jnp.stack(args, axis=0),
                                     *y[1:])


def solve_ivp_dt_two_directions(
    fun: GeneralTangentFunction,
    y0: M,
    t0: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    method: Union[str, GeneralIntegrator],
    num_steps_forward: int,
    num_steps_backward: int,
    include_y0: bool = True,
    steps_per_dt: int = 1,
    use_scan: bool = True,
    ode_int_kwargs: Optional[Dict[str, Union[float, int]]] = None
) -> M:
  """Equivalent to `solve_ivp_dt` but you can specify unrolling the problem for a fixed number of steps in both time directions."""
  yt = []
  if num_steps_backward > 0:
    yt_bck = solve_ivp_dt(
        fun=fun,
        y0=y0,
        t0=t0,
        dt=- dt,
        method=method,
        num_steps=num_steps_backward,
        steps_per_dt=steps_per_dt,
        use_scan=use_scan,
        ode_int_kwargs=ode_int_kwargs
    )[1]
    yt.append(jax.tree_map(lambda x: jnp.flip(x, axis=0), yt_bck))
  if include_y0:
    yt.append(jax.tree_map(lambda x: x[None], y0))
  if num_steps_forward > 0:
    yt_fwd = solve_ivp_dt(
        fun=fun,
        y0=y0,
        t0=t0,
        dt=dt,
        method=method,
        num_steps=num_steps_forward,
        steps_per_dt=steps_per_dt,
        use_scan=use_scan,
        ode_int_kwargs=ode_int_kwargs
    )[1]
    yt.append(yt_fwd)
  if len(yt) > 1:
    return jax.tree_multimap(lambda *a: jnp.concatenate(a, axis=0), *yt)
  else:
    return yt[0]


def solve_ivp_t_eval(
    fun: GeneralTangentFunction,
    t_span: TimeInterval,
    y0: M,
    method: Union[str, GeneralIntegrator],
    t_eval: Optional[jnp.ndarray] = None,
    steps_per_dt: int = 1,
    use_scan: bool = True,
    ode_int_kwargs: Optional[Dict[str, Union[float, int]]] = None
) -> Tuple[jnp.ndarray, M]:
  """Solve an initial value problem for a system of ODEs using an explicit method.

  This function numerically integrates a system of ordinary differential
  equations given an initial value::
      dy / dt = f(t, y)
      y(t0) = y0
  Here t is a one-dimensional independent variable (time), y(t) is an
  n-dimensional vector-valued function (state), and an n-dimensional
  vector-valued function f(t, y) determines the differential equations.
  The goal is to find y(t) approximately satisfying the differential
  equations, given an initial value y(t0)=y0.

  All of the solvers supported here are explicit and non-adaptive. This in
  terms makes them easy to run with fixed amount of computation and
  the solutions to be easily differentiable.

  Args:
    fun: callable
      Right-hand side of the system. The calling signature is ``fun(t, y)``.
      Here `t` is a scalar representing the time instance. `y` can be any
      type `M`, including a flat array, that is registered as a
      pytree. In addition, there is a type denoted as `TM` that represents
      the tangent space to `M`. It is assumed that any element of `TM` can be
      multiplied by arrays and scalars, can be added to other `TM` instances
      as well as they can be right added to an element of `M`, that is
      add(M, TM) exists. The function should return an element of `TM` that
      defines the time derivative of `y`.
    t_span: 2-tuple of floats
      Interval of integration (t0, tf). The solver starts with t=t0 and
      integrates until it reaches t=tf.
    y0: an instance of `M`
      Initial state at `t_span[0]`.
    method: string or `GeneralIntegrator`
      The integrator method to use. Possible values for string are:
        * general_euler - see `GeneralEuler`
        * rk2 - see `RungaKutta2`
        * rk4 - see `RungaKutta4`
        * rk38 - see `RungaKutta38`
    t_eval: array or None.
      Times at which to store the computed solution. Must be sorted and lie
      within `t_span`. If None then t_eval = [t_span[-1]]
    steps_per_dt: int
      This determines the overall step size. Between any two values of t_eval
      the step size is `dt = (t_eval[i+1] - t_eval[i]) / steps_per_dt.
    use_scan: bool
      Whether for the loop to use `lax.scan` or a python loop
    ode_int_kwargs: dict
      Extra arguments to be passed to `ode.odeint` when method="adaptive"

  Returns:
    t: array
      Time points at which the solution is evaluated.
    y : an instance of M
      Values of the solution at `t`.
  """
  # Check for t_eval
  if t_eval is None:
    t_eval = np.asarray([t_span[-1]])
  if isinstance(t_span[0], float) and isinstance(t_span[1], float):
    t_span = np.asarray(t_span)
  elif isinstance(t_span[0], float) and isinstance(t_span[1], jnp.ndarray):
    t_span = (np.full_like(t_span[1], t_span[0]), t_span[1])
    t_span = np.stack(t_span, axis=0)
  elif isinstance(t_span[1], float) and isinstance(t_span[0], jnp.ndarray):
    t_span = (t_span[0], jnp.full_like(t_span[0], t_span[1]))
    t_span = np.stack(t_span, axis=0)
  else:
    t_span = np.stack(t_span, axis=0)
  def check_span(span, ts):
    # Verify t_span and t_eval
    if span[0] < span[1]:
      # Forward in time
      if not np.all(np.logical_and(span[0] <= ts, ts <= span[1])):
        raise ValueError("Values in `t_eval` are not within `t_span`.")
      if not np.all(ts[:-1] < ts[1:]):
        raise ValueError("Values in `t_eval` are not properly sorted.")
    else:
      # Backward in time
      if not np.all(np.logical_and(span[0] >= ts, ts >= span[1])):
        raise ValueError("Values in `t_eval` are not within `t_span`.")
      if not np.all(ts[:-1] > ts[1:]):
        raise ValueError("Values in `t_eval` are not properly sorted.")
  if t_span.ndim == 1:
    check_span(t_span, t_eval)
  elif t_span.ndim == 2:
    if t_eval.ndim != 2:
      raise ValueError("t_eval should have rank 2.")
    for i in range(t_span.shape[1]):
      check_span(t_span[:, i], t_eval[:, i])

  t = np.concatenate([t_span[:1], t_eval[:-1]], axis=0)

  return solve_ivp_dt(
      fun=fun,
      y0=y0,
      t0=t_span[0],
      dt=t_eval - t,
      method=method,
      steps_per_dt=steps_per_dt,
      use_scan=use_scan,
      ode_int_kwargs=ode_int_kwargs
  )


class RungaKutta(GeneralIntegrator):
  """A general Runga-Kutta integrator defined using a Butcher tableau."""

  def __init__(
      self,
      a_tableau: Sequence[Sequence[float]],
      b_tableau: Sequence[float],
      c_tableau: Sequence[float],
      order: int):
    if len(b_tableau) != len(c_tableau) + 1:
      raise ValueError("The length of b_tableau should be exactly one more than"
                       " the length of c_tableau.")
    if len(b_tableau) != len(a_tableau) + 1:
      raise ValueError("The length of b_tableau should be exactly one more than"
                       " the length of a_tableau.")
    self.a_tableau = a_tableau
    self.b_tableau = b_tableau
    self.c_tableau = c_tableau
    self.order = order

  def __call__(
      self,
      tangent_func: GeneralTangentFunction,
      t: jnp.ndarray,
      y: M,
      dt: jnp.ndarray
  ) -> M:  # pytype: disable=invalid-annotation
    k = [tangent_func(t, y)]
    zero = jax.tree_map(jnp.zeros_like, k[0])
    # We always broadcast opposite to numpy (e.g. leading dims (batch) count)
    if dt.ndim > 0:
      dt = dt.reshape(dt.shape + (1,) * (y.ndim - dt.ndim))
    if t.ndim > 0:
      t = t.reshape(t.shape + (1,) * (y.ndim - t.ndim))
    for c_n, a_n_row in zip(self.c_tableau, self.a_tableau):
      t_n = t + dt * c_n
      products = [a_i * k_i for a_i, k_i in zip(a_n_row, k) if a_i != 0.0]
      delta_n = sum(products, zero)
      y_n = y + dt * delta_n
      k.append(tangent_func(t_n, y_n))
    products = [b_i * k_i for b_i, k_i in zip(self.b_tableau, k) if b_i != 0.0]
    delta = sum(products, zero)
    return y + dt * delta


class GeneralEuler(RungaKutta):
  """The standard Euler method (for general ODE problems)."""

  def __init__(self):
    super().__init__(
        a_tableau=[],
        b_tableau=[1.0],
        c_tableau=[],
        order=1
    )


class RungaKutta2(RungaKutta):
  """The second order Runga-Kutta method corresponding to the mid-point rule."""

  def __init__(self):
    super().__init__(
        a_tableau=[[1.0 / 2.0]],
        b_tableau=[0.0, 1.0],
        c_tableau=[1.0 / 2.0],
        order=2
    )


class RungaKutta4(RungaKutta):
  """The fourth order Runga-Kutta method from [6]."""

  def __init__(self):
    super().__init__(
        a_tableau=[[1.0 / 2.0],
                   [0.0, 1.0 / 2.0],
                   [0.0, 0.0, 1.0]],
        b_tableau=[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
        c_tableau=[1.0 / 2.0, 1.0 / 2.0, 1.0],
        order=4
    )


class RungaKutta38(RungaKutta):
  """The fourth order 3/8 rule Runga-Kutta method from [7]."""

  def __init__(self):
    super().__init__(
        a_tableau=[[1.0 / 3.0],
                   [-1.0 / 3.0, 1.0],
                   [1.0, -1.0, 1.0]],
        b_tableau=[1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0],
        c_tableau=[1.0 / 3.0, 2.0 / 3.0, 1.0],
        order=4
    )


#    _____                       _           _   _
#   / ____|                     | |         | | (_)
#  | (___  _   _ _ __ ___  _ __ | | ___  ___| |_ _  ___
#   \___ \| | | | '_ ` _ \| '_ \| |/ _ \/ __| __| |/ __|
#   ____) | |_| | | | | | | |_) | |  __/ (__| |_| | (__
#  |_____/ \__, |_| |_| |_| .__/|_|\___|\___|\__|_|\___|
#           __/ |         | |
#          |___/          |_|
#   _____       _                       _   _
#  |_   _|     | |                     | | (_)
#    | |  _ __ | |_ ___  __ _ _ __ __ _| |_ _  ___  _ __
#    | | | '_ \| __/ _ \/ _` | '__/ _` | __| |/ _ \| '_ \
#   _| |_| | | | ||  __/ (_| | | | (_| | |_| | (_) | | | |
#  |_____|_| |_|\__\___|\__, |_|  \__,_|\__|_|\___/|_| |_|
#                        __/ |
#                       |___/


SymplecticIntegrator = Callable[
    [
        phase_space.SymplecticTangentFunction,
        jnp.ndarray,  # t
        phase_space.PhaseSpace,  # (q, p)
        jnp.ndarray,  # dt
    ],
    phase_space.PhaseSpace  # (q_next, p_next)
]


def solve_hamiltonian_ivp_dt(
    hamiltonian: phase_space.HamiltonianFunction,
    y0: phase_space.PhaseSpace,
    t0: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    method: Union[str, SymplecticIntegrator],
    num_steps: Optional[int] = None,
    steps_per_dt: int = 1,
    use_scan: bool = True,
    ode_int_kwargs: Optional[Dict[str, Union[float, int]]] = None
) -> Tuple[jnp.ndarray, phase_space.PhaseSpace]:
  """Solve an initial value problem for a Hamiltonian system.

  This function numerically integrates a Hamiltonian system given an
  initial value::
      dq / dt = dH / dp
      dp / dt = - dH / dq
      q(t0), p(t0) = y0.q, y0.p
  Here t is a one-dimensional independent variable (time), y(t) is an
  n-dimensional vector-valued function (state), and an n-dimensional
  vector-valued function H(t, q, p) determines the value of the Hamiltonian.
  The goal is to find q(t) and p(t) approximately satisfying the differential
  equations, given an initial values q(t0), p(t0) = y0.q, y0.p

  All of the solvers supported here are explicit and non-adaptive. This in
  terms makes them easy to run with fixed amount of computation and
  the solutions to be easily differentiable.

  Args:
    hamiltonian: callable
      The Hamiltonian function. The calling signature is ``h(t, s)``, where
      `s` is an instance of `PhaseSpace`.
    y0: an instance of `M`
      Initial state at t=t0.
    t0: float or array.
      The initial time point of integration.
    dt: array
      Array containing all consecutive increments in time, at which the integral
      to be evaluated. The size of this array along axis 0 defines the number of
      steps that the integrator would do.
    method: string or `GeneralIntegrator`
      The integrator method to use. Possible values for string are:
        * symp_euler - see `SymplecticEuler`
        * symp_euler_q - a `SymplecticEuler` with position_first=True
        * symp_euler_p - a `SymplecticEuler` with position_first=False
        * leap_frog - see `LeapFrog`
        * leap_frog_q - a `LeapFrog` with position_first=True
        * leap_frog_p - a `LeapFrog` with position_first=False
        * stormer_verlet - same as leap_frog
        * stormer_verlet_q - same as leap_frog_q
        * stormer_verlet_p - same as leap_frog_p
        * ruth4 - see `Ruth4`,
        * sym4 - see `Symmetric4`
        * sym6 - see `Symmetric6`
        * so4 - see `SymmetricSo4`
        * so4_q - a `SymmetricSo4` with position_first=True
        * so4_p - a `SymmetricSo4` with position_first=False
        * so6 - see `SymmetricSo6`
        * so6_q - a `SymmetricSo6` with position_first=True
        * so6_p - a `SymmetricSo6` with position_first=False
        * so8 - see `SymmetricSo8`
        * so8_q - a `SymmetricSo8` with position_first=True
        * so8_p - a `SymmetricSo8` with position_first=False
    num_steps: Optional int.
      If provided the `dt` will be treated as the same per step time interval,
      applied for this many steps. In other words setting this argument is
      equivalent to replicating `dt` num_steps times and stacking over axis=0.
    steps_per_dt: int
      This determines the overall step size. Between any two values of t_eval
      the step size is `dt = (t_eval[i+1] - t_eval[i]) / steps_per_dt.
    use_scan: bool
      Whether for the loop to use `lax.scan` or a python loop
    ode_int_kwargs: dict
      Extra arguments to be passed to `ode.odeint` when method="adaptive"

  Returns:
    t: array
      Time points at which the solution is evaluated.
    y : an instance of M
      Values of the solution at `t`.
  """
  if not isinstance(y0, phase_space.PhaseSpace):
    raise ValueError("The initial state must be an instance of `PhaseSpace`.")
  dy_dt = phase_space.poisson_bracket_with_q_and_p(hamiltonian)

  return solve_ivp_dt(
      fun=dy_dt,
      y0=y0,
      t0=t0,
      dt=dt,
      method=method,
      num_steps=num_steps,
      steps_per_dt=steps_per_dt,
      use_scan=use_scan,
      ode_int_kwargs=ode_int_kwargs
  )


def solve_hamiltonian_ivp_t_eval(
    hamiltonian: phase_space.HamiltonianFunction,
    t_span: TimeInterval,
    y0: phase_space.PhaseSpace,
    method: Union[str, SymplecticIntegrator],
    t_eval: Optional[jnp.ndarray] = None,
    steps_per_dt: int = 1,
    use_scan: bool = True,
    ode_int_kwargs: Optional[Dict[str, Union[float, int]]] = None
) -> Tuple[jnp.ndarray, phase_space.PhaseSpace]:
  """Solve an initial value problem for a Hamiltonian system.

  This function numerically integrates a Hamiltonian system given an
  initial value::
      dq / dt = dH / dp
      dp / dt = - dH / dq
      q(t0), p(t0) = y0.q, y0.p
  Here t is a one-dimensional independent variable (time), y(t) is an
  n-dimensional vector-valued function (state), and an n-dimensional
  vector-valued function H(t, q, p) determines the value of the Hamiltonian.
  The goal is to find q(t) and p(t) approximately satisfying the differential
  equations, given an initial values q(t0), p(t0) = y0.q, y0.p

  All of the solvers supported here are explicit and non-adaptive. This in
  terms makes them easy to run with fixed amount of computation and
  the solutions to be easily differentiable.

  Args:
    hamiltonian: callable
      The Hamiltonian function. The calling signature is ``h(t, s)``, where
      `s` is an instance of `PhaseSpace`.
    t_span: 2-tuple of floats
      Interval of integration (t0, tf). The solver starts with t=t0 and
      integrates until it reaches t=tf.
    y0: an instance of `M`
      Initial state at `t_span[0]`.
    method: string or `GeneralIntegrator`
      The integrator method to use. Possible values for string are:
        * symp_euler - see `SymplecticEuler`
        * symp_euler_q - a `SymplecticEuler` with position_first=True
        * symp_euler_p - a `SymplecticEuler` with position_first=False
        * leap_frog - see `LeapFrog`
        * leap_frog_q - a `LeapFrog` with position_first=True
        * leap_frog_p - a `LeapFrog` with position_first=False
        * stormer_verlet - same as leap_frog
        * stormer_verlet_q - same as leap_frog_q
        * stormer_verlet_p - same as leap_frog_p
        * ruth4 - see `Ruth4`,
        * sym4 - see `Symmetric4`
        * sym6 - see `Symmetric6`
        * so4 - see `SymmetricSo4`
        * so4_q - a `SymmetricSo4` with position_first=True
        * so4_p - a `SymmetricSo4` with position_first=False
        * so6 - see `SymmetricSo6`
        * so6_q - a `SymmetricSo6` with position_first=True
        * so6_p - a `SymmetricSo6` with position_first=False
        * so8 - see `SymmetricSo8`
        * so8_q - a `SymmetricSo8` with position_first=True
        * so8_p - a `SymmetricSo8` with position_first=False
    t_eval: array or None.
      Times at which to store the computed solution. Must be sorted and lie
      within `t_span`. If None then t_eval = [t_span[-1]]
    steps_per_dt: int
      This determines the overall step size. Between any two values of t_eval
      the step size is `dt = (t_eval[i+1] - t_eval[i]) / steps_per_dt.
    use_scan: bool
      Whether for the loop to use `lax.scan` or a python loop
    ode_int_kwargs: dict
      Extra argumrnts to be passed to `ode.odeint` when method="adaptive"

  Returns:
    t: array
      Time points at which the solution is evaluated.
    y : an instance of M
      Values of the solution at `t`.
  """
  if not isinstance(y0, phase_space.PhaseSpace):
    raise ValueError("The initial state must be an instance of `PhaseSpace`.")
  dy_dt = phase_space.poisson_bracket_with_q_and_p(hamiltonian)
  if method == "adaptive":
    dy_dt = phase_space.transform_symplectic_tangent_function_using_array(dy_dt)

  return solve_ivp_t_eval(
      fun=dy_dt,
      t_span=t_span,
      y0=y0,
      method=method,
      t_eval=t_eval,
      steps_per_dt=steps_per_dt,
      use_scan=use_scan,
      ode_int_kwargs=ode_int_kwargs
    )


class CompositionSymplectic(SymplecticIntegrator):
  """A generalized symplectic integrator based on compositions.

  Simulates Hamiltonian dynamics using a composition of symplectic steps:
    q_{0} = q_init, p_{0} = p_init
    for i in [1, n]:
      p_{i+1} = p_{i} - c_{i} * dH/dq(q_{i}) * dt
      q_{i+1} = q_{i} + d_{i} * dH/dp(p_{i+1}) * dt
    q_next = q_{n}, p_next = p_{n}

  This integrator always starts with updating the momentum.
  The order argument is used mainly for testing to estimate the error when
  integrating various systems.
  """

  def __init__(
      self,
      momentum_coefficients: Sequence[float],
      position_coefficients: Sequence[float],
      order: int):
    if len(position_coefficients) != len(momentum_coefficients):
      raise ValueError("The number of momentum_coefficients and "
                       "position_coefficients must be the same.")
    if not np.allclose(sum(position_coefficients), 1.0):
      raise ValueError("The sum of the position_coefficients "
                       "must be equal to 1.")
    if not np.allclose(sum(momentum_coefficients), 1.0):
      raise ValueError("The sum of the momentum_coefficients "
                       "must be equal to 1.")
    self.momentum_coefficients = momentum_coefficients
    self.position_coefficients = position_coefficients
    self.order = order

  def __call__(
      self,
      tangent_func: phase_space.SymplecticTangentFunction,
      t: jnp.ndarray,
      y: phase_space.PhaseSpace,
      dt: jnp.ndarray
  ) -> phase_space.PhaseSpace:
    q, p = y.q, y.p
    # This is intentional to prevent a bug where one uses y later
    del y
    # We always broadcast opposite to numpy (e.g. leading dims (batch) count)
    if dt.ndim > 0:
      dt = dt.reshape(dt.shape + (1,) * (q.ndim - dt.ndim))
    if t.ndim > 0:
      t = t.reshape(t.shape + (1,) * (q.ndim - t.ndim))
    t_q = t
    t_p = t
    for c, d in zip(self.momentum_coefficients, self.position_coefficients):
      # Update momentum
      if c != 0.0:
        dp_dt = tangent_func(t_p, phase_space.PhaseSpace(q, p)).p
        p = p + c * dt * dp_dt
        t_p = t_p + c * dt
      # Update position
      if d != 0.0:
        dq_dt = tangent_func(t_q, phase_space.PhaseSpace(q, p)).q
        q = q + d * dt * dq_dt
        t_q = t_q + d * dt
    return phase_space.PhaseSpace(position=q, momentum=p)


class SymplecticEuler(CompositionSymplectic):
  """The symplectic Euler method (for Hamiltonian systems).

  If position_first = True:
    q_{t+1} = q_{t} + dH/dp(p_{t}) * dt
    p_{t+1} = p_{t} - dH/dq(q_{t+1}) * dt
  else:
    p_{t+1} = p_{t} - dH/dq(q_{t}) * dt
    q_{t+1} = q_{t} + dH/dp(p_{t+1}) * dt
  """

  def __init__(self, position_first=True):
    if position_first:
      super().__init__(
          momentum_coefficients=[0.0, 1.0],
          position_coefficients=[1.0, 0.0],
          order=1
      )
    else:
      super().__init__(
          momentum_coefficients=[1.0],
          position_coefficients=[1.0],
          order=1
      )


class SymmetricCompositionSymplectic(CompositionSymplectic):
  """A generalized composition integrator that is symmetric.

  The integrators produced are always of the form:
    [update_q, update_p, ..., update_p, update_q]
  or
    [update_p, update_q, ..., update_q, update_p]
  based on the position_first argument. The method will expect which ever is
  updated first to have one more coefficient.
  """

  def __init__(
      self,
      momentum_coefficients: Sequence[float],
      position_coefficients: Sequence[float],
      position_first: bool,
      order: int):
    position_coefficients = list(position_coefficients)
    momentum_coefficients = list(momentum_coefficients)
    if position_first:
      if len(position_coefficients) != len(momentum_coefficients) + 1:
        raise ValueError("The number of position_coefficients must be one more "
                         "than momentum_coefficients when position_first=True.")
      momentum_coefficients = [0.0] + momentum_coefficients
    else:
      if len(position_coefficients) + 1 != len(momentum_coefficients):
        raise ValueError("The number of momentum_coefficients must be one more "
                         "than position_coefficients when position_first=True.")
      position_coefficients = position_coefficients + [0.0]
    super().__init__(
        position_coefficients=position_coefficients,
        momentum_coefficients=momentum_coefficients,
        order=order
    )


def symmetrize_coefficients(
    coefficients: Sequence[float],
    odd_number: bool
) -> Sequence[float]:
  """Symmetrizes the coefficients for an integrator."""
  coefficients = list(coefficients)
  if odd_number:
    final = 1.0 - 2.0 * sum(coefficients)
    return coefficients + [final] + coefficients[::-1]
  else:
    final = 0.5 - sum(coefficients)
    return coefficients + [final, final] + coefficients[::-1]


class LeapFrog(SymmetricCompositionSymplectic):
  """The standard Leap-Frog method (also known as Stormer-Verlet).

  If position_first = True:
    q_half = q_{t} + dH/dp(p_{t}) * dt / 2
    p_{t+1} = p_{t} - dH/dq(q_half) * dt
    q_{t+1} = q_half + dH/dp(p_{t+1}) * dt / 2
  else:
    p_half = p_{t} - dH/dq(q_{t}) * dt / 2
    q_{t+1} = q_{t} + dH/dp(p_half) * dt
    p_{t+1} = p_half - dH/dq(q_{t+1}) * dt / 2
  """

  def __init__(self, position_first=False):
    if position_first:
      super().__init__(
          position_coefficients=[0.5, 0.5],
          momentum_coefficients=[1.0],
          position_first=True,
          order=2
      )
    else:
      super().__init__(
          position_coefficients=[1.0],
          momentum_coefficients=[0.5, 0.5],
          position_first=False,
          order=2
      )


class Ruth4(SymmetricCompositionSymplectic):
  """The Fourth order method from [2]."""

  def __init__(self):
    cbrt_2 = float(np.cbrt(2.0))

    c = [1.0 / (2.0 - cbrt_2)]
    # 3: [c1, 1.0 - 2*c1, c1]
    c = symmetrize_coefficients(c, odd_number=True)

    d = [1.0 / (4.0 - 2.0 * cbrt_2)]
    # 4: [d1, 0.5 - d1, 0.5 - d1, d1]
    d = symmetrize_coefficients(d, odd_number=False)

    super().__init__(
        position_coefficients=d,
        momentum_coefficients=c,
        position_first=True,
        order=4
    )


class Symmetric4(SymmetricCompositionSymplectic):
  """The fourth order method from Table 6.1 in [1] (originally from [3])."""

  def __init__(self):
    c = [0.0792036964311957, 0.353172906049774, -0.0420650803577195]
    # 7 : [c1, c2, c3, 1.0 - c1 - c2 - c3, c3, c2, c1]
    c = symmetrize_coefficients(c, odd_number=True)

    d = [0.209515106613362, -0.143851773179818]
    # 6: [d1, d2, 0.5 - d1, 0.5 - d1, d2,  d1]
    d = symmetrize_coefficients(d, odd_number=False)

    super().__init__(
        position_coefficients=d,
        momentum_coefficients=c,
        position_first=False,
        order=4
    )


class Symmetric6(SymmetricCompositionSymplectic):
  """The sixth order method from Table 6.1 in [1] (originally from [3])."""

  def __init__(self):
    c = [0.0502627644003922, 0.413514300428344, 0.0450798897943977,
         -0.188054853819569, 0.541960678450780]
    # 11 : [c1, c2, c3, c4, c5, 1.0 - sum(ci), c5, c4, c3, c2, c1]
    c = symmetrize_coefficients(c, odd_number=True)

    d = [0.148816447901042, -0.132385865767784, 0.067307604692185,
         0.432666402578175]
    # 10: [d1, d2, d3, d4, 0.5 - sum(di), 0.5 - sum(di), d4, d3, d2,  d1]
    d = symmetrize_coefficients(d, odd_number=False)

    super().__init__(
        position_coefficients=d,
        momentum_coefficients=c,
        position_first=False,
        order=4
    )


def coefficients_based_on_composing_second_order(
    weights: Sequence[float]
) -> Tuple[Sequence[float], Sequence[float]]:
  """Constructs the coefficients for methods based on second-order schemes."""
  coefficients_0 = []
  coefficients_1 = []
  coefficients_0.append(weights[0] / 2.0)
  for i in range(len(weights) - 1):
    coefficients_1.append(weights[i])
    coefficients_0.append((weights[i] + weights[i + 1]) / 2.0)
  coefficients_1.append(weights[-1])
  coefficients_0.append(weights[-1] / 2.0)
  return coefficients_0, coefficients_1


class SymmetricSo4(SymmetricCompositionSymplectic):
  """The fourth order method from Table 6.2 in [1] (originally from [4])."""

  def __init__(self, position_first: bool = False):
    w = [0.28, 0.62546642846767004501]
    # 5
    w = symmetrize_coefficients(w, odd_number=True)
    c0, c1 = coefficients_based_on_composing_second_order(w)
    c_q, c_p = (c0, c1) if position_first else (c1, c0)
    super().__init__(
        position_coefficients=c_q,
        momentum_coefficients=c_p,
        position_first=position_first,
        order=4
    )


class SymmetricSo6(SymmetricCompositionSymplectic):
  """The sixth order method from Table 6.2 in [1] (originally from [5])."""

  def __init__(self, position_first: bool = False):
    w = [0.78451361047755726382, 0.23557321335935813368,
         -1.17767998417887100695]
    # 7
    w = symmetrize_coefficients(w, odd_number=True)
    c0, c1 = coefficients_based_on_composing_second_order(w)
    c_q, c_p = (c0, c1) if position_first else (c1, c0)
    super().__init__(
        position_coefficients=c_q,
        momentum_coefficients=c_p,
        position_first=position_first,
        order=6
    )


class SymmetricSo8(SymmetricCompositionSymplectic):
  """The eighth order method from Table 6.2 in [1] (originally from [4])."""

  def __init__(self, position_first: bool = False):
    w = [0.74167036435061295345, -0.40910082580003159400,
         0.19075471029623837995, -0.57386247111608226666,
         0.29906418130365592384, 0.33462491824529818378,
         0.31529309239676659663]
    # 15
    w = symmetrize_coefficients(w, odd_number=True)
    c0, c1 = coefficients_based_on_composing_second_order(w)
    c_q, c_p = (c0, c1) if position_first else (c1, c0)
    super().__init__(
        position_coefficients=c_q,
        momentum_coefficients=c_p,
        position_first=position_first,
        order=8
    )


general_integrators = dict(
    general_euler=GeneralEuler(),
    rk2=RungaKutta2(),
    rk4=RungaKutta4(),
    rk38=RungaKutta38()
)

symplectic_integrators = dict(
    symp_euler=SymplecticEuler(position_first=True),
    symp_euler_q=SymplecticEuler(position_first=True),
    symp_euler_p=SymplecticEuler(position_first=False),
    leap_frog=LeapFrog(position_first=False),
    leap_frog_q=LeapFrog(position_first=True),
    leap_frog_p=LeapFrog(position_first=False),
    stormer_verlet=LeapFrog(position_first=False),
    stormer_verlet_q=LeapFrog(position_first=True),
    stormer_verlet_p=LeapFrog(position_first=False),
    ruth4=Ruth4(),
    sym4=Symmetric4(),
    sym6=Symmetric6(),
    so4=SymmetricSo4(position_first=False),
    so4_q=SymmetricSo4(position_first=True),
    so4_p=SymmetricSo4(position_first=False),
    so6=SymmetricSo6(position_first=False),
    so6_q=SymmetricSo6(position_first=True),
    so6_p=SymmetricSo6(position_first=False),
    so8=SymmetricSo8(position_first=False),
    so8_q=SymmetricSo8(position_first=True),
    so8_p=SymmetricSo8(position_first=False),
)


def get_integrator(
    name_or_callable: Union[str, GeneralIntegrator]
) -> GeneralIntegrator:
  """Returns any integrator with the provided name or the argument."""
  if isinstance(name_or_callable, str):
    if name_or_callable in general_integrators:
      return general_integrators[name_or_callable]
    elif name_or_callable in symplectic_integrators:
      return symplectic_integrators[name_or_callable]
    else:
      raise ValueError(f"Unrecognized integrator with name {name_or_callable}.")
  if not callable(name_or_callable):
    raise ValueError(f"Expected a callable, but got {type(name_or_callable)}.")
  return name_or_callable
