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
"""Module containing all of the networks as Haiku modules."""
from typing import Any, Mapping, Optional, Tuple, Union

from dm_hamiltonian_dynamics_suite.hamiltonian_systems import phase_space
import haiku as hk
import jax
import jax.numpy as jnp

from physics_inspired_models import integrators
from physics_inspired_models import utils
from physics_inspired_models.models import networks

_PhysicsSimulationOutput = Union[
    phase_space.PhaseSpace,
    Tuple[phase_space.PhaseSpace, Mapping[str, jnp.ndarray]]
]


class PhysicsSimulationNetwork(hk.Module):
  """A model for simulating an abstract physical system, whose energy is defined by a neural network."""

  def __init__(
      self,
      system_dim: int,
      input_space: str,
      simulation_space: str,
      potential_func_form: str,
      kinetic_func_form: str,
      parametrize_mass_matrix: bool,
      net_kwargs: Mapping[str, Any],
      mass_eps: float = 1.0,
      integrator_method: Optional[str] = None,
      steps_per_dt: int = 1,
      ode_int_kwargs: Optional[Mapping[str, float]] = None,
      use_scan: bool = True,
      feature_axis: int = -1,
      features_extra_dims: Optional[int] = None,
      network_creation_func=networks.make_flexible_net,
      name: Optional[str] = None
  ):
    """Initializes the model.

    Args:
      system_dim: The number of system dimensions. Note that this specifies the
        number of dimensions only of the position vectors, not of position and
        momentum. Hence the generalized coordinates would be of dimension
        `2 * system_dim`.
      input_space: Either `velocity` or `momentum`. Specifies whether the inputs
        to the model are to be interpreted as `(position, velocity)` or as
        `(position, momentum)`.
      simulation_space: Either `velocity` or `momentum`. Specifies whether the
        model should simulate the dynamics in `(position, velocity)` space
        using the Lagrangian formulation or in `(position, momentum)` space
        using the Hamiltonian formulation. If this is different than the value
        of `input_space` then `kinetic_func_form` must be one of pure_quad,
        matrix_diag_quad, matrix_quad, matrix_dep_diag_quad, matrix_dep_quad.
        In all other cases one can not compute analytically the form of the
        functional (Lagrangian or Hamiltonian) from the other.
      potential_func_form: String specifying the form of the potential energy:
        * separable_net - The network uses only the position:
          U(q, q_dot/p) = f(q)    f: R^d -> R
        * dep_net - The network uses both the position and velocity/momentum:
          U(q, q_dot/p) = f(q, q_dot/p)     f: R^d x R^d -> R
        * embed_quad - A quadratic of the embedding of a network embedding of
          the velocity/momentum:
          U(q, q_dot/p) = f(q)^T f(q) / 2    f: R^d -> R^d
      kinetic_func_form: String specifying the form of the potential energy:
        * separable_net - The network uses only the velocity/momentum:
          K(q, q_dot/p) = f(q_dot/p)    f: R^d -> R
        * dep_net - The network uses both the position and velocity/momentum:
          K(q, q_dot/p) = f(q, q_dot/p)     f: R^d x R^d -> R
        * pure_quad - A quadratic function of the velocity/momentum:
          K(q, q_dot/p) = (q_dot/p)^T (q_dot/p) / 2
        * matrix_diag_quad - A quadratic function of the velocity/momentum,
          where there is diagonal mass matrix, whose log `P` is a parameter:
          K(q, q_dot) = q_dot^T M q_dot / 2
          K(q, p) = p^T M^-1 p / 2
          [if `parameterize_mass_matrix`]
          M = diag(exp(P) + mass_eps)
          [else]
          M^-1 = diag(exp(P) + mass_eps)
        * matrix_quad - A quadratic function of the velocity/momentum, where
          there is a full mass matrix, whose Cholesky factor L is a parameter:
          K(q, q_dot) = q_dot^T M q_dot / 2
          K(q, p) = p^T M^-1 p / 2
          [if `parameterize_mass_matrix`]
          M = LL^T + mass_eps * I
          [else]
          M^-1 = LL^T + mass_eps * I
        * matrix_dep_quad - A quadratic function of the velocity/momentum, where
          there is a full mass matrix defined as a function of the position:
          K(q, q_dot) = q_dot^T M(q) q_dot / 2
          K(q, p) = p^T M(q)^-1 p / 2
          [if `parameterize_mass_matrix`]
          M(q) = g(q) g(q)^T + mass_eps * I    g: R^d -> R^(d(d+1)/2)
          [else]
          M(q)^-1 = g(q) g(q)^T + mass_eps * I    g: R^d -> R^(d(d+1)/2)
        * embed_quad - A quadratic of the embedding of a network embedding of
          the velocity/momentum:
          K(q, q_dot/p) = f(q_dot/p)^T f(q_dot/p) / 2    f: R^d -> R^d
        * matrix_dep_diag_embed_quad - A quadratic of the embedding of a network
          embedding of the velocity/momentum where there is diagonal mass matrix
          defined as a function of the position:
          K(q, q_dot) = f(q_dot)^T M(q) f(q_dot) / 2    f: R^d -> R^d
          K(q, p) = f(p)^T M(q)^-1 f(p) / 2    f: R^d -> R^d
          [if `parameterize_mass_matrix`]
          M(q) = diag(exp(g(q)) + mass_eps * I    g: R^d -> R^d
          [else]
          M(q)^-1 = diag(exp(g(q)) + mass_eps * I    g: R^d -> R^d
        * matrix_dep_embed_quad - A quadratic of the embedding of a network
          embedding of the velocity/momentum where there is a full mass matrix
          defined as a function of the position:
          K(q, q_dot) = f(q_dot)^T M(q) f(q_dot) / 2    f: R^d -> R^d
          K(q, p) = f(p)^T M(q)^-1 f(p) / 2    f: R^d -> R^d
          [if `parameterize_mass_matrix`]
          M(q) = g(q) g(q)^T + mass_eps * I    g: R^d -> R^(d(d+1)/2)
          [else]
          M(q)^-1 = g(q) g(q)^T + mass_eps * I    g: R^d -> R^(d(d+1)/2)
        For any of the function forms with mass matrices, if we have a
          convolutional input it is assumed that the matrix is shared across all
          spatial locations.
      parametrize_mass_matrix: Defines for the kinetic functional form, whether
        the network output defines the mass or the inverse of the mass matrix.
      net_kwargs: Any keyword arguments to pass down to the networks.
      mass_eps: The additional weight of the identity added to the mass matrix,
        when relevant.
      integrator_method: What method to use for integrating the system.
      steps_per_dt: How many internal steps per a single `dt` step to do.
      ode_int_kwargs: Extra arguments when using "implicit" integrator method.
      use_scan: Whether to use `lax.scan` for explicit integrators.
      feature_axis: The number of the features axis in the inputs.
      features_extra_dims: If the inputs have extra features (like spatial for
        convolutions) this specifies how many of them there are.
      network_creation_func: A function that creates the networks. Should have a
        signature `network_creation_func(output_dims, name, **net_kwargs)`.
      name: The name of this Haiku module.
    """
    super().__init__(name=name)
    if input_space not in ("velocity", "momentum"):
      raise ValueError("input_space must be either velocity or momentum.")
    if simulation_space not in ("velocity", "momentum"):
      raise ValueError("simulation_space must be either velocity or momentum.")
    if potential_func_form not in ("separable_net", "dep_net", "embed_quad"):
      raise ValueError("The potential network can be only a network.")
    if kinetic_func_form not in ("separable_net", "dep_net", "pure_quad",
                                 "matrix_diag_quad", "matrix_quad",
                                 "matrix_dep_diag_quad", "matrix_dep_quad",
                                 "embed_quad", "matrix_dep_diag_embed_quad",
                                 "matrix_dep_embed_quad"):
      raise ValueError(f"Unrecognized kinetic func form {kinetic_func_form}.")
    if input_space != simulation_space:
      if kinetic_func_form not in (
          "pure_quad", "matrix_diag_quad", "matrix_quad",
          "matrix_dep_diag_quad", "matrix_dep_quad"):
        raise ValueError(
            "When the input and simulation space are not the same, it is "
            "possible to simulate the physical system only if kinetic_func_form"
            " is one of pure_quad, matrix_diag_quad, matrix_quad, "
            "matrix_dep_diag_quad, matrix_dep_quad. In all other cases one can"
            "not compute analytically the form of the functional (Lagrangian or"
            " Hamiltonian) from the other.")
    if feature_axis != -1:
      raise ValueError("Currently we only support features_axis=-1.")
    if integrator_method is None:
      if simulation_space == "velocity":
        integrator_method = "rk2"
      else:
        integrator_method = "leap_frog"
    if features_extra_dims is None:
      if net_kwargs["net_type"] == "mlp":
        features_extra_dims = 0
      elif net_kwargs["net_type"] == "conv":
        features_extra_dims = 2
      else:
        raise NotImplementedError()
    ode_int_kwargs = dict(ode_int_kwargs or {})
    ode_int_kwargs.setdefault("rtol", 1e-6)
    ode_int_kwargs.setdefault("atol", 1e-6)
    ode_int_kwargs.setdefault("mxstep", 50)

    self.system_dim = system_dim
    self.input_space = input_space
    self.simulation_space = simulation_space
    self.potential_func_form = potential_func_form
    self.kinetic_func_form = kinetic_func_form
    self.parametrize_mass_matrix = parametrize_mass_matrix
    self.features_axis = feature_axis
    self.features_extra_dims = features_extra_dims
    self.integrator_method = integrator_method
    self.steps_per_dt = steps_per_dt
    self.ode_int_kwargs = ode_int_kwargs
    self.net_kwargs = net_kwargs
    self.mass_eps = mass_eps
    self.use_scan = use_scan
    self.name = name

    self.potential_net = network_creation_func(
        output_dims=1, name="PotentialNet", **net_kwargs)

    if kinetic_func_form in ("separable_net", "dep_net"):
      self.kinetic_net = network_creation_func(
          output_dims=1, name="KineticNet", **net_kwargs)
    else:
      self.kinetic_net = None
    if kinetic_func_form in ("matrix_dep_quad", "matrix_dep_embed_quad"):
      output_dims = (system_dim * (system_dim + 1)) // 2
      name = "MatrixNet" if parametrize_mass_matrix else "InvMatrixNet"
      self.mass_matrix_net = network_creation_func(
          output_dims=output_dims, name=name, **net_kwargs)
    elif kinetic_func_form in ("matrix_dep_diag_quad",
                               "matrix_dep_diag_embed_quad",
                               "matrix_dep_embed_quad"):
      name = "MatrixNet" if parametrize_mass_matrix else "InvMatrixNet"
      self.mass_matrix_net = network_creation_func(
          output_dims=system_dim, name=name, **net_kwargs)
    else:
      self.mass_matrix_net = None
    if kinetic_func_form in ("embed_quad", "matrix_dep_diag_embed_quad",
                             "matrix_dep_embed_quad"):
      self.kinetic_embed_net = network_creation_func(
          output_dims=system_dim, name="KineticEmbed", **net_kwargs)
    else:
      self.kinetic_embed_net = None

  def sum_per_dim_energy(self, energy: jnp.ndarray) -> jnp.ndarray:
    """Sums the per dimension energy."""
    axis = [-i-1 for i in range(self.features_extra_dims + 1)]
    return jnp.sum(energy, axis=axis)

  def feature_matrix_vector(self, m, v):
    """A utility function to compute the product of a matrix and vector in the features axis."""
    v = jnp.expand_dims(v, axis=self.features_axis-1)
    return jnp.sum(m * v, axis=self.features_axis)

  def mass_matrix_mul(
      self,
      q: jnp.ndarray,
      v: jnp.ndarray,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the product of the mass matrix with a vector and throws an error if not applicable."""
    if self.kinetic_func_form in ("separable_net", "dep_net"):
      raise ValueError("It is not possible to compute `M q_dot` when using a "
                       "network for the kinetic energy.")
    if self.kinetic_func_form in ("pure_quad", "embed_quad"):
      return v
    if self.kinetic_func_form == "matrix_diag_quad":
      if self.parametrize_mass_matrix:
        m_diag_log = hk.get_parameter("MassMatrixDiagLog",
                                      shape=[self.system_dim],
                                      init=hk.initializers.Constant(0.0))
        m_diag = jnp.exp(m_diag_log) + self.mass_eps
      else:
        m_inv_diag_log = hk.get_parameter("InvMassMatrixDiagLog",
                                          shape=[self.system_dim],
                                          init=hk.initializers.Constant(0.0))
        m_diag = 1.0 / (jnp.exp(m_inv_diag_log) + self.mass_eps)
      return m_diag * v
    if self.kinetic_func_form == "matrix_quad":
      if self.parametrize_mass_matrix:
        m_triu = hk.get_parameter("MassMatrixU",
                                  shape=[self.system_dim, self.system_dim],
                                  init=hk.initializers.Identity())
        m_triu = jnp.triu(m_triu)
        m = jnp.matmul(m_triu.T, m_triu)
        m = m + self.mass_eps * jnp.eye(self.system_dim)
        return self.feature_matrix_vector(m, v)
      else:
        m_inv_triu = hk.get_parameter("InvMassMatrixU",
                                      shape=[self.system_dim, self.system_dim],
                                      init=hk.initializers.Identity())
        m_inv_triu = jnp.triu(m_inv_triu)
        m_inv = jnp.matmul(m_inv_triu.T, m_inv_triu)
        m_inv = m_inv + self.mass_eps * jnp.eye(self.system_dim)
        solve = jnp.linalg.solve
        for _ in range(v.ndim + 1 - m_inv.ndim):
          solve = jax.vmap(solve, in_axes=(None, 0))
        return solve(m_inv, v)
    if self.kinetic_func_form in ("matrix_dep_diag_quad",
                                  "matrix_dep_diag_embed_quad"):
      if self.parametrize_mass_matrix:
        m_diag_log = self.mass_matrix_net(q, **kwargs)
        m_diag = jnp.exp(m_diag_log) + self.mass_eps
      else:
        m_inv_diag_log = self.mass_matrix_net(q, **kwargs)
        m_diag = 1.0 / (jnp.exp(m_inv_diag_log) + self.mass_eps)
      return m_diag * v
    if self.kinetic_func_form in ("matrix_dep_quad",
                                  "matrix_dep_embed_quad"):
      if self.parametrize_mass_matrix:
        m_triu = self.mass_matrix_net(q, **kwargs)
        m_triu = utils.triu_matrix_from_v(m_triu, self.system_dim)
        m = jnp.matmul(jnp.swapaxes(m_triu, -1, -2), m_triu)
        m = m + self.mass_eps * jnp.eye(self.system_dim)
        return self.feature_matrix_vector(m, v)
      else:
        m_inv_triu = self.mass_matrix_net(q, **kwargs)
        m_inv_triu = utils.triu_matrix_from_v(m_inv_triu, self.system_dim)
        m_inv = jnp.matmul(jnp.swapaxes(m_inv_triu, -1, -2), m_inv_triu)
        m_inv = m_inv + self.mass_eps * jnp.eye(self.system_dim)
        return jnp.linalg.solve(m_inv, v)
    raise NotImplementedError()

  def mass_matrix_inv_mul(
      self,
      q: jnp.ndarray,
      v: jnp.ndarray,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the product of the inverse mass matrix with a vector."""
    if self.kinetic_func_form in ("separable_net", "dep_net"):
      raise ValueError("It is not possible to compute `M^-1 p` when using a "
                       "network for the kinetic energy.")
    if self.kinetic_func_form in ("pure_quad", "embed_quad"):
      return v
    if self.kinetic_func_form == "matrix_diag_quad":
      if self.parametrize_mass_matrix:
        m_diag_log = hk.get_parameter("MassMatrixDiagLog",
                                      shape=[self.system_dim],
                                      init=hk.initializers.Constant(0.0))
        m_inv_diag = 1.0 / (jnp.exp(m_diag_log) + self.mass_eps)
      else:
        m_inv_diag_log = hk.get_parameter("InvMassMatrixDiagLog",
                                          shape=[self.system_dim],
                                          init=hk.initializers.Constant(0.0))
        m_inv_diag = jnp.exp(m_inv_diag_log) + self.mass_eps
      return m_inv_diag * v
    if self.kinetic_func_form == "matrix_quad":
      if self.parametrize_mass_matrix:
        m_triu = hk.get_parameter("MassMatrixU",
                                  shape=[self.system_dim, self.system_dim],
                                  init=hk.initializers.Identity())
        m_triu = jnp.triu(m_triu)
        m = jnp.matmul(m_triu.T, m_triu)
        m = m + self.mass_eps * jnp.eye(self.system_dim)
        solve = jnp.linalg.solve
        for _ in range(v.ndim + 1 - m.ndim):
          solve = jax.vmap(solve, in_axes=(None, 0))
        return solve(m, v)
      else:
        m_inv_triu = hk.get_parameter("InvMassMatrixU",
                                      shape=[self.system_dim, self.system_dim],
                                      init=hk.initializers.Identity())
        m_inv_triu = jnp.triu(m_inv_triu)
        m_inv = jnp.matmul(m_inv_triu.T, m_inv_triu)
        m_inv = m_inv + self.mass_eps * jnp.eye(self.system_dim)
        return self.feature_matrix_vector(m_inv, v)
    if self.kinetic_func_form in ("matrix_dep_diag_quad",
                                  "matrix_dep_diag_embed_quad"):
      if self.parametrize_mass_matrix:
        m_diag_log = self.mass_matrix_net(q, **kwargs)
        m_inv_diag = 1.0 / (jnp.exp(m_diag_log) + self.mass_eps)
      else:
        m_inv_diag_log = self.mass_matrix_net(q, **kwargs)
        m_inv_diag = jnp.exp(m_inv_diag_log) + self.mass_eps
      return m_inv_diag * v
    if self.kinetic_func_form in ("matrix_dep_quad",
                                  "matrix_dep_embed_quad"):
      if self.parametrize_mass_matrix:
        m_triu = self.mass_matrix_net(q, **kwargs)
        m_triu = utils.triu_matrix_from_v(m_triu, self.system_dim)
        m = jnp.matmul(jnp.swapaxes(m_triu, -2, -1), m_triu)
        m = m + self.mass_eps * jnp.eye(self.system_dim)
        return jnp.linalg.solve(m, v)
      else:
        m_inv_triu = self.mass_matrix_net(q, **kwargs)
        m_inv_triu = utils.triu_matrix_from_v(m_inv_triu, self.system_dim)
        m_inv = jnp.matmul(jnp.swapaxes(m_inv_triu, -2, -1), m_inv_triu)
        m_inv = m_inv + self.mass_eps * jnp.eye(self.system_dim)
        return self.feature_matrix_vector(m_inv, v)
    raise NotImplementedError()

  def momentum_from_velocity(
      self,
      q: jnp.ndarray,
      q_dot: jnp.ndarray,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the momentum from position and velocity."""
    def local_lagrangian(q_dot_):
      # We take the sum so we can easily take gradients
      return jnp.sum(self.lagrangian(
          phase_space.PhaseSpace(q, q_dot_), **kwargs))
    return jax.grad(local_lagrangian)(q_dot)

  def velocity_from_momentum(
      self,
      q: jnp.ndarray,
      p: jnp.ndarray,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the velocity from position and momentum."""
    def local_hamiltonian(p_):
      # We take the sum so we can easily take gradients
      return jnp.sum(self.hamiltonian(
          phase_space.PhaseSpace(q, p_), **kwargs))
    return jax.grad(local_hamiltonian)(p)

  def kinetic_energy_velocity(
      self,
      q: jnp.ndarray,
      q_dot: jnp.ndarray,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the kinetic energy in velocity coordinates."""
    if self.kinetic_func_form in ("separable_net", "dep_net"):
      if self.input_space != "velocity":
        raise ValueError("Can not evaluate the Kinetic energy from velocity, "
                         "when the input space is momentum and "
                         "kinetic_func_form is separable_net or dep_net.")
      if self.kinetic_func_form == "separable_net":
        s = q_dot
      else:
        s = jnp.concatenate([q, q_dot], axis=-1)
      per_dim_energy = self.kinetic_net(s, **kwargs)
    else:
      if self.kinetic_embed_net is not None:
        if self.input_space != "velocity":
          raise ValueError("Can not evaluate the Kinetic energy from velocity, "
                           "when the input space is momentum and "
                           "kinetic_func_form is embed_quad, "
                           "matrix_dep_diag_embed_quad or "
                           "matrix_dep_embed_quad.")
        q_dot = self.kinetic_embed_net(q_dot, **kwargs)
      m_q_dot = self.mass_matrix_mul(q, q_dot, **kwargs)
      per_dim_energy = q_dot * m_q_dot / 2

    return self.sum_per_dim_energy(per_dim_energy)

  def kinetic_energy_momentum(
      self,
      q: jnp.ndarray,
      p: jnp.ndarray,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the kinetic energy in momentum coordinates."""
    if self.kinetic_func_form in ("separable_net", "dep_net"):
      if self.input_space != "momentum":
        raise ValueError("Can not evaluate the Kinetic energy from momentum, "
                         "when the input space is velocity and "
                         "kinetic_func_form is separable_net or dep_net.")
      if self.kinetic_func_form == "separable_net":
        s = p
      else:
        s = jnp.concatenate([q, p], axis=-1)
      per_dim_energy = self.kinetic_net(s, **kwargs)
    else:
      if self.kinetic_embed_net is not None:
        if self.input_space != "momentum":
          raise ValueError("Can not evaluate the Kinetic energy from momentum, "
                           "when the input space is velocity and "
                           "kinetic_func_form is embed_quad, "
                           "matrix_dep_diag_embed_quad or "
                           "matrix_dep_embed_quad.")
        p = self.kinetic_embed_net(p, **kwargs)
      m_inv_p = self.mass_matrix_inv_mul(q, p, **kwargs)
      per_dim_energy = p * m_inv_p / 2

    return self.sum_per_dim_energy(per_dim_energy)

  def potential_energy_velocity(
      self,
      q: jnp.ndarray,
      q_dot: jnp.ndarray,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the potential energy in velocity coordinates."""
    if self.potential_func_form == "separable_net":
      per_dim_energy = self.potential_net(q, **kwargs)
    elif self.input_space != "momentum":
      raise ValueError("Can not evaluate the Potential energy from velocity, "
                       "when the input space is momentum and "
                       "potential_func_form is dep_net.")
    else:
      s = jnp.concatenate([q, q_dot], axis=-1)
      per_dim_energy = self.potential_net(s, **kwargs)
    return self.sum_per_dim_energy(per_dim_energy)

  def potential_energy_momentum(
      self,
      q: jnp.ndarray,
      p: jnp.ndarray,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the potential energy in momentum coordinates."""
    if self.potential_func_form == "separable_net":
      per_dim_energy = self.potential_net(q, **kwargs)
    elif self.input_space != "momentum":
      raise ValueError("Can not evaluate the Potential energy from momentum, "
                       "when the input space is velocity and "
                       "potential_func_form is dep_net.")
    else:
      s = jnp.concatenate([q, p], axis=-1)
      per_dim_energy = self.potential_net(s, **kwargs)
    return self.sum_per_dim_energy(per_dim_energy)

  def hamiltonian(
      self,
      s: phase_space.PhaseSpace,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the Hamiltonian in momentum coordinates."""
    potential = self.potential_energy_momentum(s.q, s.p, **kwargs)
    kinetic = self.kinetic_energy_momentum(s.q, s.p, **kwargs)
    # Sanity check
    assert potential.shape == kinetic.shape
    return kinetic + potential

  def lagrangian(
      self,
      s: phase_space.PhaseSpace,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the Lagrangian in velocity coordinates."""
    potential = self.potential_energy_velocity(s.q, s.p, **kwargs)
    kinetic = self.kinetic_energy_velocity(s.q, s.p, **kwargs)
    # Sanity check
    assert potential.shape == kinetic.shape
    return kinetic - potential

  def energy_from_momentum(
      self,
      s: phase_space.PhaseSpace,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the energy of the system in momentum coordinates."""
    return self.hamiltonian(s, **kwargs)

  def energy_from_velocity(
      self,
      s: phase_space.PhaseSpace,
      **kwargs
  ) -> jnp.ndarray:
    """Computes the energy of the system in velocity coordinates."""
    q, q_dot = s.q, s.p
    p = self.momentum_from_velocity(q, q_dot, **kwargs)
    q_dot_p = jnp.sum(q_dot * p, self.features_axis)
    return q_dot_p - self.lagrangian(s, **kwargs)

  def velocity_and_acceleration(
      self,
      q: jnp.ndarray,
      q_dot: jnp.ndarray,
      **kwargs
  ) -> phase_space.TangentPhaseSpace:
    """Computes the velocity and acceleration of the system in velocity coordinates."""
    def local_lagrangian(*q_and_q_dot):
      # We take the sum so we can easily take gradients
      return jnp.sum(self.lagrangian(
          phase_space.PhaseSpace(*q_and_q_dot), **kwargs))

    grad_q = jax.grad(local_lagrangian, 0)(q, q_dot)
    grad_q_dot_func = jax.grad(local_lagrangian, 1)
    _, grad_q_dot_grad_q_times_q_dot = jax.jvp(grad_q_dot_func, (q, q_dot),
                                               (q_dot, jnp.zeros_like(q_dot)))
    pre_acc_vector = grad_q - grad_q_dot_grad_q_times_q_dot
    if self.kinetic_func_form in ("pure_quad", "matrix_diag_quad",
                                  "matrix_quad", "matrix_dep_diag_quad",
                                  "matrix_dep_quad"):
      q_dot_dot = self.mass_matrix_inv_mul(q, pre_acc_vector, **kwargs)
    else:
      hess_q_dot = jax.vmap(jax.hessian(local_lagrangian, 1))(q, q_dot)
      q_dot_dot = jnp.linalg.solve(hess_q_dot, pre_acc_vector)
    return phase_space.TangentPhaseSpace(q_dot, q_dot_dot)

  def simulate(
      self,
      y0: phase_space.PhaseSpace,
      dt: Union[float, jnp.ndarray],
      num_steps_forward: int,
      num_steps_backward: int,
      include_y0: bool,
      return_stats: bool = True,
      **nets_kwargs
  ) -> _PhysicsSimulationOutput:
    """Simulates the continuous dynamics of the physical system.

    Args:
      y0: Initial state of the system.
      dt: The size of the time intervals at which to evolve the system.
      num_steps_forward: Number of steps to make into the future.
      num_steps_backward: Number of steps to make into the past.
      include_y0: Whether to include the initial state in the result.
      return_stats: Whether to return additional statistics.
      **nets_kwargs: Keyword arguments to pass to the networks.

    Returns:
      * The state of the system evolved as many steps as specified by the
      arguments into the past and future, all in chronological order.
      * Optionally return a dictionary of additional statistics. For the moment
        this only returns the energy of the system at each evaluation point.
    """
    # Define the dynamics
    if self.simulation_space == "velocity":
      dy_dt = lambda t_, y: self.velocity_and_acceleration(  # pylint: disable=g-long-lambda
          y.q, y.p, **nets_kwargs)
      # Special Haiku magic to avoid tracer issues
      if hk.running_init():
        return self.lagrangian(y0, **nets_kwargs)
    else:
      hamiltonian = lambda t_, y: self.hamiltonian(y, **nets_kwargs)
      dy_dt = phase_space.poisson_bracket_with_q_and_p(hamiltonian)
      if hk.running_init():
        return self.hamiltonian(y0, **nets_kwargs)

    # Optionally switch coordinate frame
    if self.input_space == "velocity" and self.simulation_space == "momentum":
      p = self.momentum_from_velocity(y0.q, y0.p, **nets_kwargs)
      y0 = phase_space.PhaseSpace(y0.q, p)
    if self.input_space == "momentum" and self.simulation_space == "velocity":
      q_dot = self.velocity_from_momentum(y0.q, y0.p, **nets_kwargs)
      y0 = phase_space.PhaseSpace(y0.q, q_dot)

    yt = integrators.solve_ivp_dt_two_directions(
        fun=dy_dt,
        y0=y0,
        t0=0.0,
        dt=dt,
        method=self.integrator_method,
        num_steps_forward=num_steps_forward,
        num_steps_backward=num_steps_backward,
        include_y0=include_y0,
        steps_per_dt=self.steps_per_dt,
        ode_int_kwargs=self.ode_int_kwargs
    )
    # Make time axis second
    yt = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), yt)

    # Compute energies for the full trajectory
    yt_energy = jax.tree_map(utils.merge_first_dims, yt)
    if self.simulation_space == "momentum":
      energy = self.energy_from_momentum(yt_energy, **nets_kwargs)
    else:
      energy = self.energy_from_velocity(yt_energy, **nets_kwargs)
    energy = energy.reshape(yt.q.shape[:2])

    # Optionally switch back to input coordinate frame
    if self.input_space == "velocity" and self.simulation_space == "momentum":
      q_dot = self.velocity_from_momentum(yt.q, yt.p, **nets_kwargs)
      yt = phase_space.PhaseSpace(yt.q, q_dot)
    if self.input_space == "momentum" and self.simulation_space == "velocity":
      p = self.momentum_from_velocity(yt.q, yt.p, **nets_kwargs)
      yt = phase_space.PhaseSpace(yt.q, p)

    # Compute energy deficit
    t = energy.shape[-1]
    non_zero_diffs = float((t * (t - 1)) // 2)
    energy_deficits = jnp.abs(energy[..., None, :] - energy[..., None])
    avg_deficit = jnp.sum(energy_deficits, axis=(-2, -1)) / non_zero_diffs
    max_deficit = jnp.max(energy_deficits)

    # Return the states and energies
    if return_stats:
      return yt, dict(avg_energy_deficit=avg_deficit,
                      max_energy_deficit=max_deficit)
    else:
      return yt

  def __call__(self, *args, **kwargs):
    return self.simulate(*args, **kwargs)


class OdeNetwork(hk.Module):
  """A simple haiku module for constructing a NeuralODE."""

  def __init__(
      self,
      system_dim: int,
      net_kwargs: Mapping[str, Any],
      integrator_method: Optional[str] = None,
      steps_per_dt: int = 1,
      ode_int_kwargs: Optional[Mapping[str, float]] = None,
      use_scan: bool = True,
      network_creation_func=networks.make_flexible_net,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    ode_int_kwargs = dict(ode_int_kwargs or {})
    ode_int_kwargs.setdefault("rtol", 1e-6)
    ode_int_kwargs.setdefault("atol", 1e-6)
    ode_int_kwargs.setdefault("mxstep", 50)

    self.system_dim = system_dim
    self.integrator_method = integrator_method or "adaptive"
    self.steps_per_dt = steps_per_dt
    self.ode_int_kwargs = ode_int_kwargs
    self.net_kwargs = net_kwargs
    self.use_scan = use_scan

    self.core = network_creation_func(
        output_dims=system_dim, name="Net", **net_kwargs)

  def simulate(
      self,
      y0: jnp.ndarray,
      dt: Union[float, jnp.ndarray],
      num_steps_forward: int,
      num_steps_backward: int,
      include_y0: bool,
      return_stats: bool = True,
      **nets_kwargs
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]]:
    """Simulates the continuous dynamics of the ODE specified by the network.

    Args:
      y0: Initial state of the system.
      dt: The size of the time intervals at which to evolve the system.
      num_steps_forward: Number of steps to make into the future.
      num_steps_backward: Number of steps to make into the past.
      include_y0: Whether to include the initial state in the result.
      return_stats: Whether to return additional statistics.
      **nets_kwargs: Keyword arguments to pass to the networks.

    Returns:
      * The state of the system evolved as many steps as specified by the
      arguments into the past and future, all in chronological order.
      * Optionally return a dictionary of additional statistics. For the moment
        this is just an empty dictionary.
    """
    if hk.running_init():
      return self.core(y0, **nets_kwargs)
    yt = integrators.solve_ivp_dt_two_directions(
        fun=lambda t, y: self.core(y, **nets_kwargs),
        y0=y0,
        t0=0.0,
        dt=dt,
        method=self.integrator_method,
        num_steps_forward=num_steps_forward,
        num_steps_backward=num_steps_backward,
        include_y0=include_y0,
        steps_per_dt=self.steps_per_dt,
        ode_int_kwargs=self.ode_int_kwargs
    )
    # Make time axis second
    yt = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), yt)
    if return_stats:
      return yt, dict()
    else:
      return yt

  def __call__(self, *args, **kwargs):
    return self.simulate(*args, **kwargs)


class DiscreteDynamicsNetwork(hk.Module):
  """A simple haiku module for constructing a discrete dynamics network."""

  def __init__(
      self,
      system_dim: int,
      residual: bool,
      net_kwargs: Mapping[str, Any],
      use_scan: bool = True,
      network_creation_func=networks.make_flexible_net,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.system_dim = system_dim
    self.residual = residual
    self.net_kwargs = net_kwargs
    self.use_scan = use_scan
    self.core = network_creation_func(
        output_dims=system_dim, name="Net", **net_kwargs)

  def simulate(
      self,
      y0: jnp.ndarray,
      num_steps_forward: int,
      include_y0: bool,
      return_stats: bool = True,
      **nets_kwargs
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]]:
    """Simulates the dynamics of the discrete system.

    Args:
      y0: Initial state of the system.
      num_steps_forward: Number of steps to make into the future.
      include_y0: Whether to include the initial state in the result.
      return_stats: Whether to return additional statistics.
      **nets_kwargs: Keyword arguments to pass to the networks.

    Returns:
      * The state of the system evolved as many steps as specified by the
      arguments into the past and future, all in chronological order.
      * Optionally return a dictionary of additional statistics. For the moment
        this is just an empty dictionary.
    """
    if num_steps_forward < 0:
      raise ValueError("It is required to unroll at least one step.")
    nets_kwargs.pop("dt", None)
    nets_kwargs.pop("num_steps_backward", None)
    if hk.running_init():
      return self.core(y0, **nets_kwargs)

    def step(*args):
      y, _ = args
      if self.residual:
        y_next = y + self.core(y, **nets_kwargs)
      else:
        y_next = self.core(y, **nets_kwargs)
      return y_next, y_next

    if self.use_scan:
      _, yt = jax.lax.scan(step, init=y0, xs=None, length=num_steps_forward)
      if include_y0:
        yt = jnp.concatenate([y0[None], yt], axis=0)
      # Make time axis second
      yt = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), yt)
    else:
      yt = [y0]
      for _ in range(num_steps_forward):
        yt.append(step(yt[-1], None)[0])
      if not include_y0:
        yt = yt[1:]
      if len(yt) == 1:
        yt = yt[0][:, None]
      else:
        yt = jax.tree_multimap(lambda args: jnp.stack(args, 1), yt)
    if return_stats:
      return yt, dict()
    else:
      return yt

  def __call__(self, *args, **kwargs):
    return self.simulate(*args, **kwargs)
