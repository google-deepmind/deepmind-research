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
"""Module for all models."""
from typing import Any, Dict, Optional

import physics_inspired_models.models.autoregressive as autoregressive
import physics_inspired_models.models.deterministic_vae as deterministic_vae

_physics_arguments = (
    "input_space", "simulation_space", "potential_func_form",
    "kinetic_func_form", "hgn_kinetic_func_form", "lgn_kinetic_func_form",
    "parametrize_mass_matrix", "hgn_parametrize_mass_matrix",
    "lgn_parametrize_mass_matrix", "mass_eps"
)


def construct_model(
    name: str,
    *args,
    **kwargs: Dict[str, Any]
):
  """Constructs the correct instance of a model given the short name."""
  latent_dynamics_type: Optional[str] = kwargs.pop("latent_dynamics_type", None)  # pytype: disable=annotation-type-mismatch
  latent_system_kwargs = dict(**kwargs.pop("latent_system_kwargs", dict()))
  if name == "AR":
    assert latent_dynamics_type in ("vanilla", "lstm", "gru")
    # This arguments are not part of the AR models
    for k in _physics_arguments + ("integrator_method", "residual"):
      latent_system_kwargs.pop(k, None)
    return autoregressive.TeacherForcingAutoregressiveModel(
        *args,
        latent_dynamics_type=latent_dynamics_type,
        latent_system_kwargs=latent_system_kwargs,
        **kwargs
    )
  elif name == "RGN":
    assert latent_dynamics_type in ("Discrete", None)
    latent_dynamics_type = "Discrete"
    # This arguments are not part of the RGN models
    for k in _physics_arguments + ("integrator_method",):
      latent_system_kwargs.pop(k, None)
  elif name == "ODE":
    assert latent_dynamics_type in ("ODE", None)
    latent_dynamics_type = "ODE"
    # This arguments are not part of the ODE models
    for k in _physics_arguments + ("residual",):
      latent_system_kwargs.pop(k, None)
  elif name == "HGN":
    assert latent_dynamics_type in ("Physics", None)
    latent_dynamics_type = "Physics"
    assert latent_system_kwargs.get("input_space", None) in ("momentum", None)
    latent_system_kwargs["input_space"] = "momentum"
    assert (latent_system_kwargs.get("simulation_space", None)
            in ("momentum", None))
    latent_system_kwargs["simulation_space"] = "momentum"
    # Kinetic func form
    hgn_specific = latent_system_kwargs.pop("hgn_kinetic_func_form", None)
    if hgn_specific is not None:
      latent_system_kwargs["kinetic_func_form"] = hgn_specific
    # Mass matrix
    hgn_specific = latent_system_kwargs.pop("hgn_parametrize_mass_matrix",
                                            None)
    if hgn_specific is not None:
      latent_system_kwargs["parametrize_mass_matrix"] = hgn_specific
    # This arguments are not part of the HGN models
    latent_system_kwargs.pop("residual", None)
    latent_system_kwargs.pop("lgn_kinetic_func_form", None)
    latent_system_kwargs.pop("lgn_parametrize_mass_matrix", None)
  elif name == "LGN":
    assert latent_dynamics_type in ("Physics", None)
    latent_dynamics_type = "Physics"
    assert latent_system_kwargs.get("input_space", None) in ("velocity", None)
    latent_system_kwargs["input_space"] = "velocity"
    assert (latent_system_kwargs.get("simulation_space", None) in
            ("velocity", None))
    latent_system_kwargs["simulation_space"] = "velocity"
    # Kinetic func form
    lgn_specific = latent_system_kwargs.pop("lgn_kinetic_func_form", None)
    if lgn_specific is not None:
      latent_system_kwargs["kinetic_func_form"] = lgn_specific
    # Mass matrix
    lgn_specific = latent_system_kwargs.pop("lgn_parametrize_mass_matrix",
                                            None)
    if lgn_specific is not None:
      latent_system_kwargs["parametrize_mass_matrix"] = lgn_specific
    # This arguments are not part of the HGN models
    latent_system_kwargs.pop("residual", None)
    latent_system_kwargs.pop("hgn_kinetic_func_form", None)
    latent_system_kwargs.pop("hgn_parametrize_mass_matrix", None)
  elif name == "PGN":
    assert latent_dynamics_type in ("Physics", None)
    latent_dynamics_type = "Physics"
    # This arguments are not part of the PGN models
    latent_system_kwargs.pop("residual")
    latent_system_kwargs.pop("hgn_kinetic_func_form", None)
    latent_system_kwargs.pop("hgn_parametrize_mass_matrix", None)
    latent_system_kwargs.pop("lgn_kinetic_func_form", None)
    latent_system_kwargs.pop("lgn_parametrize_mass_matrix", None)
  else:
    raise NotImplementedError()
  return deterministic_vae.DeterministicLatentsGenerativeModel(
      *args,
      latent_dynamics_type=latent_dynamics_type,
      latent_system_kwargs=latent_system_kwargs,
      **kwargs)
