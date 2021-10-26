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
"""Module containing the main models code."""
import functools
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import distrax
from dm_hamiltonian_dynamics_suite.hamiltonian_systems import phase_space
import haiku as hk
import jax.numpy as jnp
import jax.random as jnr
import numpy as np

from physics_inspired_models import metrics
from physics_inspired_models import utils
from physics_inspired_models.models import base
from physics_inspired_models.models import dynamics

_ArrayOrPhase = Union[jnp.ndarray, phase_space.PhaseSpace]


class DeterministicLatentsGenerativeModel(base.SequenceModel[_ArrayOrPhase]):
  """Common class for generative models with deterministic latent dynamics."""

  def __init__(
      self,
      latent_system_dim: int,
      latent_system_net_type: str,
      latent_system_kwargs: Dict[str, Any],
      latent_dynamics_type: str,
      encoder_aggregation_type: Optional[str],
      decoder_de_aggregation_type: Optional[str],
      encoder_kwargs: Dict[str, Any],
      decoder_kwargs: Dict[str, Any],
      num_inference_steps: int,
      num_target_steps: int,
      latent_training_type: str,
      training_data_split: str,
      objective_type: str,
      dt: float = 0.125,
      render_from_q_only: bool = True,
      prior_type: str = "standard_normal",
      use_analytical_kl: bool = True,
      geco_kappa: float = 0.001,
      geco_alpha: Optional[float] = 0.0,
      elbo_beta_delay: int = 0,
      elbo_beta_final: float = 1.0,
      name: Optional[str] = None,
      **kwargs
  ):
    can_run_backwards = latent_dynamics_type in ("ODE", "Physics")

    # Verify arguments
    if objective_type not in ("GECO", "ELBO", "NON-PROB"):
      raise ValueError(f"Unrecognized training type - {objective_type}")
    if geco_alpha is None:
      geco_alpha = 0
    if geco_alpha < 0 or geco_alpha >= 1:
      raise ValueError("GECO alpha parameter must be in [0, 1).")
    if prior_type not in ("standard_normal", "made", "made_gated"):
      raise ValueError(f"Unrecognized prior_type='{prior_type}.")
    if (latent_training_type == "forward_backward" and
        training_data_split != "include_inference"):
      raise ValueError("Training forward_backward works only when "
                       "training_data_split=include_inference.")
    if (latent_training_type == "forward_backward" and
        num_inference_steps % 2 == 0):
      raise ValueError("Training forward_backward works only when "
                       "num_inference_steps are odd.")
    if latent_training_type == "forward_backward" and not can_run_backwards:
      raise ValueError("Training forward_backward works only when the model can"
                       " be run backwards.")
    if prior_type != "standard_normal":
      raise ValueError("For now we support only `standard_normal`.")

    super().__init__(
        can_run_backwards=can_run_backwards,
        latent_system_dim=latent_system_dim,
        latent_system_net_type=latent_system_net_type,
        latent_system_kwargs=latent_system_kwargs,
        encoder_aggregation_type=encoder_aggregation_type,
        decoder_de_aggregation_type=decoder_de_aggregation_type,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        num_inference_steps=num_inference_steps,
        num_target_steps=num_target_steps,
        name=name,
        **kwargs
    )
    # VAE specific arguments
    self.prior_type = prior_type
    self.objective_type = objective_type
    self.use_analytical_kl = use_analytical_kl
    self.geco_kappa = geco_kappa
    self.geco_alpha = geco_alpha
    self.elbo_beta_delay = elbo_beta_delay
    self.elbo_beta_final = jnp.asarray(elbo_beta_final)

    # The dynamics module and arguments
    self.latent_dynamics_type = latent_dynamics_type
    self.latent_training_type = latent_training_type
    self.training_data_split = training_data_split
    self.dt = dt
    self.render_from_q_only = render_from_q_only
    latent_system_kwargs["net_kwargs"] = dict(
        latent_system_kwargs["net_kwargs"])
    latent_system_kwargs["net_kwargs"]["net_type"] = self.latent_system_net_type

    if self.latent_dynamics_type == "Physics":
      # Note that here system_dim means the dimensionality of `q` and `p`.
      model_constructor = functools.partial(
          dynamics.PhysicsSimulationNetwork,
          system_dim=self.latent_system_dim // 2,
          name="Physics",
          **latent_system_kwargs
      )
    elif self.latent_dynamics_type == "ODE":
      model_constructor = functools.partial(
          dynamics.OdeNetwork,
          system_dim=self.latent_system_dim,
          name="ODE",
          **latent_system_kwargs
      )
    elif self.latent_dynamics_type == "Discrete":
      model_constructor = functools.partial(
          dynamics.DiscreteDynamicsNetwork,
          system_dim=self.latent_system_dim,
          name="Discrete",
          **latent_system_kwargs
      )
    else:
      raise NotImplementedError()
    self.dynamics = hk.transform(
        lambda *args, **kwargs_: model_constructor()(*args, **kwargs_))  # pylint: disable=unnecessary-lambda

  def process_inputs_for_encoder(self, x: jnp.ndarray) -> jnp.ndarray:
    return utils.stack_time_into_channels(x, self.data_format)

  def process_latents_for_dynamics(self, z: jnp.ndarray) -> _ArrayOrPhase:
    if self.latent_dynamics_type == "Physics":
      return phase_space.PhaseSpace.from_state(z)
    return z

  def process_latents_for_decoder(self, z: _ArrayOrPhase) -> jnp.ndarray:
    if self.latent_dynamics_type == "Physics":
      return z.q if self.render_from_q_only else z.single_state
    return z

  @property
  def inferred_index(self) -> int:
    if self.latent_training_type == "forward":
      return self.num_inference_steps - 1
    elif self.latent_training_type == "forward_backward":
      assert self.num_inference_steps % 2 == 1
      return self.num_inference_steps // 2
    else:
      raise NotImplementedError()

  @property
  def targets_index_offset(self) -> int:
    if self.training_data_split == "overlap_by_one":
      return -1
    elif self.training_data_split == "no_overlap":
      return 0
    elif self.training_data_split == "include_inference":
      return - self.num_inference_steps
    else:
      raise NotImplementedError()

  @property
  def targets_length(self) -> int:
    if self.training_data_split == "include_inference":
      return self.num_inference_steps + self.num_target_steps
    return self.num_target_steps

  @property
  def train_sequence_length(self) -> int:
    """Computes the total length of a sequence needed for training."""
    if self.training_data_split == "overlap_by_one":
      # Input -     [-------------------------------------------------]
      # Inference - [---------------]
      # Targets -                   [---------------------------------]
      return self.num_inference_steps + self.num_target_steps - 1
    elif self.training_data_split == "no_overlap":
      # Input -     [-------------------------------------------------]
      # Inference - [---------------]
      # Targets -                    [--------------------------------]
      return self.num_inference_steps + self.num_target_steps
    elif self.training_data_split == "include_inference":
      # Input -     [-------------------------------------------------]
      # Inference - [---------------]
      # Targets -   [-------------------------------------------------]
      return self.num_inference_steps + self.num_target_steps
    else:
      raise NotImplementedError()

  def train_data_split(
      self,
      images: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, Any]]:
    images = images[:, :self.train_sequence_length]
    inf_idx = self.num_inference_steps
    t_idx = self.num_inference_steps + self.targets_index_offset
    if self.latent_training_type == "forward":
      inference_data = images[:, :inf_idx]
      target_data = images[:, t_idx:]
      if self.training_data_split == "include_inference":
        num_steps_backward = self.inferred_index
      else:
        num_steps_backward = 0
      num_steps_forward = self.num_target_steps
      if self.training_data_split == "overlap_by_one":
        num_steps_forward -= 1
      unroll_kwargs = dict(
          num_steps_backward=num_steps_backward,
          include_z0=self.training_data_split != "no_overlap",
          num_steps_forward=num_steps_forward,
          dt=self.dt
      )
    elif self.latent_training_type == "forward_backward":
      assert self.training_data_split == "include_inference"
      n_fwd = images.shape[0] // 2
      inference_fwd = images[:n_fwd, :inf_idx]
      targets_fwd = images[:n_fwd, t_idx:]
      inference_bckwd = images[n_fwd:, -inf_idx:]
      targets_bckwd = jnp.flip(images[n_fwd:, :images.shape[1] - t_idx], axis=1)
      inference_data = jnp.concatenate([inference_fwd, inference_bckwd], axis=0)
      target_data = jnp.concatenate([targets_fwd, targets_bckwd], axis=0)
      # This needs to by numpy rather than jax.numpy, because we make some
      # verification checks in `integrators.py:149-161`.
      dt_fwd = np.full([n_fwd], self.dt)
      dt_bckwd = np.full([images.shape[0] - n_fwd], self.dt)
      dt = np.concatenate([dt_fwd, -dt_bckwd], axis=0)
      unroll_kwargs = dict(
          num_steps_backward=self.inferred_index,
          include_z0=True,
          num_steps_forward=self.targets_length - self.inferred_index - 1,
          dt=dt
      )
    else:
      raise NotImplementedError()
    return inference_data, target_data, unroll_kwargs

  def prior(self) -> distrax.Distribution:
    """Given the parameters returns the prior distribution of the model."""
    # Allow to run with both the full parameters and only the priors
    if self.prior_type == "standard_normal":
      # assert self.prior_nets is None and self.gated_made is None
      if self.latent_system_net_type == "mlp":
        event_shape = (self.latent_system_dim,)
      elif self.latent_system_net_type == "conv":
        if self.data_format == "NHWC":
          event_shape = self.latent_spatial_shape + (self.latent_system_dim,)
        else:
          event_shape = (self.latent_system_dim,) + self.latent_spatial_shape
      else:
        raise NotImplementedError()
      return distrax.Normal(jnp.zeros(event_shape), jnp.ones(event_shape))
    else:
      raise ValueError(f"Unrecognized prior_type='{self.prior_type}'.")

  def sample_latent_from_prior(
      self,
      params: utils.Params,
      rng: jnp.ndarray,
      num_samples: int = 1,
      **kwargs: Any) -> jnp.ndarray:
    """Takes sample from the prior (and optionally puts them through the latent transform function."""
    _, sample_key, transf_key = jnr.split(rng, 3)
    prior = self.prior()
    z_raw = prior.sample(seed=sample_key, sample_shape=[num_samples])
    return self.apply_latent_transform(params, transf_key, z_raw, **kwargs)

  def sample_trajectories_from_prior(
      self,
      params: utils.Params,
      num_steps: int,
      rng: jnp.ndarray,
      num_samples: int = 1,
      is_training: bool = False,
      **kwargs
  ) -> distrax.Distribution:
    """Generates samples from the prior (unconditional generation)."""
    sample_key, unroll_key, dec_key = jnr.split(rng, 3)
    z0 = self.sample_latent_from_prior(params, sample_key, num_samples,
                                       is_training=is_training)
    z, _ = self.unroll_latent_dynamics(
        z=self.process_latents_for_dynamics(z0),
        params=params,
        key=unroll_key,
        num_steps_forward=num_steps,
        num_steps_backward=0,
        include_z0=True,
        is_training=is_training,
        **kwargs
    )
    z = self.process_latents_for_decoder(z)
    return self.decode_latents(params, dec_key, z, is_training=is_training)

  def verify_unroll_args(
      self,
      num_steps_forward: int,
      num_steps_backward: int,
      include_z0: bool
  ) -> None:
    if num_steps_forward < 0 or num_steps_backward < 0:
      raise ValueError("num_steps_forward and num_steps_backward can not be "
                       "negative.")
    if num_steps_forward == 0 and num_steps_backward == 0:
      raise ValueError("You need one of num_steps_forward or "
                       "num_of_steps_backward to be positive.")
    if num_steps_forward > 0 and num_steps_backward > 0 and not include_z0:
      raise ValueError("When both num_steps_forward and num_steps_backward are "
                       "positive include_t0 should be True.")
    if num_steps_backward > 0 and not self.can_run_backwards:
      raise ValueError("This model can not be unrolled backward in time.")

  def unroll_latent_dynamics(
      self,
      z: phase_space.PhaseSpace,
      params: hk.Params,
      key: jnp.ndarray,
      num_steps_forward: int,
      num_steps_backward: int,
      include_z0: bool,
      is_training: bool,
      **kwargs: Any
  ) -> Tuple[_ArrayOrPhase, Mapping[str, jnp.ndarray]]:
    self.verify_unroll_args(num_steps_forward, num_steps_backward, include_z0)
    return self.dynamics.apply(
        params,
        key,
        y0=z,
        dt=kwargs.pop("dt", self.dt),
        num_steps_forward=num_steps_forward,
        num_steps_backward=num_steps_backward,
        include_y0=include_z0,
        return_stats=True,
        is_training=is_training
    )

  def _models_core(
      self,
      params: utils.Params,
      keys: jnp.ndarray,
      image_data: jnp.ndarray,
      use_mean: bool,
      is_training: bool,
      **unroll_kwargs: Any
  ) -> Tuple[distrax.Distribution, distrax.Distribution, distrax.Distribution,
             jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray]]:
    enc_key, sample_key, transform_key, unroll_key, dec_key, _ = keys

    # Calculate the approximate posterior q(z|x)
    inference_data = self.process_inputs_for_encoder(image_data)
    q_z: distrax.Distribution = self.encoder.apply(params, enc_key,
                                                   inference_data,
                                                   is_training=is_training)

    # Sample latent variables or take the mean
    z_raw = q_z.mean() if use_mean else q_z.sample(seed=sample_key)

    # Apply latent transformation
    z0 = self.apply_latent_transform(params, transform_key, z_raw,
                                     is_training=is_training)

    # Unroll the latent variable
    z, dyn_stats = self.unroll_latent_dynamics(
        z=self.process_latents_for_dynamics(z0),
        params=params,
        key=unroll_key,
        is_training=is_training,
        **unroll_kwargs
    )
    decoder_z = self.process_latents_for_decoder(z)

    # Compute p(x|z)
    p_x = self.decode_latents(params, dec_key, decoder_z,
                              is_training=is_training)

    z = z.single_state if isinstance(z, phase_space.PhaseSpace) else z
    return p_x, q_z, self.prior(), z0, z, dyn_stats

  def training_objectives(
      self,
      params: utils.Params,
      state: hk.State,
      rng: jnp.ndarray,
      inputs: jnp.ndarray,
      step: jnp.ndarray,
      is_training: bool = True,
      use_mean_for_eval_stats: bool = True
  ) -> Tuple[jnp.ndarray, Sequence[Dict[str, jnp.ndarray]]]:
    # Split all rng keys
    keys = jnr.split(rng, 6)

    # Process training data
    images = utils.extract_image(inputs)
    image_data, target_data, unroll_kwargs = self.train_data_split(images)

    p_x, q_z, prior, _, _, dyn_stats = self._models_core(
        params=params,
        keys=keys,
        image_data=image_data,
        use_mean=False,
        is_training=is_training,
        **unroll_kwargs
    )

    # Note: we reuse the rng key used to sample the latent variable here
    # so that it can be reused to evaluate a (non-analytical) KL at that sample.
    stats = metrics.training_statistics(
        p_x=p_x,
        targets=target_data,
        rescale_by=self.rescale_by,
        rng=keys[1],
        q_z=q_z,
        prior=prior,
        p_x_learned_sigma=self.decoder_kwargs.get("learned_sigma", False)
    )
    stats.update(dyn_stats)

    # Compute other (non-reported statistics)
    z_stats = dict()
    other_stats = dict(x_reconstruct=p_x.mean(), z_stats=z_stats)

    # The loss computation and GECO state update
    new_state = dict()
    if self.objective_type == "GECO":
      geco_stats = metrics.geco_objective(
          l2_loss=stats["l2"],
          kl=stats["kl"],
          alpha=self.geco_alpha,
          kappa=self.geco_kappa,
          constraint_ema=state["GECO"]["geco_constraint_ema"],
          lambda_var=params["GECO"]["geco_lambda_var"],
          is_training=is_training
      )
      new_state["GECO"] = dict(
          geco_constraint_ema=geco_stats["geco_constraint_ema"])
      stats.update(geco_stats)
    elif self.objective_type == "ELBO":
      elbo_stats = metrics.elbo_objective(
          neg_log_p_x=stats["neg_log_p_x"],
          kl=stats["kl"],
          final_beta=self.elbo_beta_final,
          beta_delay=self.elbo_beta_delay,
          step=step
      )
      stats.update(elbo_stats)
    elif self.objective_type == "NON-PROB":
      stats["loss"] = stats["neg_log_p_x"]
    else:
      raise ValueError()

    if not is_training:
      if self.training_data_split == "overlap_by_one":
        reconstruction_skip = self.num_inference_steps - 1
      elif self.training_data_split == "no_overlap":
        reconstruction_skip = self.num_inference_steps
      elif self.training_data_split == "include_inference":
        reconstruction_skip = 0
      else:
        raise NotImplementedError()
      # We intentionally reuse the same rng as the training, in order to be able
      # to run tests and verify that the evaluation and reconstruction work
      # correctly.
      # We need to be able to set `use_mean = False` for some of the tests
      stats.update(metrics.evaluation_only_statistics(
          reconstruct_func=functools.partial(
              self.reconstruct, use_mean=use_mean_for_eval_stats),
          params=params,
          inputs=inputs,
          rng=rng,
          rescale_by=self.rescale_by,
          can_run_backwards=self.can_run_backwards,
          train_sequence_length=self.train_sequence_length,
          reconstruction_skip=reconstruction_skip,
          p_x_learned_sigma=self.decoder_kwargs.get("learned_sigma", False)
      ))

    # Make new state the same type as state
    new_state = utils.convert_to_pytype(new_state, state)
    return stats["loss"], (new_state, stats, other_stats)

  def reconstruct(
      self,
      params: utils.Params,
      inputs: jnp.ndarray,
      rng: Optional[jnp.ndarray],
      forward: bool,
      use_mean: bool = True,
  ) -> distrax.Distribution:
    if not self.can_run_backwards and not forward:
      raise ValueError("This model can not be run backwards.")
    images = utils.extract_image(inputs)
    # This is intentionally matching the split for the training stats
    if forward:
      num_steps_backward = self.inferred_index
      num_steps_forward = images.shape[1] - num_steps_backward - 1
    else:
      num_steps_forward = self.num_inference_steps - self.inferred_index - 1
      num_steps_backward = images.shape[1] - num_steps_forward - 1
    if not self.can_run_backwards:
      num_steps_backward = 0

    if forward:
      image_data = images[:, :self.num_inference_steps]
    else:
      image_data = images[:, -self.num_inference_steps:]

    return self._models_core(
        params=params,
        keys=jnr.split(rng, 6),
        image_data=image_data,
        use_mean=use_mean,
        is_training=False,
        num_steps_forward=num_steps_forward,
        num_steps_backward=num_steps_backward,
        include_z0=True,
    )[0]

  def gt_state_and_latents(
      self,
      params: hk.Params,
      rng: jnp.ndarray,
      inputs: Dict[str, jnp.ndarray],
      seq_length: int,
      is_training: bool = False,
      unroll_direction: str = "forward",
      **kwargs: Dict[str, Any]
  ) -> Tuple[jnp.ndarray, jnp.ndarray,
             Union[distrax.Distribution, jnp.ndarray]]:
    """Computes the ground state and matching latents."""
    assert unroll_direction in ("forward", "backward")
    if unroll_direction == "backward" and not self.can_run_backwards:
      raise ValueError("This model can not be unrolled backwards.")

    images = utils.extract_image(inputs)
    gt_state = utils.extract_gt_state(inputs)

    if unroll_direction == "forward":
      image_data = images[:, :self.num_inference_steps]
      if self.can_run_backwards:
        num_steps_backward = self.inferred_index
        gt_start_idx = 0
      else:
        num_steps_backward = 0
        gt_start_idx = self.inferred_index
      num_steps_forward = seq_length - num_steps_backward - 1
      gt_state = gt_state[:, gt_start_idx: seq_length + gt_start_idx]
    elif unroll_direction == "backward":
      inference_start_idx = seq_length - self.num_inference_steps
      image_data = images[:, inference_start_idx: seq_length]
      num_steps_forward = self.num_inference_steps - self.inferred_index - 1
      num_steps_backward = seq_length - num_steps_forward - 1
      gt_state = gt_state[:, :seq_length]
    else:
      raise NotImplementedError()

    _, q_z, _, z0, z, _ = self._models_core(
        params=params,
        keys=jnr.split(rng, 6),
        image_data=image_data,
        use_mean=True,
        is_training=False,
        num_steps_forward=num_steps_forward,
        num_steps_backward=num_steps_backward,
        include_z0=True,
    )

    if self.has_latent_transform:
      return gt_state, z, z0
    else:
      return gt_state, z, q_z

  def _init_non_model_params_and_state(
      self,
      rng: jnp.ndarray
  ) -> Tuple[utils.Params, utils.Params]:
    if self.objective_type == "GECO":
      # Initialize such that softplus(lambda_var) = 1
      geco_lambda_var = jnp.asarray(jnp.log(jnp.e - 1.0))
      geco_constraint_ema = jnp.asarray(0.0)
      return (dict(GECO=dict(geco_lambda_var=geco_lambda_var)),
              dict(GECO=dict(geco_constraint_ema=geco_constraint_ema)))
    else:
      return dict(), dict()

  def _init_latent_system(
      self,
      rng: jnp.ndarray,
      z: jnp.ndarray,
      **kwargs: Mapping[str, Any]
  ) -> hk.Params:
    """Initializes the parameters of the latent system."""
    return self.dynamics.init(
        rng,
        y0=z,
        dt=self.dt,
        num_steps_forward=1,
        num_steps_backward=0,
        include_y0=True,
        **kwargs
    )
