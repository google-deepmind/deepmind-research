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
"""Module containing code for computing various metrics for training and evaluation."""
from typing import Callable, Dict, Optional

import distrax
import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np

import physics_inspired_models.utils as utils


_ReconstructFunc = Callable[[utils.Params, jnp.ndarray, jnp.ndarray, bool],
                            distrax.Distribution]


def calculate_small_latents(dist, threshold=0.5):
  """Calculates the number of active latents by thresholding the variance of their distribution."""
  if not isinstance(dist, distrax.Normal):
    raise NotImplementedError()
  latent_means = dist.mean()
  latent_stddevs = dist.variance()
  small_latents = jnp.sum(
      (latent_stddevs < threshold) & (jnp.abs(latent_means) > 0.1), axis=1)
  return jnp.mean(small_latents)


def compute_scale(
    targets: jnp.ndarray,
    rescale_by: str
) -> jnp.ndarray:
  """Compute a scaling factor based on targets shape and the rescale_by argument."""
  if rescale_by == "pixels_and_time":
    return jnp.asarray(np.prod(targets.shape[-4:]))
  elif rescale_by is not None:
    raise ValueError(f"Unrecognized rescale_by={rescale_by}.")
  else:
    return jnp.ones([])


def compute_data_domain_stats(
    p_x: distrax.Distribution,
    targets: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
  """Compute several statistics in the data domain, such as L2 and negative log likelihood."""
  axis = tuple(range(2, targets.ndim))
  l2_over_time = jnp.sum((p_x.mean() - targets) ** 2, axis=axis)
  l2 = jnp.sum(l2_over_time, axis=1)

  # Calculate relative L2 normalised by image "length"
  norm_factor = jnp.sum(targets**2, axis=(2, 3, 4))
  l2_over_time_norm = l2_over_time / norm_factor
  l2_norm = jnp.sum(l2_over_time_norm, axis=1)

  # Compute negative log-likelihood under p(x)
  neg_log_p_x_over_time = - np.sum(p_x.log_prob(targets), axis=axis)
  neg_log_p_x = jnp.sum(neg_log_p_x_over_time, axis=1)

  return dict(
      neg_log_p_x_over_time=neg_log_p_x_over_time,
      neg_log_p_x=neg_log_p_x,
      l2_over_time=l2_over_time,
      l2=l2,
      l2_over_time_norm=l2_over_time_norm,
      l2_norm=l2_norm,
  )


def compute_vae_stats(
    neg_log_p_x: jnp.ndarray,
    rng: jnp.ndarray,
    q_z: distrax.Distribution,
    prior: distrax.Distribution
) -> Dict[str, jnp.ndarray]:
  """Compute the KL(q(z|x)||p(z)) and the negative ELBO, which are used for VAE models."""
  # Compute the KL
  kl = distrax.estimate_kl_best_effort(q_z, prior, rng_key=rng, num_samples=1)
  kl = np.sum(kl, axis=list(range(1, kl.ndim)))
  # Sanity check
  assert kl.shape == neg_log_p_x.shape
  return dict(
      kl=kl,
      neg_elbo=neg_log_p_x + kl,
  )


def training_statistics(
    p_x: distrax.Distribution,
    targets: jnp.ndarray,
    rescale_by: Optional[str],
    rng: Optional[jnp.ndarray] = None,
    q_z: Optional[distrax.Distribution] = None,
    prior: Optional[distrax.Distribution] = None,
    p_x_learned_sigma: bool = False
) -> Dict[str, jnp.ndarray]:
  """Computes various statistics we track during training."""
  stats = compute_data_domain_stats(p_x, targets)

  if rng is not None and q_z is not None and prior is not None:
    stats.update(compute_vae_stats(stats["neg_log_p_x"], rng, q_z, prior))
  else:
    assert rng is None and q_z is None and prior is None

  # Rescale these stats accordingly
  scale = compute_scale(targets, rescale_by)
  # Note that "_over_time" stats are getting normalised by time here
  stats = jax.tree_map(lambda x: x / scale, stats)
  if p_x_learned_sigma:
    stats["p_x_sigma"] = p_x.variance().reshape([-1])[0]
  if q_z is not None:
    stats["small_latents"] = calculate_small_latents(q_z)
  return stats


def evaluation_only_statistics(
    reconstruct_func: _ReconstructFunc,
    params: hk.Params,
    inputs: jnp.ndarray,
    rng: jnp.ndarray,
    rescale_by: str,
    can_run_backwards: bool,
    train_sequence_length: int,
    reconstruction_skip: int,
    p_x_learned_sigma: bool = False,
) -> Dict[str, jnp.ndarray]:
  """Computes various statistics we track only during evaluation."""
  full_trajectory = utils.extract_image(inputs)
  prefixes = ("forward", "backward") if can_run_backwards else ("forward",)

  full_forward_targets = jax.tree_map(
      lambda x: x[:, reconstruction_skip:], full_trajectory)
  full_backward_targets = jax.tree_map(
      lambda x: x[:, :x.shape[1]-reconstruction_skip], full_trajectory)
  train_targets_length = train_sequence_length - reconstruction_skip
  full_targets_length = full_forward_targets.shape[1]

  stats = dict()
  keys = ()

  for prefix in prefixes:
    # Fully unroll the model and reconstruct the whole sequence
    full_prediction = reconstruct_func(params, full_trajectory, rng,
                                       prefix == "forward")
    assert isinstance(full_prediction, distrax.Normal)
    full_targets = (full_forward_targets if prefix == "forward" else
                    full_backward_targets)
    # In cases where the model can run backwards it is possible to reconstruct
    # parts which were indented to be skipped, so here we take care of that.
    if full_prediction.mean().shape[1] > full_targets_length:
      if prefix == "forward":
        full_prediction = jax.tree_map(lambda x: x[:, -full_targets_length:],
                                       full_prediction)
      else:
        full_prediction = jax.tree_map(lambda x: x[:, :full_targets_length],
                                       full_prediction)

    # Based on the prefix and suffix fetch correct predictions and targets
    for suffix in ("train", "extrapolation", "full"):
      if prefix == "forward" and suffix == "train":
        predict, targets = jax.tree_map(lambda x: x[:, :train_targets_length],
                                        (full_prediction, full_targets))
      elif prefix == "forward" and suffix == "extrapolation":
        predict, targets = jax.tree_map(lambda x: x[:, train_targets_length:],
                                        (full_prediction, full_targets))
      elif prefix == "backward" and suffix == "train":
        predict, targets = jax.tree_map(lambda x: x[:, -train_targets_length:],
                                        (full_prediction, full_targets))
      elif prefix == "backward" and suffix == "extrapolation":
        predict, targets = jax.tree_map(lambda x: x[:, :-train_targets_length],
                                        (full_prediction, full_targets))
      else:
        predict, targets = full_prediction, full_targets

      # Compute train statistics
      train_stats = training_statistics(predict, targets, rescale_by,
                                        p_x_learned_sigma=p_x_learned_sigma)
      for key, value in train_stats.items():
        stats[prefix + "_" + suffix + "_" + key] = value
      # Copy all stats keys
      keys = tuple(train_stats.keys())

  # Make a combined metric summing forward and backward
  if can_run_backwards:
    # Also compute
    for suffix in ("train", "extrapolation", "full"):
      for key in keys:
        forward = stats["forward_" + suffix + "_" + key]
        backward = stats["backward_" + suffix + "_" + key]
        combined = (forward + backward) / 2
        stats["combined_" + suffix + "_" + key] = combined

  return stats


def geco_objective(
    l2_loss,
    kl,
    alpha,
    kappa,
    constraint_ema,
    lambda_var,
    is_training
) -> Dict[str, jnp.ndarray]:
  """Computes the objective for GECO and some of it statistics used ofr updates."""
  # C_t
  constraint_t = l2_loss - kappa
  if is_training:
    # We update C_ma only during training
    constraint_ema = alpha * constraint_ema + (1 - alpha) * constraint_t
  lagrange = nn.softplus(lambda_var)
  lagrange = jnp.broadcast_to(lagrange, constraint_ema.shape)
  # Add this special op for getting all gradients correct
  loss = utils.geco_lagrange_product(lagrange, constraint_ema, constraint_t)
  return dict(
      loss=loss + kl,
      geco_multiplier=lagrange,
      geco_constraint=constraint_t,
      geco_constraint_ema=constraint_ema
  )


def elbo_objective(neg_log_p_x, kl, final_beta, beta_delay, step):
  """Computes objective for optimizing the Evidence Lower Bound (ELBO)."""
  if beta_delay == 0:
    beta = final_beta
  else:
    delayed_beta = jnp.minimum(float(step) / float(beta_delay), 1.0)
    beta = delayed_beta * final_beta
  return dict(
      loss=neg_log_p_x + beta * kl,
      elbo_beta=beta
  )
