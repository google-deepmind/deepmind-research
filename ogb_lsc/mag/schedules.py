# Copyright 2021 DeepMind Technologies Limited.
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

"""Scheduling utilities."""

import jax.numpy as jnp


def apply_ema_decay(
    ema_value: jnp.ndarray,
    current_value: jnp.ndarray,
    decay: jnp.ndarray,
) -> jnp.ndarray:
  """Implements EMA."""
  return ema_value * decay + current_value * (1 - decay)


def ema_decay_schedule(
    base_rate: jnp.ndarray,
    step: jnp.ndarray,
    total_steps: jnp.ndarray,
    use_schedule: bool,
) -> jnp.ndarray:
  """Anneals decay rate to 1 with cosine schedule."""
  if not use_schedule:
    return base_rate
  multiplier = _cosine_decay(step, total_steps, 1.)
  return 1. - (1. - base_rate) * multiplier


def _cosine_decay(
    global_step: jnp.ndarray,
    max_steps: int,
    initial_value: float,
) -> jnp.ndarray:
  """Simple implementation of cosine decay from TF1."""
  global_step = jnp.minimum(global_step, max_steps).astype(jnp.float32)
  cosine_decay_value = 0.5 * (1 + jnp.cos(jnp.pi * global_step / max_steps))
  decayed_learning_rate = initial_value * cosine_decay_value
  return decayed_learning_rate


def learning_schedule(
    global_step: jnp.ndarray,
    base_learning_rate: float,
    total_steps: int,
    warmup_steps: int,
    use_schedule: bool,
) -> float:
  """Cosine learning rate scheduler."""
  # Compute LR & Scaled LR
  if not use_schedule:
    return base_learning_rate
  warmup_learning_rate = (
      global_step.astype(jnp.float32) / int(warmup_steps) *
      base_learning_rate if warmup_steps > 0 else base_learning_rate)

  # Cosine schedule after warmup.
  decay_learning_rate = _cosine_decay(global_step - warmup_steps,
                                      total_steps - warmup_steps,
                                      base_learning_rate)
  return jnp.where(global_step < warmup_steps, warmup_learning_rate,
                   decay_learning_rate)
