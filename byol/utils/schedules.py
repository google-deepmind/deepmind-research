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

"""Learning rate schedules."""
import jax.numpy as jnp


def target_ema(global_step: jnp.ndarray,
               base_ema: float,
               max_steps: int) -> jnp.ndarray:
  decay = _cosine_decay(global_step, max_steps, 1.)
  return 1. - (1. - base_ema) * decay


def learning_schedule(global_step: jnp.ndarray,
                      batch_size: int,
                      base_learning_rate: float,
                      total_steps: int,
                      warmup_steps: int) -> float:
  """Cosine learning rate scheduler."""
  # Compute LR & Scaled LR
  scaled_lr = base_learning_rate * batch_size / 256.
  learning_rate = (
      global_step.astype(jnp.float32) / int(warmup_steps) *
      scaled_lr if warmup_steps > 0 else scaled_lr)

  # Cosine schedule after warmup.
  return jnp.where(
      global_step < warmup_steps, learning_rate,
      _cosine_decay(global_step - warmup_steps, total_steps - warmup_steps,
                    scaled_lr))


def _cosine_decay(global_step: jnp.ndarray,
                  max_steps: int,
                  initial_value: float) -> jnp.ndarray:
  """Simple implementation of cosine decay from TF1."""
  global_step = jnp.minimum(global_step, max_steps)
  cosine_decay_value = 0.5 * (1 + jnp.cos(jnp.pi * global_step / max_steps))
  decayed_learning_rate = initial_value * cosine_decay_value
  return decayed_learning_rate
