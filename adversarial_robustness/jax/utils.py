# Copyright 2021 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions."""

import re
from typing import Optional, Sequence, Tuple

import chex
import einops
import haiku as hk
import jax
import jax.numpy as jnp
import optax


def get_cosine_schedule(
    max_learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0) -> optax.Schedule:
  """Builds a cosine decay schedule with initial warm-up."""
  if total_steps < warmup_steps:
    return optax.linear_schedule(init_value=0., end_value=max_learning_rate,
                                 transition_steps=warmup_steps)
  return optax.join_schedules([
      optax.linear_schedule(init_value=0., end_value=max_learning_rate,
                            transition_steps=warmup_steps),
      optax.cosine_decay_schedule(init_value=max_learning_rate,
                                  decay_steps=total_steps - warmup_steps),
  ], [warmup_steps])


def get_step_schedule(
    max_learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0) -> optax.Schedule:
  """Builds a step schedule with initial warm-up."""
  if total_steps < warmup_steps:
    return optax.linear_schedule(init_value=0., end_value=max_learning_rate,
                                 transition_steps=warmup_steps)
  return optax.join_schedules([
      optax.linear_schedule(init_value=0., end_value=max_learning_rate,
                            transition_steps=warmup_steps),
      optax.piecewise_constant_schedule(
          init_value=max_learning_rate,
          boundaries_and_scales={total_steps * 2 // 3: .1}),
  ], [warmup_steps])


def sgd_momentum(learning_rate_fn: optax.Schedule,
                 momentum: float = 0.,
                 nesterov: bool = False) -> optax.GradientTransformation:
  return optax.chain(
      optax.trace(decay=momentum, nesterov=nesterov),
      optax.scale_by_schedule(learning_rate_fn),
      optax.scale(-1.))


def cross_entropy(logits: chex.Array, labels: chex.Array) -> chex.Array:
  return -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)


def kl_divergence(q_logits: chex.Array,
                  p_logits: chex.Array) -> chex.Array:
  """Compute the KL divergence."""
  p_probs = jax.nn.softmax(p_logits)
  return cross_entropy(q_logits, p_probs) - cross_entropy(p_logits, p_probs)


def accuracy(logits: chex.Array, labels: chex.Array) -> chex.Array:
  predicted_label = jnp.argmax(logits, axis=-1)
  correct = jnp.equal(predicted_label, labels).astype(jnp.float32)
  return jnp.sum(correct, axis=0) / logits.shape[0]


def weight_decay(params: hk.Params,
                 regex_match: Optional[Sequence[str]] = None,
                 regex_ignore: Optional[Sequence[str]] = None) -> chex.Array:
  """Computes the L2 regularization loss."""
  if regex_match is None:
    regex_match = ('.*w$', '.*b$')
  if regex_ignore is None:
    regex_ignore = ('.*batchnorm.*',)
  l2_norm = 0.
  for mod_name, mod_params in params.items():
    for param_name, param in mod_params.items():
      name = '/'.join([mod_name, param_name])
      if (regex_match and
          all(not re.match(regex, name) for regex in regex_match)):
        continue
      if (regex_ignore and
          any(re.match(regex, name) for regex in regex_ignore)):
        continue
      l2_norm += jnp.sum(jnp.square(param))
  return .5 * l2_norm


def ema_update(step: chex.Array,
               avg_params: chex.ArrayTree,
               new_params: chex.ArrayTree,
               decay_rate: float = 0.99,
               warmup_steps: int = 0,
               dynamic_decay: bool = True) -> chex.ArrayTree:
  """Applies an exponential moving average."""
  factor = (step >= warmup_steps).astype(jnp.float32)
  if dynamic_decay:
    # Uses TF-style EMA.
    delta = step - warmup_steps
    decay = jnp.minimum(decay_rate, (1. + delta) / (10. + delta))
  else:
    decay = decay_rate
  decay *= factor
  def _weighted_average(p1, p2):
    d = decay.astype(p1.dtype)
    return (1 - d) * p1 + d * p2
  return jax.tree_multimap(_weighted_average, new_params, avg_params)


def cutmix(rng: chex.PRNGKey,
           images: chex.Array,
           labels: chex.Array,
           alpha: float = 1.,
           beta: float = 1.,
           split: int = 1) -> Tuple[chex.Array, chex.Array]:
  """Composing two images by inserting a patch into another image."""
  batch_size, height, width, _ = images.shape
  split_batch_size = batch_size // split if split > 1 else batch_size

  # Masking bounding box.
  box_rng, lam_rng, rng = jax.random.split(rng, num=3)
  lam = jax.random.beta(lam_rng, a=alpha, b=beta, shape=())
  cut_rat = jnp.sqrt(1. - lam)
  cut_w = jnp.array(width * cut_rat, dtype=jnp.int32)
  cut_h = jnp.array(height * cut_rat, dtype=jnp.int32)
  box_coords = _random_box(box_rng, height, width, cut_h, cut_w)
  # Adjust lambda.
  lam = 1. - (box_coords[2] * box_coords[3] / (height * width))
  idx = jax.random.permutation(rng, split_batch_size)
  def _cutmix(x, y):
    images_a = x
    images_b = x[idx, :, :, :]
    y = lam * y + (1. - lam) * y[idx, :]
    x = _compose_two_images(images_a, images_b, box_coords)
    return x, y

  if split <= 1:
    return _cutmix(images, labels)

  # Apply CutMix separately on each sub-batch. This reverses the effect of
  # `repeat` in datasets.
  images = einops.rearrange(images, '(b1 b2) ... -> b1 b2 ...', b2=split)
  labels = einops.rearrange(labels, '(b1 b2) ... -> b1 b2 ...', b2=split)
  images, labels = jax.vmap(_cutmix, in_axes=1, out_axes=1)(images, labels)
  images = einops.rearrange(images, 'b1 b2 ... -> (b1 b2) ...', b2=split)
  labels = einops.rearrange(labels, 'b1 b2 ... -> (b1 b2) ...', b2=split)
  return images, labels


def _random_box(rng: chex.PRNGKey,
                height: chex.Numeric,
                width: chex.Numeric,
                cut_h: chex.Array,
                cut_w: chex.Array) -> chex.Array:
  """Sample a random box of shape [cut_h, cut_w]."""
  height_rng, width_rng = jax.random.split(rng)
  i = jax.random.randint(
      height_rng, shape=(), minval=0, maxval=height, dtype=jnp.int32)
  j = jax.random.randint(
      width_rng, shape=(), minval=0, maxval=width, dtype=jnp.int32)
  bby1 = jnp.clip(i - cut_h // 2, 0, height)
  bbx1 = jnp.clip(j - cut_w // 2, 0, width)
  h = jnp.clip(i + cut_h // 2, 0, height) - bby1
  w = jnp.clip(j + cut_w // 2, 0, width) - bbx1
  return jnp.array([bby1, bbx1, h, w])


def _compose_two_images(images: chex.Array,
                        image_permutation: chex.Array,
                        bbox: chex.Array) -> chex.Array:
  """Inserting the second minibatch into the first at the target locations."""
  def _single_compose_two_images(image1, image2):
    height, width, _ = image1.shape
    mask = _window_mask(bbox, (height, width))
    return image1 * (1. - mask) + image2 * mask
  return jax.vmap(_single_compose_two_images)(images, image_permutation)


def _window_mask(destination_box: chex.Array,
                 size: Tuple[int, int]) -> jnp.ndarray:
  """Mask a part of the image."""
  height_offset, width_offset, h, w = destination_box
  h_range = jnp.reshape(jnp.arange(size[0]), [size[0], 1, 1])
  w_range = jnp.reshape(jnp.arange(size[1]), [1, size[1], 1])
  return jnp.logical_and(
      jnp.logical_and(height_offset <= h_range,
                      h_range < height_offset + h),
      jnp.logical_and(width_offset <= w_range,
                      w_range < width_offset + w)).astype(jnp.float32)
