# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities."""

from typing import Callable, List, Mapping, NamedTuple, Optional, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


Batch = Mapping[str, np.ndarray]
OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
Scalars = Mapping[str, jnp.ndarray]
ParamsOrState = Union[hk.Params, hk.State]


NORM_NAMES = ['layer_norm', 'batchnorm']


# any_in and topk_correct taken from
# https://github.com/deepmind/deepmind-research/blob/master/nfnets/utils.py
@jax.vmap
def any_in(prediction, target):
  """For each row in a and b, checks if any element of a is in b."""
  return jnp.isin(prediction, target)


def topk_correct(logits, labels, mask=None, prefix='', topk=(1, 5)):
  """Calculate top-k error for multiple k values."""
  metrics = {}
  argsorted_logits = jnp.argsort(logits)
  for k in topk:
    pred_labels = argsorted_logits[..., -k:]
    # Get the number of examples where the label is in the top-k predictions
    correct = any_in(pred_labels, labels).any(axis=-1).astype(jnp.float32)
    if mask is not None:
      correct *= mask
    metrics[f'{prefix}top_{k}_acc'] = correct
  return metrics


def softmax_cross_entropy(logits, labels):
  """Computes softmax cross entropy given logits and one-hot class labels.

  Args:
    logits: Logit output values.
    labels: Ground truth one-hot-encoded labels.

  Returns:
    Loss value with the same shape as `labels`;
  """
  return jnp.asarray(optax.softmax_cross_entropy(logits, labels))


def _get_batch_scaled_lr(total_batch_size, lr, scale_by_batch=True):
  # This is the linear scaling rule in Section 5.1 of
  # https://arxiv.org/pdf/1706.02677.pdf.

  if scale_by_batch:
    lr = (lr * total_batch_size) / 256

  return lr


def get_learning_rate_schedule(
    total_batch_size, steps_per_epoch, total_steps, optimizer_config):
  """Build the learning rate schedule function."""
  base_lr = _get_batch_scaled_lr(total_batch_size, optimizer_config.base_lr,
                                 optimizer_config.scale_by_batch)

  schedule_type = optimizer_config.schedule_type
  if schedule_type == 'steps':
    boundaries = optimizer_config.step_decay_kwargs.decay_boundaries
    boundaries.sort()

    decay_rate = optimizer_config.step_decay_kwargs.decay_rate
    boundaries_and_scales = {
        int(boundary * total_steps): decay_rate for boundary in boundaries}
    schedule_fn = optax.piecewise_constant_schedule(
        init_value=base_lr, boundaries_and_scales=boundaries_and_scales)
  elif schedule_type == 'cosine':
    warmup_steps = (optimizer_config.cosine_decay_kwargs.warmup_epochs
                    * steps_per_epoch)
    # Batch scale the other lr values as well:
    init_value = _get_batch_scaled_lr(
        total_batch_size,
        optimizer_config.cosine_decay_kwargs.init_value,
        optimizer_config.scale_by_batch)
    end_value = _get_batch_scaled_lr(
        total_batch_size,
        optimizer_config.cosine_decay_kwargs.end_value,
        optimizer_config.scale_by_batch)

    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=end_value)
  elif schedule_type == 'constant_cosine':
    # Convert end_value to alpha, used by cosine_decay_schedule.
    alpha = optimizer_config.constant_cosine_decay_kwargs.end_value / base_lr

    # Number of steps spent in constant phase.
    constant_steps = int(
        optimizer_config.constant_cosine_decay_kwargs.constant_fraction
        * total_steps)
    decay_steps = total_steps - constant_steps

    constant_phase = optax.constant_schedule(value=base_lr)
    decay_phase = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=decay_steps,
        alpha=alpha)
    schedule_fn = optax.join_schedules(
        schedules=[constant_phase, decay_phase],
        boundaries=[constant_steps])
  else:
    raise ValueError(f'Unknown learning rate schedule: {schedule_type}')

  return schedule_fn


def _weight_decay_exclude(
    exclude_names: Optional[List[str]] = None
) -> Callable[[str, str, jnp.ndarray], bool]:
  """Logic for deciding which parameters to include for weight decay..

  Args:
    exclude_names: an optional list of names to include for weight_decay. ['w']
      by default.

  Returns:
    A predicate that returns True for params that need to be excluded from
    weight_decay.
  """
  # By default weight_decay the weights but not the biases.
  if not exclude_names:
    exclude_names = ['b']

  def exclude(module_name: str, name: str, value: jnp.array):
    del value
    # Do not weight decay the parameters of normalization blocks.
    if any([norm_name in module_name for norm_name in NORM_NAMES]):
      return True
    else:
      return name in exclude_names

  return exclude


class AddWeightDecayState(NamedTuple):
  """Stateless transformation."""


def add_weight_decay(
    weight_decay: float,
    exclude_names: Optional[List[str]] = None) -> optax.GradientTransformation:
  """Add parameter scaled by `weight_decay` to the `updates`.

  Same as optax.add_decayed_weights but can exclude parameters by name.

  Args:
    weight_decay: weight_decay coefficient.
    exclude_names: an optional list of names to exclude for weight_decay. ['b']
      by default.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return AddWeightDecayState()

  def update_fn(updates, state, params):
    exclude = _weight_decay_exclude(exclude_names=exclude_names)

    u_ex, u_in = hk.data_structures.partition(exclude, updates)
    _, p_in = hk.data_structures.partition(exclude, params)
    u_in = jax.tree_multimap(lambda g, p: g + weight_decay * p, u_in, p_in)
    updates = hk.data_structures.merge(u_ex, u_in)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def make_optimizer(optimizer_config, lr_schedule):
  """Construct the optax optimizer with given LR schedule."""
  if (optimizer_config.get('decay_pos_embs') is None or
      optimizer_config.decay_pos_embs):
    # Decay learned position embeddings by default.
    weight_decay_exclude_names = ['b']
  else:
    weight_decay_exclude_names = ['pos_embs', 'b']

  optax_chain = []
  if optimizer_config.max_norm > 0:
    optax_chain.append(
        optax.clip_by_global_norm(optimizer_config.max_norm))

  if optimizer_config.optimizer == 'adam':
    # See: https://arxiv.org/abs/1412.6980
    optax_chain.extend([
        optax.scale_by_adam(**optimizer_config.adam_kwargs),
        add_weight_decay(
            optimizer_config.weight_decay,
            exclude_names=weight_decay_exclude_names)
    ])
  elif optimizer_config.optimizer == 'lamb':
    # See: https://arxiv.org/abs/1904.00962
    optax_chain.extend([
        optax.scale_by_adam(**optimizer_config.lamb_kwargs),
        add_weight_decay(
            optimizer_config.weight_decay,
            exclude_names=weight_decay_exclude_names),
        optax.scale_by_trust_ratio()
    ])
  else:
    raise ValueError(f'Undefined optimizer {optimizer_config.optimizer}')

  # Scale by the (negative) learning rate.
  optax_chain.extend([
      optax.scale_by_schedule(lr_schedule),
      optax.scale(-1),
  ])

  return optax.chain(*optax_chain)
