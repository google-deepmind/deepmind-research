# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Utils."""
import dill
import jax
import jax.numpy as jnp
import tree


def reduce_fn(x, mode):
  """Reduce fn for various losses."""
  if mode == 'none' or mode is None:
    return jnp.asarray(x)
  elif mode == 'sum':
    return jnp.sum(x)
  elif mode == 'mean':
    return jnp.mean(x)
  else:
    raise ValueError('Unsupported reduction option.')


def softmax_cross_entropy(logits, labels, reduction='sum'):
  """Computes softmax cross entropy given logits and one-hot class labels.

  Args:
    logits: Logit output values.
    labels: Ground truth one-hot-encoded labels.
    reduction: Type of reduction to apply to loss.

  Returns:
    Loss value. If `reduction` is `none`, this has the same shape as `labels`;
    otherwise, it is scalar.

  Raises:
    ValueError: If the type of `reduction` is unsupported.
  """
  loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return reduce_fn(loss, reduction)


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


@jax.vmap
def any_in(prediction, target):
  """For each row in a and b, checks if any element of a is in b."""
  return jnp.isin(prediction, target)


def tf1_ema(ema_value, current_value, decay, step):
  """Implements EMA with TF1-style decay warmup."""
  decay = jnp.minimum(decay, (1.0 + step) / (10.0 + step))
  return ema_value * decay + current_value * (1 - decay)


def ema(ema_value, current_value, decay, step):
  """Implements EMA without any warmup."""
  del step
  return ema_value * decay + current_value * (1 - decay)


to_bf16 = lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x
from_bf16 = lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x


def _replicate(x, devices=None):
  """Replicate an object on each device."""
  x = jax.numpy.array(x)
  if devices is None:
    devices = jax.local_devices()
  return jax.device_put_sharded(len(devices) * [x], devices)


def broadcast(obj):
  """Broadcasts an object to all devices."""
  if obj is not None and not isinstance(obj, bool):
    return _replicate(obj)
  else:
    return obj


def split_tree(tuple_tree, base_tree, n):
  """Splits tuple_tree with n-tuple leaves into n trees."""
  return [tree.map_structure_up_to(base_tree, lambda x: x[i], tuple_tree)  # pylint: disable=cell-var-from-loop
          for i in range(n)]


def load_haiku_file(filename):
  """Loads a haiku parameter tree, using dill."""
  with open(filename, 'rb') as in_file:
    output = dill.load(in_file)
  return output


def flatten_haiku_tree(haiku_dict):
  """Flattens a haiku parameter tree into a flat dictionary."""
  out = {}
  for module in haiku_dict.keys():
    out_module = module.replace('/~/', '.').replace('/', '.')
    for key in haiku_dict[module]:
      out_key = f'{out_module}.{key}'
      out[out_key] = haiku_dict[module][key]
  return out
