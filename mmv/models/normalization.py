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

"""Normalize functions constructors."""

from typing import Any, Dict, Optional, Sequence, Union

import haiku as hk
from jax import numpy as jnp

from mmv.models import types


class _BatchNorm(hk.BatchNorm):
  """A `hk.BatchNorm` with adapted default arguments."""

  def __init__(self,
               create_scale: bool = True,
               create_offset: bool = True,
               decay_rate: float = 0.9,
               eps: float = 1e-5,
               test_local_stats: bool = False,
               **kwargs):
    # Check args.
    if kwargs.get('cross_replica_axis', None) is not None:
      raise ValueError(
          'Attempting to use \'batch_norm\' normalizer, but specifying '
          '`cross_replica_axis`. If you want this behavior use '
          '`normalizer=\'cross_replica_batch_norm\'` directly.')

    self._test_local_stats = test_local_stats
    super().__init__(create_scale=create_scale,
                     create_offset=create_offset,
                     decay_rate=decay_rate,
                     eps=eps,
                     **kwargs)

  def __call__(self,
               x: types.TensorLike,
               is_training: bool) -> jnp.ndarray:
    return super().__call__(x, is_training,
                            test_local_stats=self._test_local_stats)


class _CrossReplicaBatchNorm(hk.BatchNorm):
  """A `hk.BatchNorm` with adapted default arguments for cross replica."""

  def __init__(self,
               create_scale: bool = True,
               create_offset: bool = True,
               decay_rate: float = 0.9,
               eps: float = 1e-5,
               test_local_stats: bool = False,
               **kwargs):
    # Check args.
    if 'cross_replica_axis' in kwargs and kwargs['cross_replica_axis'] is None:
      raise ValueError(
          'Attempting to use \'cross_replica_batch_norm\' normalizer, but '
          'specifying `cross_replica_axis` to be None. If you want this '
          'behavior use `normalizer=\'batch_norm\'` directly.')

    self._test_local_stats = test_local_stats
    kwargs['cross_replica_axis'] = kwargs.get('cross_replica_axis', 'i')
    super().__init__(create_scale=create_scale,
                     create_offset=create_offset,
                     decay_rate=decay_rate,
                     eps=eps,
                     **kwargs)

  def __call__(self,
               x: types.TensorLike,
               is_training: bool) -> jnp.ndarray:
    return super().__call__(x, is_training,
                            test_local_stats=self._test_local_stats)


class _LayerNorm(hk.LayerNorm):
  """A `hk.LayerNorm` accepting (and discarding) an `is_training` argument."""

  def __init__(self,
               axis: Union[int, Sequence[int]] = (1, 2),
               create_scale: bool = True,
               create_offset: bool = True,
               **kwargs):
    super().__init__(axis=axis,
                     create_scale=create_scale,
                     create_offset=create_offset,
                     **kwargs)

  def __call__(self,
               x: types.TensorLike,
               is_training: bool) -> jnp.ndarray:
    del is_training  # Unused.
    return super().__call__(x)


_NORMALIZER_NAME_TO_CLASS = {
    'batch_norm': _BatchNorm,
    'cross_replica_batch_norm': _CrossReplicaBatchNorm,
    'layer_norm': _LayerNorm,
}


def get_normalize_fn(
    normalizer_name: str = 'batch_norm',
    normalizer_kwargs: Optional[Dict[str, Any]] = None,
) -> types.NormalizeFn:
  """Handles NormalizeFn creation.

  These functions are expected to be used as part of Haiku model. On each
  application of the returned normalization_fn, a new Haiku layer will be added
  to the model.

  Args:
    normalizer_name: The name of the normalizer to be constructed.
    normalizer_kwargs: The kwargs passed to the normalizer constructor.

  Returns:
    A `types.NormalizeFn` that when applied will create a new layer.

  Raises:
    ValueError: If `normalizer_name` is unknown.
  """
  # Check args.
  if normalizer_name not in _NORMALIZER_NAME_TO_CLASS:
    raise ValueError(f'Unrecognized `normalizer_name` {normalizer_name}.')

  normalizer_class = _NORMALIZER_NAME_TO_CLASS[normalizer_name]
  normalizer_kwargs = normalizer_kwargs or dict()

  return lambda *a, **k: normalizer_class(**normalizer_kwargs)(*a, **k)  # pylint: disable=unnecessary-lambda
