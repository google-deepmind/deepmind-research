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
"""Position encodings and utilities."""

import abc
import functools

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def generate_fourier_features(
    pos, num_bands, max_resolution=(224, 224),
    concat_pos=True, sine_only=False):
  """Generate a Fourier frequency position encoding with linear spacing.

  Args:
    pos: The position of n points in d dimensional space.
      A jnp array of shape [n, d].
    num_bands: The number of bands (K) to use.
    max_resolution: The maximum resolution (i.e. the number of pixels per dim).
      A tuple representing resolution for each dimension
    concat_pos: Concatenate the input position encoding to the Fourier features?
    sine_only: Whether to use a single phase (sin) or two (sin/cos) for each
      frequency band.
  Returns:
    embedding: A 1D jnp array of shape [n, n_channels]. If concat_pos is True
      and sine_only is False, output dimensions are ordered as:
        [dim_1, dim_2, ..., dim_d,
         sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ...,
         sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d),
         cos(pi*f_1*dim_1), ..., cos(pi*f_K*dim_1), ...,
         cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)],
       where dim_i is pos[:, i] and f_k is the kth frequency band.
  """
  min_freq = 1.0
  # Nyquist frequency at the target resolution:

  freq_bands = jnp.stack([
      jnp.linspace(min_freq, res / 2, num=num_bands, endpoint=True)
      for res in max_resolution], axis=0)

  # Get frequency bands for each spatial dimension.
  # Output is size [n, d * num_bands]
  per_pos_features = pos[:, :, None] * freq_bands[None, :, :]
  per_pos_features = jnp.reshape(per_pos_features,
                                 [-1, np.prod(per_pos_features.shape[1:])])

  if sine_only:
    # Output is size [n, d * num_bands]
    per_pos_features = jnp.sin(jnp.pi * (per_pos_features))
  else:
    # Output is size [n, 2 * d * num_bands]
    per_pos_features = jnp.concatenate(
        [jnp.sin(jnp.pi * per_pos_features),
         jnp.cos(jnp.pi * per_pos_features)], axis=-1)
  # Concatenate the raw input positions.
  if concat_pos:
    # Adds d bands to the encoding.
    per_pos_features = jnp.concatenate([pos, per_pos_features], axis=-1)
  return per_pos_features


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
  """Generate an array of position indices for an N-D input array.

  Args:
    index_dims: The shape of the index dimensions of the input array.
    output_range: The min and max values taken by each input index dimension.
  Returns:
    A jnp array of shape [index_dims[0], index_dims[1], .., index_dims[-1], N].
  """
  def _linspace(n_xels_per_dim):
    return jnp.linspace(
        output_range[0], output_range[1],
        num=n_xels_per_dim,
        endpoint=True, dtype=jnp.float32)

  dim_ranges = [
      _linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
  array_index_grid = jnp.meshgrid(*dim_ranges, indexing='ij')

  return jnp.stack(array_index_grid, axis=-1)


class AbstractPositionEncoding(hk.Module, metaclass=abc.ABCMeta):
  """Abstract Perceiver decoder."""

  @abc.abstractmethod
  def __call__(self, batch_size, pos):
    raise NotImplementedError


class TrainablePositionEncoding(AbstractPositionEncoding):
  """Trainable position encoding."""

  def __init__(self, index_dim, num_channels=128, init_scale=0.02, name=None):
    super(TrainablePositionEncoding, self).__init__(name=name)
    self._index_dim = index_dim
    self._num_channels = num_channels
    self._init_scale = init_scale

  def __call__(self, batch_size, pos=None):
    del pos  # Unused.
    pos_embs = hk.get_parameter(
        'pos_embs', [self._index_dim, self._num_channels],
        init=hk.initializers.TruncatedNormal(stddev=self._init_scale))

    if batch_size is not None:
      pos_embs = jnp.broadcast_to(
          pos_embs[None, :, :], (batch_size,) + pos_embs.shape)
    return pos_embs


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
  """Checks or builds spatial position features (x, y, ...).

  Args:
    pos: None, or an array of position features. If None, position features
      are built. Otherwise, their size is checked.
    index_dims: An iterable giving the spatial/index size of the data to be
      featurized.
    batch_size: The batch size of the data to be featurized.
  Returns:
    An array of position features, of shape [batch_size, prod(index_dims)].
  """
  if pos is None:
    pos = build_linear_positions(index_dims)
    pos = jnp.broadcast_to(pos[None], (batch_size,) + pos.shape)
    pos = jnp.reshape(pos, [batch_size, np.prod(index_dims), -1])
  else:
    # Just a warning label: you probably don't want your spatial features to
    # have a different spatial layout than your pos coordinate system.
    # But feel free to override if you think it'll work!
    assert pos.shape[-1] == len(index_dims)

  return pos


class FourierPositionEncoding(AbstractPositionEncoding):
  """Fourier (Sinusoidal) position encoding."""

  def __init__(self, index_dims, num_bands, concat_pos=True,
               max_resolution=None, sine_only=False, name=None):
    super(FourierPositionEncoding, self).__init__(name=name)
    self._num_bands = num_bands
    self._concat_pos = concat_pos
    self._sine_only = sine_only
    self._index_dims = index_dims
    # Use the index dims as the maximum resolution if it's not provided.
    self._max_resolution = max_resolution or index_dims

  def __call__(self, batch_size, pos=None):
    pos = _check_or_build_spatial_positions(pos, self._index_dims, batch_size)
    build_ff_fn = functools.partial(
        generate_fourier_features,
        num_bands=self._num_bands,
        max_resolution=self._max_resolution,
        concat_pos=self._concat_pos,
        sine_only=self._sine_only)
    return jax.vmap(build_ff_fn, 0, 0)(pos)


class PositionEncodingProjector(AbstractPositionEncoding):
  """Projects a position encoding to a target size."""

  def __init__(self, output_size, base_position_encoding, name=None):
    super(PositionEncodingProjector, self).__init__(name=name)
    self._output_size = output_size
    self._base_position_encoding = base_position_encoding

  def __call__(self, batch_size, pos=None):
    base_pos = self._base_position_encoding(batch_size, pos)
    projected_pos = hk.Linear(output_size=self._output_size)(base_pos)
    return projected_pos


def build_position_encoding(
    position_encoding_type,
    index_dims,
    project_pos_dim=-1,
    trainable_position_encoding_kwargs=None,
    fourier_position_encoding_kwargs=None,
    name=None):
  """Builds the position encoding."""

  if position_encoding_type == 'trainable':
    assert trainable_position_encoding_kwargs is not None
    output_pos_enc = TrainablePositionEncoding(
        # Construct 1D features:
        index_dim=np.prod(index_dims),
        name=name,
        **trainable_position_encoding_kwargs)
  elif position_encoding_type == 'fourier':
    assert fourier_position_encoding_kwargs is not None
    output_pos_enc = FourierPositionEncoding(
        index_dims=index_dims,
        name=name,
        **fourier_position_encoding_kwargs)
  else:
    raise ValueError(f'Unknown position encoding: {position_encoding_type}.')

  if project_pos_dim > 0:
    # Project the position encoding to a target dimension:
    output_pos_enc = PositionEncodingProjector(
        output_size=project_pos_dim,
        base_position_encoding=output_pos_enc)

  return output_pos_enc
