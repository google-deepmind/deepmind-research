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
"""Module containing the base abstract classes for sequence models."""
import abc
from typing import Any, Dict, Generic, Mapping, Optional, Sequence, Tuple, TypeVar, Union

from absl import logging
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jnr


from physics_inspired_models import utils
from physics_inspired_models.models import networks

T = TypeVar("T")


class SequenceModel(abc.ABC, Generic[T]):
  """An abstract class for sequence models."""

  def __init__(
      self,
      can_run_backwards: bool,
      latent_system_dim: int,
      latent_system_net_type: str,
      latent_system_kwargs: Dict[str, Any],
      encoder_aggregation_type: Optional[str],
      decoder_de_aggregation_type: Optional[str],
      encoder_kwargs: Dict[str, Any],
      decoder_kwargs: Dict[str, Any],
      num_inference_steps: int,
      num_target_steps: int,
      name: str,
      latent_spatial_shape: Optional[Tuple[int, int]] = (4, 4),
      has_latent_transform: bool = False,
      latent_transform_kwargs: Optional[Dict[str, Any]] = None,
      rescale_by: Optional[str] = "pixels_and_time",
      data_format: str = "NHWC",
      **unused_kwargs
  ):
    # Arguments checks
    encoder_kwargs = encoder_kwargs or dict()
    decoder_kwargs = decoder_kwargs or dict()

    # Set the decoder de-aggregation type the "same" type as the encoder if not
    # provided
    if (decoder_de_aggregation_type is None and
        encoder_aggregation_type is not None):
      if encoder_aggregation_type == "linear_projection":
        decoder_de_aggregation_type = "linear_projection"
      elif encoder_aggregation_type in ("mean", "max"):
        decoder_de_aggregation_type = "tile"
      else:
        raise ValueError(f"Unrecognized encoder_aggregation_type="
                         f"{encoder_aggregation_type}")
    if latent_system_net_type == "conv":
      if encoder_aggregation_type is not None:
        raise ValueError("When the latent system is convolutional, the encoder "
                         "aggregation type should be None.")
      if decoder_de_aggregation_type is not None:
        raise ValueError("When the latent system is convolutional, the decoder "
                         "aggregation type should be None.")
    else:
      if encoder_aggregation_type is None:
        raise ValueError("When the latent system is not convolutional, the "
                         "you must provide an encoder aggregation type.")
      if decoder_de_aggregation_type is None:
        raise ValueError("When the latent system is not convolutional, the "
                         "you must provide an decoder aggregation type.")
    if has_latent_transform and latent_transform_kwargs is None:
      raise ValueError("When using latent transformation you have to provide "
                       "the latent_transform_kwargs argument.")
    if unused_kwargs:
      logging.warning("Unused kwargs: %s", str(unused_kwargs))
    super().__init__(**unused_kwargs)
    self.can_run_backwards = can_run_backwards
    self.latent_system_dim = latent_system_dim
    self.latent_system_kwargs = latent_system_kwargs
    self.latent_system_net_type = latent_system_net_type
    self.latent_spatial_shape = latent_spatial_shape
    self.num_inference_steps = num_inference_steps
    self.num_target_steps = num_target_steps
    self.rescale_by = rescale_by
    self.data_format = data_format
    self.name = name

    # Encoder
    self.encoder_kwargs = encoder_kwargs
    self.encoder = hk.transform(
        lambda *args, **kwargs: networks.SpatialConvEncoder(  # pylint: disable=unnecessary-lambda,g-long-lambda
            latent_dim=latent_system_dim,
            aggregation_type=encoder_aggregation_type,
            data_format=data_format,
            name="Encoder",
            **encoder_kwargs
        )(*args, **kwargs))

    # Decoder
    self.decoder_kwargs = decoder_kwargs
    self.decoder = hk.transform(
        lambda *args, **kwargs: networks.SpatialConvDecoder(  # pylint: disable=unnecessary-lambda,g-long-lambda
            initial_spatial_shape=self.latent_spatial_shape,
            de_aggregation_type=decoder_de_aggregation_type,
            data_format=data_format,
            max_de_aggregation_dims=self.latent_system_dim // 2,
            name="Decoder",
            **decoder_kwargs,
        )(*args, **kwargs))

    self.has_latent_transform = has_latent_transform
    if has_latent_transform:
      self.latent_transform = hk.transform(
          lambda *args, **kwargs: networks.make_flexible_net(  # pylint: disable=unnecessary-lambda,g-long-lambda
              net_type=latent_system_net_type,
              output_dims=latent_system_dim,
              name="LatentTransform",
              **latent_transform_kwargs
          )(*args, **kwargs))
    else:
      self.latent_transform = None

    self._jit_init = None

  @property
  @abc.abstractmethod
  def train_sequence_length(self) -> int:
    """Computes the total length of a sequence needed for training or evaluation."""
    pass

  @abc.abstractmethod
  def train_data_split(
      self,
      images: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, Any]]:
    """Extracts from the inputs the data splits for training."""
    pass

  def decode_latents(
      self,
      params: hk.Params,
      rng: jnp.ndarray,
      z: jnp.ndarray,
      **kwargs: Any
  ) -> distrax.Distribution:
    """Decodes the latent variable given the parameters of the model."""
    # Allow to run with both the full parameters and only the decoders
    if self.latent_system_net_type == "mlp":
      fixed_dims = 1
    elif self.latent_system_net_type == "conv":
      fixed_dims = 1 + len(self.latent_spatial_shape)
    else:
      raise NotImplementedError()
    n_shape = z.shape[:-fixed_dims]
    z = z.reshape((-1,) + z.shape[-fixed_dims:])
    x = self.decoder.apply(params, rng, z, **kwargs)
    return jax.tree_map(lambda a: a.reshape(n_shape + a.shape[1:]), x)

  def apply_latent_transform(
      self,
      params: hk.Params,
      key: jnp.ndarray,
      z: jnp.ndarray,
      **kwargs: Any
  ) -> jnp.ndarray:
    if self.latent_transform is not None:
      return self.latent_transform.apply(params, key, z, **kwargs)
    else:
      return z

  @abc.abstractmethod
  def process_inputs_for_encoder(self, x: jnp.ndarray) -> jnp.ndarray:
    pass

  @abc.abstractmethod
  def process_latents_for_dynamics(self, z: jnp.ndarray) -> T:
    pass

  @abc.abstractmethod
  def process_latents_for_decoder(self, z: T) -> jnp.ndarray:
    pass

  @abc.abstractmethod
  def unroll_latent_dynamics(
      self,
      z: T,
      params: utils.Params,
      key: jnp.ndarray,
      num_steps_forward: int,
      num_steps_backward: int,
      include_z0: bool,
      is_training: bool,
      **kwargs: Any
  ) -> Tuple[T, Mapping[str, jnp.ndarray]]:
    """Unrolls the latent dynamics starting from z and pre-processing for the decoder."""
    pass

  @abc.abstractmethod
  def reconstruct(
      self,
      params: utils.Params,
      inputs: jnp.ndarray,
      rng_key: Optional[jnp.ndarray],
      forward: bool,
  ) -> distrax.Distribution:
    """Using the first `num_inference_steps` parts of inputs reconstructs the rest."""
    pass

  @abc.abstractmethod
  def training_objectives(
      self,
      params: utils.Params,
      state: hk.State,
      rng: jnp.ndarray,
      inputs: Union[Dict[str, jnp.ndarray], jnp.ndarray],
      step: jnp.ndarray,
      is_training: bool = True,
      use_mean_for_eval_stats: bool = True
  ) -> Tuple[jnp.ndarray, Sequence[Dict[str, jnp.ndarray]]]:
    """Returns all training objectives statistics and update states."""
    pass

  @property
  @abc.abstractmethod
  def inferred_index(self):
    """Returns the time index in the input sequence, for which the encoder infers.

    If the encoder takes as input the sequence x[0:n-1], where
    `n = self.num_inference_steps`, then this outputs the index `k` relative to
    the begging of the input sequence `x_0`, which the encoder infers.
    """
    pass

  @property
  def inferred_right_offset(self):
    return self.num_inference_steps - 1 - self.inferred_index

  @abc.abstractmethod
  def gt_state_and_latents(
      self,
      params: hk.Params,
      rng: jnp.ndarray,
      inputs: Dict[str, jnp.ndarray],
      seq_len: int,
      is_training: bool = False,
      unroll_direction: str = "forward",
      **kwargs: Dict[str, Any]
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes the ground state and matching latents."""
    pass

  @abc.abstractmethod
  def _init_non_model_params_and_state(
      self,
      rng: jnp.ndarray
  ) -> Tuple[utils.Params, utils.Params]:
    """Initializes any non-model parameters and state."""
    pass

  @abc.abstractmethod
  def _init_latent_system(
      self,
      rng: jnp.ndarray,
      z: jnp.ndarray,
      **kwargs: Any
  ) -> hk.Params:
    """Initializes the parameters of the latent system."""
    pass

  def _init(
      self,
      rng: jnp.ndarray,
      images: jnp.ndarray
  ) -> Tuple[hk.Params, hk.State]:
    """Initializes the whole model parameters and state."""
    inference_data, _, _ = self.train_data_split(images)
    # Initialize parameters and state for the vae training
    rng, key = jnr.split(rng)
    params, state = self._init_non_model_params_and_state(key)

    # Initialize and run encoder
    inference_data = self.process_inputs_for_encoder(inference_data)
    rng, key = jnr.split(rng)
    encoder_params = self.encoder.init(key, inference_data, is_training=True)
    rng, key = jnr.split(rng)
    z_in = self.encoder.apply(encoder_params, key, inference_data,
                              is_training=True)

    # For probabilistic models this will be a distribution
    if isinstance(z_in, distrax.Distribution):
      z_in = z_in.mean()

    # Initialize and run the optional latent transform
    if self.latent_transform is not None:
      rng, key = jnr.split(rng)
      transform_params = self.latent_transform.init(key, z_in, is_training=True)
      rng, key = jnr.split(rng)
      z_in = self.latent_transform.apply(transform_params, key, z_in,
                                         is_training=True)
    else:
      transform_params = dict()

    # Initialize and run the latent system
    z_in = self.process_latents_for_dynamics(z_in)
    rng, key = jnr.split(rng)
    latent_params = self._init_latent_system(key, z_in, is_training=True)
    rng, key = jnr.split(rng)
    z_out, _ = self.unroll_latent_dynamics(
        z=z_in,
        params=latent_params,
        key=key,
        num_steps_forward=1,
        num_steps_backward=0,
        include_z0=False,
        is_training=True
    )
    z_out = self.process_latents_for_decoder(z_out)

    # Initialize and run the decoder
    rng, key = jnr.split(rng)
    decoder_params = self.decoder.init(key, z_out[:, 0], is_training=True)
    _ = self.decoder.apply(decoder_params, rng, z_out[:, 0], is_training=True)

    # Combine all and make immutable
    params = hk.data_structures.merge(params, encoder_params, transform_params,
                                      latent_params, decoder_params)
    params = hk.data_structures.to_immutable_dict(params)
    state = hk.data_structures.to_immutable_dict(state)

    return params, state

  def init(
      self,
      rng: jnp.ndarray,
      inputs_or_shape: Union[jnp.ndarray, Mapping[str, jnp.ndarray],
                             Sequence[int]],
  ) -> Tuple[utils.Params, hk.State]:
    """Initializes the whole model parameters and state."""
    if (isinstance(inputs_or_shape, (tuple, list))
        and isinstance(inputs_or_shape[0], int)):
      images = jnp.zeros(inputs_or_shape)
    else:
      images = utils.extract_image(inputs_or_shape)
    if self._jit_init is None:
      self._jit_init = jax.jit(self._init)
    return self._jit_init(rng, images)
