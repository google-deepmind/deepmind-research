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
"""Module containing all of the networks as Haiku modules."""
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from absl import logging
import distrax
import haiku as hk
import jax.numpy as jnp

from physics_inspired_models import utils

Activation = Union[str, Callable[[jnp.ndarray], jnp.ndarray]]


class DenseNet(hk.Module):
  """A feed forward network (MLP)."""

  def __init__(
      self,
      num_units: Sequence[int],
      activate_final: bool = False,
      activation: Activation = "leaky_relu",
      name: Optional[str] = None):
    super().__init__(name=name)
    self.num_units = num_units
    self.num_layers = len(self.num_units)
    self.activate_final = activate_final
    self.activation = utils.get_activation(activation)

    self.linear_modules = []
    for i in range(self.num_layers):
      self.linear_modules.append(
          hk.Linear(
              output_size=self.num_units[i],
              name=f"ff_{i}"
          )
      )

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    net = inputs
    for i, linear in enumerate(self.linear_modules):
      net = linear(net)
      if i < self.num_layers - 1 or self.activate_final:
        net = self.activation(net)
    return net


class Conv2DNet(hk.Module):
  """Convolutional Network."""

  def __init__(
      self,
      output_channels: Sequence[int],
      kernel_shapes: Union[int, Sequence[int]] = 3,
      strides: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[str]] = "SAME",
      data_format: str = "NHWC",
      with_batch_norm: bool = False,
      activate_final: bool = False,
      activation: Activation = "leaky_relu",
      name: Optional[str] = None):
    super().__init__(name=name)
    self.output_channels = tuple(output_channels)
    self.num_layers = len(self.output_channels)
    self.kernel_shapes = utils.bcast_if(kernel_shapes, int, self.num_layers)
    self.strides = utils.bcast_if(strides, int, self.num_layers)
    self.padding = utils.bcast_if(padding, str, self.num_layers)
    self.data_format = data_format
    self.with_batch_norm = with_batch_norm
    self.activate_final = activate_final
    self.activation = utils.get_activation(activation)

    if len(self.kernel_shapes) != self.num_layers:
      raise ValueError(f"Kernel shapes is of size {len(self.kernel_shapes)}, "
                       f"while output_channels is of size{self.num_layers}.")
    if len(self.strides) != self.num_layers:
      raise ValueError(f"Strides is of size {len(self.kernel_shapes)}, while "
                       f"output_channels is of size{self.num_layers}.")
    if len(self.padding) != self.num_layers:
      raise ValueError(f"Padding is of size {len(self.padding)}, while "
                       f"output_channels is of size{self.num_layers}.")

    self.conv_modules = []
    self.bn_modules = []
    for i in range(self.num_layers):
      self.conv_modules.append(
          hk.Conv2D(
              output_channels=self.output_channels[i],
              kernel_shape=self.kernel_shapes[i],
              stride=self.strides[i],
              padding=self.padding[i],
              data_format=data_format,
              name=f"conv_2d_{i}")
      )
      if with_batch_norm:
        self.bn_modules.append(
            hk.BatchNorm(
                create_offset=True,
                create_scale=False,
                decay_rate=0.999,
                name=f"batch_norm_{i}")
        )
      else:
        self.bn_modules.append(None)

  def __call__(self, inputs: jnp.ndarray, is_training: bool):
    assert inputs.ndim == 4
    net = inputs
    for i, (conv, bn) in enumerate(zip(self.conv_modules, self.bn_modules)):
      net = conv(net)
      # Batch norm
      if bn is not None:
        net = bn(net, is_training=is_training)
      if i < self.num_layers - 1 or self.activate_final:
        net = self.activation(net)
    return net


class SpatialConvEncoder(hk.Module):
  """Spatial Convolutional Encoder for learning the Hamiltonian."""

  def __init__(
      self,
      latent_dim: int,
      conv_channels: Union[Sequence[int], int],
      num_blocks: int,
      blocks_depth: int = 2,
      distribution_name: str = "diagonal_normal",
      aggregation_type: Optional[str] = None,
      data_format: str = "NHWC",
      activation: Activation = "leaky_relu",
      scale_factor: int = 2,
      kernel_shapes: Union[Sequence[int], int] = 3,
      padding: Union[Sequence[str], str] = "SAME",
      name: Optional[str] = None):
    super().__init__(name=name)
    if aggregation_type not in (None, "max", "mean", "linear_projection"):
      raise ValueError(f"Unrecognized aggregation_type={aggregation_type}.")
    self.latent_dim = latent_dim
    self.conv_channels = conv_channels
    self.num_blocks = num_blocks
    self.scale_factor = scale_factor
    self.data_format = data_format
    self.distribution_name = distribution_name
    self.aggregation_type = aggregation_type

    # Compute the required size of the output
    if distribution_name is None:
      self.output_dim = latent_dim
    elif distribution_name == "diagonal_normal":
      self.output_dim = 2 * latent_dim
    else:
      raise ValueError(f"Unrecognized distribution_name={distribution_name}.")

    if isinstance(conv_channels, int):
      conv_channels = [[conv_channels] * blocks_depth
                       for _ in range(num_blocks)]
      conv_channels[-1] += [self.output_dim]
    else:
      assert isinstance(conv_channels, (list, tuple))
      assert len(conv_channels) == num_blocks
      conv_channels = list(list(c) for c in conv_channels)
      conv_channels[-1].append(self.output_dim)

    if isinstance(kernel_shapes, tuple):
      kernel_shapes = list(kernel_shapes)

    # Convolutional blocks
    self.blocks = []
    for i, channels in enumerate(conv_channels):
      if isinstance(kernel_shapes, int):
        extra_kernel_shapes = 0
      else:
        extra_kernel_shapes = [3] * (len(channels) - len(kernel_shapes))

      self.blocks.append(Conv2DNet(
          output_channels=channels,
          kernel_shapes=kernel_shapes + extra_kernel_shapes,
          strides=[self.scale_factor] + [1] * (len(channels) - 1),
          padding=padding,
          data_format=data_format,
          with_batch_norm=False,
          activate_final=i < num_blocks - 1,
          activation=activation,
          name=f"block_{i}"
      ))

  def spatial_aggregation(self, x: jnp.ndarray) -> jnp.ndarray:
    if self.aggregation_type is None:
      return x
    axis = (1, 2) if self.data_format == "NHWC" else (2, 3)
    if self.aggregation_type == "max":
      return jnp.max(x, axis=axis)
    if self.aggregation_type == "mean":
      return jnp.mean(x, axis=axis)
    if self.aggregation_type == "linear_projection":
      x = x.reshape(x.shape[:-3] + (-1,))
      return hk.Linear(self.output_dim, name="LinearProjection")(x)
    raise NotImplementedError()

  def make_distribution(self, net_output: jnp.ndarray) -> distrax.Distribution:
    if self.distribution_name is None:
      return net_output
    elif self.distribution_name == "diagonal_normal":
      if self.aggregation_type is None:
        split_axis, num_axes = self.data_format.index("C"), 3
      else:
        split_axis, num_axes = 1, 1
      # Add an extra axis if the input has more than 1 batch dimension
      split_axis += net_output.ndim - num_axes - 1
      loc, log_scale = jnp.split(net_output, 2, axis=split_axis)
      return distrax.Normal(loc, jnp.exp(log_scale))
    else:
      raise NotImplementedError()

  def __call__(
      self,
      inputs: jnp.ndarray,
      is_training: bool
  ) -> Union[jnp.ndarray, distrax.Distribution]:
    # Treat any extra dimensions (like time) as the batch
    batched_shape = inputs.shape[:-3]
    net = jnp.reshape(inputs, (-1,) + inputs.shape[-3:])

    # Apply all blocks in sequence
    for block in self.blocks:
      net = block(net, is_training=is_training)

    # Final projection
    net = self.spatial_aggregation(net)

    # Reshape back to correct dimensions (like batch + time)
    net = jnp.reshape(net, batched_shape + net.shape[1:])

    # Return a distribution over the observations
    return self.make_distribution(net)


class SpatialConvDecoder(hk.Module):
  """Spatial Convolutional Decoder for learning the Hamiltonian."""

  def __init__(
      self,
      initial_spatial_shape: Sequence[int],
      conv_channels: Union[Sequence[int], int],
      num_blocks: int,
      max_de_aggregation_dims: int,
      blocks_depth: int = 2,
      scale_factor: int = 2,
      output_channels: int = 3,
      h_const_channels: int = 2,
      data_format: str = "NHWC",
      activation: Activation = "leaky_relu",
      learned_sigma: bool = False,
      de_aggregation_type: Optional[str] = None,
      final_activation: Activation = "sigmoid",
      discard_half_de_aggregated: bool = False,
      kernel_shapes: Union[Sequence[int], int] = 3,
      padding: Union[Sequence[str], str] = "SAME",
      name: Optional[str] = None):
    super().__init__(name=name)
    if de_aggregation_type not in (None, "tile", "linear_projection"):
      raise ValueError(f"Unrecognized de_aggregation_type="
                       f"{de_aggregation_type}.")
    self.num_blocks = num_blocks
    self.scale_factor = scale_factor
    self.h_const_channels = h_const_channels
    self.data_format = data_format
    self.learned_sigma = learned_sigma
    self.initial_spatial_shape = tuple(initial_spatial_shape)
    self.final_activation = utils.get_activation(final_activation)
    self.de_aggregation_type = de_aggregation_type
    self.max_de_aggregation_dims = max_de_aggregation_dims
    self.discard_half_de_aggregated = discard_half_de_aggregated

    if isinstance(conv_channels, int):
      conv_channels = [[conv_channels] * blocks_depth
                       for _ in range(num_blocks)]
      conv_channels[-1] += [output_channels]
    else:
      assert isinstance(conv_channels, (list, tuple))
      assert len(conv_channels) == num_blocks
      conv_channels = list(list(c) for c in conv_channels)
      conv_channels[-1].append(output_channels)

    # Convolutional blocks
    self.blocks = []
    for i, channels in enumerate(conv_channels):
      is_final_block = i == num_blocks - 1
      self.blocks.append(
          Conv2DNet(  # pylint: disable=g-complex-comprehension
              output_channels=channels,
              kernel_shapes=kernel_shapes,
              strides=1,
              padding=padding,
              data_format=data_format,
              with_batch_norm=False,
              activate_final=not is_final_block,
              activation=activation,
              name=f"block_{i}"
          ))

  def spatial_de_aggregation(self, x: jnp.ndarray) -> jnp.ndarray:
    if self.de_aggregation_type is None:
      assert x.ndim >= 4
      if self.data_format == "NHWC":
        assert x.shape[1:3] == self.initial_spatial_shape
      elif self.data_format == "NCHW":
        assert x.shape[2:4] == self.initial_spatial_shape
      return x
    elif self.de_aggregation_type == "linear_projection":
      assert x.ndim == 2
      n, d = x.shape
      d = min(d, self.max_de_aggregation_dims or d)
      out_d = d * self.initial_spatial_shape[0] * self.initial_spatial_shape[1]
      x = hk.Linear(out_d, name="LinearProjection")(x)
      if self.data_format == "NHWC":
        shape = (n,) + self.initial_spatial_shape + (d,)
      else:
        shape = (n, d) + self.initial_spatial_shape
      return x.reshape(shape)
    elif self.de_aggregation_type == "tile":
      assert x.ndim == 2
      if self.data_format == "NHWC":
        repeats = (1,) + self.initial_spatial_shape + (1,)
        x = x[:, None, None, :]
      else:
        repeats = (1, 1) + self.initial_spatial_shape
        x = x[:, :, None, None]
      return jnp.tile(x, repeats)
    else:
      raise NotImplementedError()

  def add_constant_channels(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # --------------------------------------------
    # This is purely for TF compatibility purposes
    if self.discard_half_de_aggregated:
      axis = self.data_format.index("C")
      inputs, _ = jnp.split(inputs, 2, axis=axis)
    # --------------------------------------------

    # An extra constant channels
    if self.data_format == "NHWC":
      h_shape = self.initial_spatial_shape + (self.h_const_channels,)
    else:
      h_shape = (self.h_const_channels,) + self.initial_spatial_shape
    h_const = hk.get_parameter("h", h_shape, dtype=inputs.dtype,
                               init=hk.initializers.Constant(1))
    h_const = jnp.tile(h_const, reps=[inputs.shape[0], 1, 1, 1])
    return jnp.concatenate([h_const, inputs], axis=self.data_format.index("C"))

  def make_distribution(self, net_output: jnp.ndarray) -> distrax.Distribution:
    if self.learned_sigma:
      init = hk.initializers.Constant(- jnp.log(2.0) / 2.0)
      log_scale = hk.get_parameter("log_scale", shape=(),
                                   dtype=net_output.dtype, init=init)
      scale = jnp.full_like(net_output, jnp.exp(log_scale))
    else:
      scale = jnp.full_like(net_output, 1 / jnp.sqrt(2.0))

    return distrax.Normal(net_output, scale)

  def __call__(
      self,
      inputs: jnp.ndarray,
      is_training: bool
  ) -> distrax.Distribution:
    # Apply the spatial de-aggregation
    inputs = self.spatial_de_aggregation(inputs)

    # Add the parameterized constant channels
    net = self.add_constant_channels(inputs)

    # Apply all the blocks
    for block in self.blocks:
      # Up-sample the image
      net = utils.nearest_neighbour_upsampling(net, self.scale_factor)
      # Apply the convolutional block
      net = block(net, is_training=is_training)

    # Apply any specific output nonlinearity
    net = self.final_activation(net)

    # Construct the distribution over the observations
    return self.make_distribution(net)


def make_flexible_net(
    net_type: str,
    output_dims: int,
    conv_channels: Union[Sequence[int], int],
    num_units: Union[Sequence[int], int],
    num_layers: Optional[int],
    activation: Activation,
    activate_final: bool = False,
    kernel_shapes: Union[Sequence[int], int] = 3,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[Sequence[str], str] = "SAME",
    name: Optional[str] = None,
    **unused_kwargs: Mapping[str, Any]
):
  """Commonly used for creating a flexible network."""
  if unused_kwargs:
    logging.warning("Unused kwargs of `make_flexible_net`: %s",
                    str(unused_kwargs))
  if net_type == "mlp":
    if isinstance(num_units, int):
      assert num_layers is not None
      num_units = [num_units] * (num_layers - 1) + [output_dims]
    else:
      num_units = list(num_units) + [output_dims]
    return DenseNet(
        num_units=num_units,
        activation=activation,
        activate_final=activate_final,
        name=name
    )
  elif net_type == "conv":
    if isinstance(conv_channels, int):
      assert num_layers is not None
      conv_channels = [conv_channels] * (num_layers - 1) + [output_dims]
    else:
      conv_channels = list(conv_channels) + [output_dims]
    return Conv2DNet(
        output_channels=conv_channels,
        kernel_shapes=kernel_shapes,
        strides=strides,
        padding=padding,
        activation=activation,
        activate_final=activate_final,
        name=name
    )
  elif net_type == "transformer":
    raise NotImplementedError()
  else:
    raise ValueError(f"Unrecognized net_type={net_type}.")


def make_flexible_recurrent_net(
    core_type: str,
    net_type: str,
    output_dims: int,
    num_units: Union[Sequence[int], int],
    num_layers: Optional[int],
    activation: Activation,
    activate_final: bool = False,
    name: Optional[str] = None,
    **unused_kwargs
):
  """Commonly used for creating a flexible recurrences."""
  if net_type != "mlp":
    raise ValueError("We do not support convolutional recurrent nets atm.")
  if unused_kwargs:
    logging.warning("Unused kwargs of `make_flexible_recurrent_net`: %s",
                    str(unused_kwargs))

  if isinstance(num_units, (list, tuple)):
    num_units = list(num_units) + [output_dims]
    num_layers = len(num_units)
  else:
    assert num_layers is not None
    num_units = [num_units] * (num_layers - 1) + [output_dims]
  name = name or f"{core_type.upper()}"

  activation = utils.get_activation(activation)
  core_list = []
  for i, n in enumerate(num_units):
    if core_type.lower() == "vanilla":
      core_list.append(hk.VanillaRNN(hidden_size=n, name=f"{name}_{i}"))
    elif core_type.lower() == "lstm":
      core_list.append(hk.LSTM(hidden_size=n, name=f"{name}_{i}"))
    elif core_type.lower() == "gru":
      core_list.append(hk.GRU(hidden_size=n, name=f"{name}_{i}"))
    else:
      raise ValueError(f"Unrecognized core_type={core_type}.")
    if i != num_layers - 1:
      core_list.append(activation)
  if activate_final:
    core_list.append(activation)

  return hk.DeepRNN(core_list, name="RNN")
