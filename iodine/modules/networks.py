# Copyright 2019 Deepmind Technologies Limited.
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
"""Network modules."""
# pylint: disable=g-multiple-import, g-doc-args, g-short-docstring-punctuation
# pylint: disable=g-no-space-after-docstring-summary
from iodine.modules.distributions import FlatParameters
from iodine.modules.utils import flatten_all_but_last, get_act_func
import numpy as np
import shapeguard
import sonnet as snt
import tensorflow.compat.v1 as tf


class CNN(snt.AbstractModule):
  """ConvNet2D followed by an MLP.

  This is a typical encoder architecture for VAEs, and has been found to work
  well. One small improvement is to append coordinate channels on the input,
  though for most datasets the improvement obtained is negligible.
  """

  def __init__(self, cnn_opt, mlp_opt, mode="flatten", name="cnn"):
    """Constructor.

        Args:
          cnn_opt: Dictionary. Kwargs for the cnn. See vae_lib.ConvNet2D for
            details.
          mlp_opt: Dictionary. Kwargs for the mlp. See vae_lib.MLP for details.
          name: String. Optional name.
    """
    super().__init__(name=name)
    if "activation" in cnn_opt:
      cnn_opt["activation"] = get_act_func(cnn_opt["activation"])
    self._cnn_opt = cnn_opt

    if "activation" in mlp_opt:
      mlp_opt["activation"] = get_act_func(mlp_opt["activation"])
    self._mlp_opt = mlp_opt

    self._mode = mode

  def set_output_shapes(self, shape):
    # assert self._mlp_opt['output_sizes'][-1] is None, self._mlp_opt
    sg = shapeguard.ShapeGuard()
    sg.guard(shape, "1, Y")
    self._mlp_opt["output_sizes"][-1] = sg.Y

  def _build(self, image):
    """Connect model to TensorFlow graph."""
    assert self._mlp_opt["output_sizes"][-1] is not None, "set output_shapes"
    sg = shapeguard.ShapeGuard()
    flat_image, unflatten = flatten_all_but_last(image, n_dims=3)
    sg.guard(flat_image, "B, H, W, C")

    cnn = snt.nets.ConvNet2D(
        activate_final=True,
        paddings=("SAME",),
        normalize_final=False,
        **self._cnn_opt)
    mlp = snt.nets.MLP(**self._mlp_opt)

    # run CNN
    net = cnn(flat_image)

    if self._mode == "flatten":
      # flatten
      net_shape = net.get_shape().as_list()
      flat_shape = net_shape[:-3] + [np.prod(net_shape[-3:])]
      net = tf.reshape(net, flat_shape)
    elif self._mode == "avg_pool":
      net = tf.reduce_mean(net, axis=[1, 2])
    else:
      raise KeyError('Unknown mode "{}"'.format(self._mode))
    # run MLP
    output = sg.guard(mlp(net), "B, Y")
    return FlatParameters(unflatten(output))


class MLP(snt.AbstractModule):
  """MLP."""

  def __init__(self, name="mlp", **mlp_opt):
    super().__init__(name=name)
    if "activation" in mlp_opt:
      mlp_opt["activation"] = get_act_func(mlp_opt["activation"])
    self._mlp_opt = mlp_opt
    assert mlp_opt["output_sizes"][-1] is None, mlp_opt

  def set_output_shapes(self, shape):
    sg = shapeguard.ShapeGuard()
    sg.guard(shape, "1, Y")
    self._mlp_opt["output_sizes"][-1] = sg.Y

  def _build(self, data):
    """Connect model to TensorFlow graph."""
    assert self._mlp_opt["output_sizes"][-1] is not None, "set output_shapes"
    sg = shapeguard.ShapeGuard()
    flat_data, unflatten = flatten_all_but_last(data)
    sg.guard(flat_data, "B, N")

    mlp = snt.nets.MLP(**self._mlp_opt)
    # run MLP
    output = sg.guard(mlp(flat_data), "B, Y")
    return FlatParameters(unflatten(output))


class DeConv(snt.AbstractModule):
  """MLP followed by Deconv net.

  This decoder is commonly used by vanilla VAE models. However, in practice
  BroadcastConv (see below) seems to disentangle slightly better.
  """

  def __init__(self, mlp_opt, cnn_opt, name="deconv"):
    """Constructor.

        Args:
          mlp_opt: Dictionary. Kwargs for vae_lib.MLP.
          cnn_opt: Dictionary. Kwargs for vae_lib.ConvNet2D for the CNN.
          name: Optional name.
    """
    super().__init__(name=name)
    assert cnn_opt["output_channels"][-1] is None, cnn_opt
    if "activation" in cnn_opt:
      cnn_opt["activation"] = get_act_func(cnn_opt["activation"])
    self._cnn_opt = cnn_opt

    if mlp_opt and "activation" in mlp_opt:
      mlp_opt["activation"] = get_act_func(mlp_opt["activation"])
    self._mlp_opt = mlp_opt
    self._target_out_shape = None

  def set_output_shapes(self, shape):
    self._target_out_shape = shape
    self._cnn_opt["output_channels"][-1] = self._target_out_shape[-1]

  def _build(self, z):
    """Connect model to TensorFlow graph."""
    sg = shapeguard.ShapeGuard()
    flat_z, unflatten = flatten_all_but_last(z)
    sg.guard(flat_z, "B, Z")
    sg.guard(self._target_out_shape, "H, W, C")
    mlp = snt.nets.MLP(**self._mlp_opt)
    cnn = snt.nets.ConvNet2DTranspose(
        paddings=("SAME",), normalize_final=False, **self._cnn_opt)
    net = mlp(flat_z)
    output = sg.guard(cnn(net), "B, H, W, C")
    return FlatParameters(unflatten(output))


class BroadcastConv(snt.AbstractModule):
  """MLP followed by a broadcast convolution.

  This decoder takes a latent vector z, (optionally) applies an MLP to it,
  then tiles the resulting vector across space to have dimension [B, H, W, C]
  i.e. tiles across H and W. Then coordinate channels are appended and a
  convolutional layer is applied.
  """

  def __init__(
      self,
      cnn_opt,
      mlp_opt=None,
      coord_type="linear",
      coord_freqs=3,
      name="broadcast_conv",
  ):
    """Args:
          cnn_opt: dict Kwargs for vae_lib.ConvNet2D for the CNN.
          mlp_opt: None or dict If dictionary, then kwargs for snt.nets.MLP. If
            None, then the model will not process the latent vector by an mlp.
          coord_type: ["linear", "cos", None] type of coordinate channels to
            add.
            None: add no coordinate channels.
            linear: two channels with values linearly spaced from -1. to 1. in
              the H and W dimension respectively.
            cos: coord_freqs^2 many channels containing cosine basis functions.
          coord_freqs: int number of frequencies used to construct the cosine
            basis functions (only for coord_type=="cos")
          name: Optional name.
    """
    super().__init__(name=name)

    assert cnn_opt["output_channels"][-1] is None, cnn_opt
    if "activation" in cnn_opt:
      cnn_opt["activation"] = get_act_func(cnn_opt["activation"])
    self._cnn_opt = cnn_opt

    if mlp_opt and "activation" in mlp_opt:
      mlp_opt["activation"] = get_act_func(mlp_opt["activation"])
    self._mlp_opt = mlp_opt

    self._target_out_shape = None
    self._coord_type = coord_type
    self._coord_freqs = coord_freqs

  def set_output_shapes(self, shape):
    self._target_out_shape = shape
    self._cnn_opt["output_channels"][-1] = self._target_out_shape[-1]

  def _build(self, z):
    """Connect model to TensorFlow graph."""
    assert self._target_out_shape is not None, "Call set_output_shape"
    # reshape components into batch dimension before processing them
    sg = shapeguard.ShapeGuard()
    flat_z, unflatten = flatten_all_but_last(z)
    sg.guard(flat_z, "B, Z")
    sg.guard(self._target_out_shape, "H, W, C")

    if self._mlp_opt is None:
      mlp = tf.identity
    else:
      mlp = snt.nets.MLP(activate_final=True, **self._mlp_opt)
    mlp_output = sg.guard(mlp(flat_z), "B, hidden")

    # tile MLP output spatially and append coordinate channels
    broadcast_mlp_output = tf.tile(
        mlp_output[:, tf.newaxis, tf.newaxis],
        multiples=tf.constant(sg["1, H, W, 1"]),
    )  # B, H, W, Z

    dec_cnn_inputs = self.append_coordinate_channels(broadcast_mlp_output)

    cnn = snt.nets.ConvNet2D(
        paddings=("SAME",), normalize_final=False, **self._cnn_opt)
    cnn_outputs = cnn(dec_cnn_inputs)
    sg.guard(cnn_outputs, "B, H, W, C")

    return FlatParameters(unflatten(cnn_outputs))

  def append_coordinate_channels(self, output):
    sg = shapeguard.ShapeGuard()
    sg.guard(output, "B, H, W, C")
    if self._coord_type is None:
      return output
    if self._coord_type == "linear":
      w_coords = tf.linspace(-1.0, 1.0, sg.W)[None, None, :, None]
      h_coords = tf.linspace(-1.0, 1.0, sg.H)[None, :, None, None]
      w_coords = tf.tile(w_coords, sg["B, H, 1, 1"])
      h_coords = tf.tile(h_coords, sg["B, 1, W, 1"])
      return tf.concat([output, h_coords, w_coords], axis=-1)
    elif self._coord_type == "cos":
      freqs = sg.guard(tf.range(0.0, self._coord_freqs), "F")
      valx = tf.linspace(0.0, np.pi, sg.W)[None, None, :, None, None]
      valy = tf.linspace(0.0, np.pi, sg.H)[None, :, None, None, None]
      x_basis = tf.cos(valx * freqs[None, None, None, :, None])
      y_basis = tf.cos(valy * freqs[None, None, None, None, :])
      xy_basis = tf.reshape(x_basis * y_basis, sg["1, H, W, F*F"])
      coords = tf.tile(xy_basis, sg["B,  1, 1, 1"])[..., 1:]
      return tf.concat([output, coords], axis=-1)
    else:
      raise KeyError('Unknown coord_type: "{}"'.format(self._coord_type))


class LSTM(snt.RNNCore):
  """Wrapper around snt.LSTM that supports multi-layers and runs K components in
  parallel.

  Expects input data of shape (B, K, H) and outputs data of shape (B, K, Y)
  """

  def __init__(self, hidden_sizes, name="lstm"):
    super().__init__(name=name)
    self._hidden_sizes = hidden_sizes
    with self._enter_variable_scope():
      self._lstm_layers = [snt.LSTM(hidden_size=h) for h in self._hidden_sizes]

  def initial_state(self, batch_size, **kwargs):
    return [
        lstm.initial_state(batch_size, **kwargs) for lstm in self._lstm_layers
    ]

  def _build(self, data, prev_states):
    assert not self._hidden_sizes or self._hidden_sizes[-1] is not None
    assert len(prev_states) == len(self._hidden_sizes)
    sg = shapeguard.ShapeGuard()
    sg.guard(data, "B, K, H")
    data = sg.reshape(data, "B*K, H")

    out = data
    new_states = []
    for lstm, pstate in zip(self._lstm_layers, prev_states):
      out, nstate = lstm(out, pstate)
      new_states.append(nstate)

    sg.guard(out, "B*K, Y")
    out = sg.reshape(out, "B, K, Y")
    return out, new_states
