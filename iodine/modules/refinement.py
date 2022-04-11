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
"""Iterative refinement modules."""
# pylint: disable=g-doc-bad-indent, unused-variable
from iodine.modules import utils
import shapeguard
import sonnet as snt
import tensorflow.compat.v1 as tf


class RefinementCore(snt.RNNCore):
  """Recurrent Refinement Module.

    Refinement modules take as inputs:
      * previous state (which could be an arbitrary nested structure)
      * current inputs which include
        * image-space inputs like pixel-based errors, or mask-posteriors
        * latent-space inputs like the previous z_dist, or dz

    They use these inputs to produce:
      * output (usually a new z_dist)
      * new_state
    """

  def __init__(self,
               encoder_net,
               recurrent_net,
               refinement_head,
               name="refinement"):
    super().__init__(name=name)
    self._encoder_net = encoder_net
    self._recurrent_net = recurrent_net
    self._refinement_head = refinement_head
    self._sg = shapeguard.ShapeGuard()

  def initial_state(self, batch_size, **unused_kwargs):
    return self._recurrent_net.initial_state(batch_size)

  def _build(self, inputs, prev_state):
    sg = self._sg
    assert "spatial" in inputs, inputs.keys()
    assert "flat" in inputs, inputs.keys()
    assert "zp" in inputs["flat"], inputs["flat"].keys()
    zp = sg.guard(inputs["flat"]["zp"], "B, K, Zp")

    x = sg.guard(self.prepare_spatial_inputs(inputs["spatial"]), "B*K, H, W, C")
    h1 = sg.guard(self._encoder_net(x).params, "B*K, H1")
    h2 = sg.guard(self.prepare_flat_inputs(h1, inputs["flat"]), "B*K, H2")
    h2_unflattened = sg.reshape(h2, "B, K, H2")
    h3, next_state = self._recurrent_net(h2_unflattened, prev_state)
    sg.guard(h3, "B, K, H3")
    outputs = sg.guard(self._refinement_head(zp, h3), "B, K, Y")

    del self._sg.B
    return outputs, next_state

  def prepare_spatial_inputs(self, inputs):
    values = []
    for name, val in sorted(inputs.items(), key=lambda it: it[0]):
      if val.shape.as_list()[1] == 1:
        self._sg.guard(val, "B, 1, H, W, _C")
        val = tf.tile(val, self._sg["1, K, 1, 1, 1"])
      else:
        self._sg.guard(val, "B, K, H, W, _C")
      values.append(val)
    concat_inputs = self._sg.guard(tf.concat(values, axis=-1), "B, K, H, W, C")
    return self._sg.reshape(concat_inputs, "B*K, H, W, C")

  def prepare_flat_inputs(self, hidden, inputs):
    values = [self._sg.guard(hidden, "B*K, H1")]

    for name, val in sorted(inputs.items(), key=lambda it: it[0]):
      self._sg.guard(val, "B, K, _")
      val_flat = tf.reshape(val, self._sg["B*K"] + [-1])
      values.append(val_flat)
    return tf.concat(values, axis=-1)


class ResHead(snt.AbstractModule):
  """Updates Zp using a residual mechanism."""

  def __init__(self, name="residual_head"):
    super().__init__(name=name)

  def _build(self, zp_old, inputs):
    sg = shapeguard.ShapeGuard()
    sg.guard(zp_old, "B, K, Zp")
    sg.guard(inputs, "B, K, H")
    update = snt.Linear(sg.Zp)

    flat_zp = sg.reshape(zp_old, "B*K, Zp")
    flat_inputs = sg.reshape(inputs, "B*K, H")

    zp = flat_zp + update(flat_inputs)

    return sg.reshape(zp, "B, K, Zp")


class PredictorCorrectorHead(snt.AbstractModule):
  """This refinement head is used for sequential data.

    At every step it computes a prediction from the λ of the previous timestep
    and an update from the refinement network of the current timestep.

    The next step λ' is computed as a gated combination of both:
    λ' = g * λ_corr + (1-g) * λ_pred

    """

  def __init__(
      self,
      hidden_sizes=(64,),
      pred_gate_bias=0.0,
      corrector_gate_bias=0.0,
      activation=tf.nn.elu,
      name="predcorr_head",
  ):
    super().__init__(name=name)
    self._hidden_sizes = hidden_sizes
    self._activation = utils.get_act_func(activation)
    self._pred_gate_bias = pred_gate_bias
    self._corrector_gate_bias = corrector_gate_bias

  def _build(self, zp_old, inputs):
    sg = shapeguard.ShapeGuard()
    sg.guard(zp_old, "B, K, Zp")
    sg.guard(inputs, "B, K, H")
    update = snt.Linear(sg.Zp)
    update_gate = snt.Linear(sg.Zp)
    predict = snt.nets.MLP(
        output_sizes=list(self._hidden_sizes) + [sg.Zp * 2],
        activation=self._activation,
    )

    flat_zp = sg.reshape(zp_old, "B*K, Zp")
    flat_inputs = sg.reshape(inputs, "B*K, H")

    g = tf.nn.sigmoid(update_gate(flat_inputs) + self._corrector_gate_bias)
    u = update(flat_inputs)

    # a slightly more efficient way of computing the gated update
    # (1-g) * flat_zp + g * u
    zp_corrected = flat_zp + g * (u - flat_zp)

    predicted = predict(flat_zp)
    pred_up = predicted[:, :sg.Zp]
    pred_gate = tf.nn.sigmoid(predicted[:, sg.Zp:] + self._pred_gate_bias)

    zp = zp_corrected + pred_gate * (pred_up - zp_corrected)

    return sg.reshape(zp, "B, K, Zp")
