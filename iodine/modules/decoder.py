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
"""Decoders for rendering images."""
# pylint: disable=missing-docstring
from iodine.modules.distributions import MixtureParameters
import shapeguard
import sonnet as snt


class ComponentDecoder(snt.AbstractModule):

  def __init__(self, pixel_decoder, name="component_decoder"):
    super().__init__(name=name)
    self._pixel_decoder = pixel_decoder
    self._sg = shapeguard.ShapeGuard()

  def set_output_shapes(self, pixel, mask):
    self._sg.guard(pixel, "K, H, W, Cp")
    self._sg.guard(mask, "K, H, W, 1")
    self._pixel_decoder.set_output_shapes(self._sg["H, W, 1 + Cp"])

  def _build(self, z):
    self._sg.guard(z, "B, K, Z")
    z_flat = self._sg.reshape(z, "B*K, Z")
    pixel_params = self._pixel_decoder(z_flat).params

    self._sg.guard(pixel_params, "B*K, H, W, 1 + Cp")
    mask_params = pixel_params[..., 0:1]
    pixel_params = pixel_params[..., 1:]

    output = MixtureParameters(
        pixel=self._sg.reshape(pixel_params, "B, K, H, W, Cp"),
        mask=self._sg.reshape(mask_params, "B, K, H, W, 1"),
    )

    del self._sg.B
    return output
