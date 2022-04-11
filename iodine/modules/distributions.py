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
"""Collection of sonnet modules that wrap useful distributions."""
# pylint: disable=missing-docstring, g-doc-args, g-short-docstring-punctuation
# pylint: disable=g-space-before-docstring-summary
# pylint: disable=g-no-space-after-docstring-summary
import collections
from iodine.modules.utils import get_act_func
from iodine.modules.utils import get_distribution
import shapeguard
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


tfd = tfp.distributions

FlatParameters = collections.namedtuple("ParameterOut", ["params"])
MixtureParameters = collections.namedtuple("MixtureOut", ["pixel", "mask"])


class DistributionModule(snt.AbstractModule):
  """Distribution Base class supporting shape inference & default priors."""

  def __init__(self, name="distribution"):
    super().__init__(name=name)
    self._output_shape = None

  def set_output_shape(self, shape):
    self._output_shape = shape

  @property
  def output_shape(self):
    return self._output_shape

  @property
  def input_shapes(self):
    raise NotImplementedError()

  def get_default_prior(self, batch_dim=(1,)):
    return self(
        tf.zeros(list(batch_dim) + self.input_shapes.params, dtype=tf.float32))


class BernoulliOutput(DistributionModule):

  def __init__(self, name="bernoulli_output"):
    super().__init__(name=name)

  @property
  def input_shapes(self):
    return FlatParameters(self.output_shape)

  def _build(self, params):
    return tfd.Independent(
        tfd.Bernoulli(logits=params, dtype=tf.float32),
        reinterpreted_batch_ndims=1)


class LocScaleDistribution(DistributionModule):
  """Generic IID location / scale distribution.

    Input parameters are concatenation of location and scale (2*Z,)

    Args:
      dist: Distribution or str Kind of distribution used. Supports Normal,
        Logistic, Laplace, and StudentT distributions.
      dist_kwargs: dict custom keyword arguments for the distribution
      scale_act: function or str or None activation function to be applied to
        the scale input
      scale: str
        different modes for computing the scale:
          * stddev: scale is computed as scale_act(s)
          * var: scale is computed as sqrt(scale_act(s))
          * prec: scale is computed as 1./scale_act(s)
          * fixed: scale is a global variable (same for all pixels) if
            scale_val==-1. then it is a trainable variable initialized to 0.1
            else it is fixed to scale_val (input shape is only (Z,) in this
            case)
      scale_val: float determines the scale value (only used if scale=='fixed').
      loc_act: function or str or None activation function to be applied to the
        location input. Supports optional activation functions for scale and
        location.
    Supports different "modes" for scaling:
      * stddev:
  """

  def __init__(
      self,
      dist=tfd.Normal,
      dist_kwargs=None,
      scale_act=tf.exp,
      scale="stddev",
      scale_val=1.0,
      loc_act=None,
      name="loc_scale_dist",
  ):
    super().__init__(name=name)
    self._scale_act = get_act_func(scale_act)
    self._loc_act = get_act_func(loc_act)
    # supports Normal, Logstic, Laplace, StudentT
    self._dist = get_distribution(dist)
    self._dist_kwargs = dist_kwargs or {}

    assert scale in ["stddev", "var", "prec", "fixed"], scale
    self._scale = scale
    self._scale_val = scale_val

  @property
  def input_shapes(self):
    if self._scale == "fixed":
      param_shape = self.output_shape
    else:
      param_shape = self.output_shape[:-1] + [self.output_shape[-1] * 2]
    return FlatParameters(param_shape)

  def _build(self, params):
    if self._scale == "fixed":
      loc = params
      scale = None  # set later
    else:
      n_channels = params.get_shape().as_list()[-1]
      assert n_channels % 2 == 0
      assert n_channels // 2 == self.output_shape[-1]
      loc = params[..., :n_channels // 2]
      scale = params[..., n_channels // 2:]

    # apply activation functions
    if self._scale != "fixed":
      scale = self._scale_act(scale)
    loc = self._loc_act(loc)

    # apply the correct parametrization
    if self._scale == "var":
      scale = tf.sqrt(scale)
    elif self._scale == "prec":
      scale = tf.reciprocal(scale)
    elif self._scale == "fixed":
      if self._scale_val == -1.0:
        scale_val = tf.get_variable(
            "scale", initializer=tf.constant(0.1, dtype=tf.float32))
      else:
        scale_val = self._scale_val
      scale = tf.ones_like(loc) * scale_val
    # else 'stddev'

    dist = self._dist(loc=loc, scale=scale, **self._dist_kwargs)

    return tfd.Independent(dist, reinterpreted_batch_ndims=1)


class MaskedMixture(DistributionModule):

  def __init__(
      self,
      num_components,
      component_dist,
      mask_activation=None,
      name="masked_mixture",
  ):
    """
        Spatial Mixture Model composed of a categorical masking distribution and
        a custom pixel-wise component distribution (usually logistic or
        gaussian).

        Args:
          num_components: int Number of mixture components >= 2
          component_dist: the distribution to use for the individual components
          mask_activation: str or function or None activation function that
            should be applied to the mask before the softmax.
          name: str
    """

    super().__init__(name=name)
    self._num_components = num_components
    self._dist = component_dist
    self._mask_activation = get_act_func(mask_activation)

  def set_output_shape(self, shape):
    super().set_output_shape(shape)
    self._dist.set_output_shape(shape)

  def _build(self, pixel, mask):
    sg = shapeguard.ShapeGuard()
    # MASKING
    sg.guard(mask, "B, K, H, W, 1")
    mask = tf.transpose(mask, perm=[0, 2, 3, 4, 1])
    mask = sg.reshape(mask, "B, H, W, K")
    mask = self._mask_activation(mask)
    mask = mask[:, tf.newaxis]  # add K=1 axis since K is removed by mixture
    mix_dist = tfd.Categorical(logits=mask)

    # COMPONENTS
    sg.guard(pixel, "B, K, H, W, Cp")
    params = tf.transpose(pixel, perm=[0, 2, 3, 1, 4])
    params = params[:, tf.newaxis]  # add K=1 axis since K is removed by mixture
    dist = self._dist(params)
    return tfd.MixtureSameFamily(
        mixture_distribution=mix_dist, components_distribution=dist)

  @property
  def input_shapes(self):
    pixel = [self._num_components] + self._dist.input_shapes.params
    mask = pixel[:-1] + [1]
    return MixtureParameters(pixel, mask)

  def get_default_prior(self, batch_dim=(1,)):
    pixel = tf.zeros(
        list(batch_dim) + self.input_shapes.pixel, dtype=tf.float32)
    mask = tf.zeros(list(batch_dim) + self.input_shapes.mask, dtype=tf.float32)
    return self(pixel, mask)
