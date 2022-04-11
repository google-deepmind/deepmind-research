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
"""Stochastic Variational inference Auto-Encoder."""
# pylint: disable=unused-variable, g-bad-todo

import collections
from iodine.modules import utils
from multi_object_datasets.segmentation_metrics import adjusted_rand_index
import numpy as np
import shapeguard
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


tfd = tfp.distributions

logging = tf.logging

DEFAULT_INPUTS = (
    "image",
    "zp",
    "mask",
    "components",
    "dmask",
    "dzp",
    "dcomponents",
    "posterior",
    "log_prob",
    "pred_mask",
    "capacity",
    "flat_capacity",
    "coordinates",
    "counterfactual",
)

DEFAULT_PREPROCESSING = [
    "dcomponents", "dmask", "dzp", "log_prob", "counterfactual"
]

DEFAULT_STOP_GRADIENT = ("dzp", "dmask", "dcomponents", "log_prob",
                         "counterfactual")


class IODINE(snt.AbstractModule):
  """Iterative Amortized Variational Autoencoder.

    Args:
        decoder (decoders.ComponentDecoder): The decoder.
        refinement_core (refinement.RefinementCore): The recurrent (refinement)
          encoder.
        latent_dist (distributions.Distribution): The distribution of the latent
          z variables.
        output_dist (distributions.MaskedMixture): The pixel-wise output
          distribution (a spatial mixture).
        n_z (int): Dimensionality of the per-object latents z_k.
        num_components (int): Number of available object slots (K).
        num_iters (int): Number of refinement iterations.
        sequential (bool): Whether the input data is sequential.
        factor_evaluator (factor_eval.FactorRegressor): The factor evaluation
          model that is trained to predict the true factors from the inferred
          latents.
        stop_gradients (List[str]): For which refinement inputs to stop
          gradients from backpropagating through the iterations. (see inputs for
          valid values)
            Default is: ["dcomponents", "dmask", "dzp", "log_prob",
              "counterfactual"]
        iter_loss_weight ("linspace" | float | List[float]): How to weigh the
          loss terms for each timestep.
            Can be:
              "linspace":  Linearly increasing weights from 0 to 1.0.
              float:       A fixed value for all steps.
              List[float]: Manually specify all weight.
        inputs (List[str]): list of inputs to use for the refinement network.
            Can include the following (default is to use all): ["image", "zp",
              "mask", "components", "dmask", "dzp", "dcomponents", "posterior",
              "log_prob", "pred_mask", "capacity", "flat_capacity",
              "coordinates", "counterfactual"]
        preprocess (List[str]): Specifies the subset of inputs that be
          preprocessed with layernorm.
            Default is: ["dcomponents", "dmask", "dzp", "log_prob",
              "counterfactual"]
        coord_type (str): Type of coordinate channels to append to the
          refinement inputs. Can be "linear" (default) or "cos".
        coord_freqs (int): If using cos based coordinate channels, then this
          specifies the number of frequencies used.
        name (str): Name of the module (within the tensorflow graph).
  """

  def __init__(
      self,
      decoder,
      refinement_core,
      latent_dist,
      output_dist,
      n_z,
      num_components,
      num_iters,
      sequential=False,
      factor_evaluator=None,
      stop_gradients=DEFAULT_STOP_GRADIENT,
      iter_loss_weight="linspace",
      inputs=DEFAULT_INPUTS,
      preprocess=None,
      coord_type="linear",
      coord_freqs=3,
      name="iodine",
  ):
    super().__init__(name=name)
    self._sg = shapeguard.ShapeGuard(dims={"K": num_components})
    self.decoder = decoder
    self.refinement_core = refinement_core

    self.latent_dist = latent_dist
    self.output_dist = output_dist

    self.n_z = n_z
    self.num_components = num_components
    self.num_iters = num_iters
    self.sequential = sequential
    self.iter_loss_weights = self._parse_iter_loss_weights(iter_loss_weight)

    self.factor_evaluator = factor_evaluator

    self.stop_gradients = stop_gradients
    self.inputs = inputs
    self.preprocess = DEFAULT_PREPROCESSING if preprocess is None else preprocess
    self.coord_type = coord_type
    self.coord_freqs = coord_freqs

    with self._enter_variable_scope():
      self.latent_dist.set_output_shape([self.n_z])
      logging.info("VAE: z shape: %s", [self.n_z])
      with tf.name_scope("prior"):
        self.prior = self.latent_dist.get_default_prior((self.num_components,))
      self._sg.guard(self.prior, "K, Z")
      with tf.variable_scope("preprocess"):
        self._layernorms = {
            name: snt.LayerNorm(name="layer_norm_" + name)
            for name in self.preprocess
        }

  def _build(self, data):
    data["image"] = data["image"][:, :self.num_iters + 1]
    if "mask" in data:
      data["mask"] = data["mask"][:, :self.num_iters + 1]
    x = self._sg.guard(data["image"], "B, T, H, W, C")
    self._propagate_shape_info(x.get_shape().as_list())

    # run iterative encoder
    iterations = self.encode(x)
    z_dist = self._sg.guard(iterations["z_dist"][-1], "B, K, Z")
    z = self._sg.guard(iterations["z"][-1], "B, K, Z")

    # decode
    x_params, x_dist = self.decode(z)
    iterations["x_dist"].append(self._sg.guard(x_dist, "B, 1, H, W, C"))

    # compute loss
    kl = self._sg.guard(self._raw_kl(z_dist), "B, K")
    img = self._get_image_for_iter(x, self.num_iters)
    re = self._sg.guard(self._reconstruction_error(x_dist, img), "B")
    iterations["kl"].append(kl)
    iterations["re"].append(re)

    iterations["recons_loss"] = [tf.reduce_mean(re) for re in iterations["re"]]

    total_rec_loss = sum([
        w * re
        for w, re in zip(self.iter_loss_weights, iterations["recons_loss"])
    ])
    total_kl_loss = sum([
        w * tf.reduce_mean(tf.reduce_sum(kl, axis=1))
        for w, kl in zip(self.iter_loss_weights, iterations["kl"])
    ])

    total_loss = total_rec_loss + total_kl_loss

    scalars = {
        "loss/kl":
            sum([
                tf.reduce_mean(tf.reduce_sum(kl, axis=1))
                for kl in iterations["kl"]
            ]),
        "loss/recons":
            total_rec_loss,
    }

    if self.factor_evaluator:
      pred_mask = self._sg.guard(x_dist.mixture_distribution.probs,
                                 "B, 1, H, W, K")
      pred_mask = tf.transpose(pred_mask, [0, 4, 2, 3, 1])
      mask_true = self._sg.guard(data["mask"], "B, T, L, H, W, 1")
      mask_true = self._get_image_for_iter(mask_true, self.num_iters)
      mask_true = mask_true[:, 0]

      factor_loss, factor_scalars, _, _, _ = self.factor_evaluator(
          tf.stop_gradient(z),
          data["factors"],
          data["visibility"],
          tf.stop_gradient(pred_mask),
          mask_true,
      )
      total_loss += factor_loss
      scalars["factor/loss"] = factor_loss
      scalars.update({"factor/" + k: v for k, v in factor_scalars.items()})
    scalars["loss/total"] = total_loss

    scalars.update(self._get_monitored_scalars(x_dist, data))
    logging.info(self._sg.dims)
    return total_loss, scalars, iterations

  @snt.reuse_variables
  def encode(self, images):
    sg = self._sg
    sg.guard(images, "B, T, H, W, C")
    zp, z_dist, z = self._get_initial_z()

    iterations = {
        "z": [z],
        "zp": [zp],
        "z_dist": [z_dist],
        "x_dist": [],
        "inputs": [],
        "kl": [],
        "re": [],
    }
    state = self.refinement_core.initial_state(sg["B*K"][0])
    for t in range(self.num_iters):
      img = sg.guard(self._get_image_for_iter(images, t), "B, 1, H, W, C")
      x_params, x_dist = self.decode(z)

      # compute loss
      kl = self._sg.guard(self._raw_kl(z_dist), "B, K")
      re = self._sg.guard(self._reconstruction_error(x_dist, img), "B")
      loss = tf.reduce_mean(re) + tf.reduce_mean(tf.reduce_sum(kl, axis=1))

      inputs = self._get_inputs_for(x_params, x_dist, img, z_dist, zp, loss)
      zp, state = self.refinement_core(inputs, state)
      sg.guard(zp, "B, K, Zp")

      z_dist = sg.guard(self.latent_dist(zp), "B, K, Z")
      z = z_dist.sample()

      # append local variables to iteration collections
      for v, name in zip(
          [z, zp, z_dist, x_dist, inputs, kl, re],
          ["z", "zp", "z_dist", "x_dist", "inputs", "kl", "re"],
      ):
        iterations[name].append(v)

    return iterations

  @snt.reuse_variables
  def decode(self, z):
    sg = shapeguard.ShapeGuard()
    sg.guard(z, "B, K, Z")
    # legacy
    z = tf.concat([z, 5.0 * tf.ones(sg["B, K, 1"], dtype=tf.float32)], axis=2)
    params = self.decoder(z)
    out_dist = self.output_dist(*params)
    return params, out_dist

  @snt.reuse_variables
  def eval(self, data):
    total_loss, scalars, iterations = self._build(data)
    sg = shapeguard.ShapeGuard()

    def get_components(dist):
      return tf.transpose(dist.components_distribution.mean()[:, 0, :, :, :, :],
                          [0, 3, 1, 2, 4])

    def get_mask(dist):
      return tf.transpose(dist.mixture_distribution.probs[:, :, :, :, :],
                          [0, 4, 2, 3, 1])

    def get_mask_logits(dist):
      return tf.transpose(dist.mixture_distribution.logits[:, :, :, :, :],
                          [0, 4, 2, 3, 1])

    def stack_iters(list_of_variables, pad_zero=False):
      if pad_zero:
        list_of_variables.insert(0, tf.zeros_like(list_of_variables[0]))
      return tf.stack(list_of_variables, axis=1)

    # data
    image = sg.guard(data["image"], "B, 1, H, W, C")
    true_mask = sg.guard(data["mask"], "B, 1, L, H, W, 1")
    visibility = sg.guard(data["visibility"], "B, L")
    factors = data["factors"]

    # inputs
    inputs_flat = {
        k: stack_iters([inp["flat"][k] for inp in iterations["inputs"]],
                       pad_zero=True)
        for k in iterations["inputs"][0]["flat"].keys()
    }
    inputs_spatial = {
        k: stack_iters([inp["spatial"][k] for inp in iterations["inputs"]],
                       pad_zero=True)
        for k in iterations["inputs"][0]["spatial"].keys()
    }
    # latent
    z = sg.guard(stack_iters(iterations["z"]), "B, T, K, Z")
    z_mean = stack_iters([zd.mean() for zd in iterations["z_dist"]])
    z_std = stack_iters([zd.stddev() for zd in iterations["z_dist"]])
    # outputs
    recons = stack_iters([xd.mean() for xd in iterations["x_dist"]])
    pred_mask = stack_iters([get_mask(xd) for xd in iterations["x_dist"]])
    pred_mask_logits = stack_iters(
        [get_mask_logits(xd) for xd in iterations["x_dist"]])
    components = stack_iters(
        [get_components(xd) for xd in iterations["x_dist"]])

    # metrics
    tm = tf.transpose(true_mask[..., 0], [0, 1, 3, 4, 2])
    tm = tf.reshape(tf.tile(tm, sg["1, T, 1, 1, 1"]), sg["B * T, H * W, L"])
    pm = tf.transpose(pred_mask[..., 0], [0, 1, 3, 4, 2])
    pm = tf.reshape(pm, sg["B * T, H * W, K"])
    ari = tf.reshape(adjusted_rand_index(tm, pm), sg["B, T"])
    ari_nobg = tf.reshape(adjusted_rand_index(tm[..., 1:], pm), sg["B, T"])

    mse = tf.reduce_mean(tf.square(recons - image[:, None]), axis=[2, 3, 4, 5])

    # losses
    loss_recons = stack_iters(iterations["re"])
    kl = stack_iters(iterations["kl"])

    info = {
        "data": {
            "image": sg.guard(image, "B, 1, H, W, C"),
            "true_mask": sg.guard(true_mask, "B, 1, L, H, W, 1"),
            "visibility": sg.guard(visibility, "B, L"),
            "factors": factors,
        },
        "inputs": {
            "flat": inputs_flat,
            "spatial": inputs_spatial
        },
        "latent": {
            "z": sg.guard(z, "B, T, K, Z"),
            "z_mean": sg.guard(z_mean, "B, T, K, Z"),
            "z_std": sg.guard(z_std, "B, T, K, Z"),
        },
        "outputs": {
            "recons": sg.guard(recons, "B, T, 1, H, W, C"),
            "pred_mask": sg.guard(pred_mask, "B, T, K, H, W, 1"),
            "pred_mask_logits": sg.guard(pred_mask_logits, "B, T, K, H, W, 1"),
            "components": sg.guard(components, "B, T, K, H, W, C"),
        },
        "losses": {
            "total": total_loss,
            "recons": sg.guard(loss_recons, "B, T"),
            "kl": sg.guard(kl, "B, T, K"),
        },
        "metrics": {
            "ari": ari,
            "ari_nobg": ari_nobg,
            "mse": mse
        },
    }

    if self.factor_evaluator:
      # factor evaluation information
      factor_info = {
          "loss": [],
          "metrics": collections.defaultdict(list),
          "predictions": collections.defaultdict(list),
          "assignment": [],
      }
      for t in range(z.get_shape().as_list()[1]):
        floss, fscalars, _, fpred, fass = self.factor_evaluator(
            z[:, t], factors, visibility, pred_mask[:, t], true_mask[:, 0])
        factor_info["loss"].append(floss)
        factor_info["assignment"].append(fass)
        for k in fpred:
          factor_info["predictions"][k].append(
              tf.reduce_sum(fpred[k] * fass[..., None], axis=2))
          factor_info["metrics"][k].append(fscalars[k])

      info["losses"]["factor"] = sg.guard(tf.stack(factor_info["loss"]), "T")
      info["factor_regressor"] = {
          "assignment":
              sg.guard(stack_iters(factor_info["assignment"]), "B, T, L, K"),
          "metrics": {
              k: tf.stack(factor_info["metrics"][k], axis=0)
              for k in factor_info["metrics"]
          },
          "predictions": {
              k: stack_iters(factor_info["predictions"][k])
              for k in factor_info["predictions"]
          },
      }

    return info

  @snt.reuse_variables
  def get_sample_images(self, nr_samples=16):
    with tf.name_scope("prior_samples"):
      prior_z = self.prior.sample(nr_samples)
      _, prior_out = self.decode(prior_z)
      prior_out = tf.clip_by_value(prior_out.mean(), 0.0, 1.0)
    return utils.images_to_grid(prior_out[:, 0])[tf.newaxis]

  @snt.reuse_variables
  def get_overview_images(self, data, nr_images=4, mask_components=False):
    x = data["image"][:nr_images, :self.num_iters + 1]
    old_b, self._sg.B = self._sg.B, x.get_shape().as_list()[0]

    iterations = self.encode(x)
    z = iterations["z"][-1]
    _, x_dist = self.decode(z)
    self._sg.B = old_b
    t = min(self.num_iters, x.get_shape().as_list()[1]) - 1
    # iterations view
    recons = tf.stack([x_dist.mean() for x_dist in iterations["x_dist"]],
                      axis=1)
    masks = tf.stack(
        [
            tf.transpose(x_dist.mixture_distribution.probs, [0, 4, 2, 3, 1])
            for x_dist in iterations["x_dist"]
        ],
        axis=1,
    )

    return {
        "overview":
            utils.get_overview_image(
                x[:, t:t + 1], x_dist, mask_components=mask_components),
        "sequence":
            utils.construct_iterations_image(x[:, :t + 1, tf.newaxis], recons,
                                             masks),
        "samples":
            self.get_sample_images(),
    }

  def _get_initial_z(self):
    # Initial z distribution
    zp_init = tf.get_variable(
        "initial_sample_distribution",
        shape=self.latent_dist.input_shapes.params,
        dtype=tf.float32,
    )
    zp = tf.tile(zp_init[tf.newaxis, tf.newaxis], self._sg["B, K, 1"])

    z_dist = self.latent_dist(zp)
    z = z_dist.sample()

    self._sg.guard(zp, "B, K, Zp")
    self._sg.guard(z_dist, "B, K, Z")
    self._sg.guard(z, "B, K, Z")

    return zp, z_dist, z

  def _parse_iter_loss_weights(self, iter_loss_weight):
    if iter_loss_weight == "linspace":
      iter_weights = np.linspace(0.0, 1.0, self.num_iters + 1).tolist()
    elif isinstance(iter_loss_weight, (float, int)):
      iter_weights = [float(iter_loss_weight)] * (self.num_iters + 1)
    elif isinstance(iter_loss_weight, (tuple, list)):
      iter_weights = [float(w) for w in iter_loss_weight]
    else:
      raise ValueError("Unknown iter_loss_weight type {}.".format(
          repr(iter_loss_weight)))
    assert len(iter_weights) == (self.num_iters + 1), iter_loss_weight
    return iter_weights

  def _propagate_shape_info(self, image_shape):
    image_shape = image_shape[-3:]  # ignore batch dims
    logging.info("VAE: image shape: %s", image_shape)
    z_param_shape = self._sg.guard(self.latent_dist.input_shapes.params, "Zp")
    logging.info("VAE: z parameter shape: %s", z_param_shape)
    self.output_dist.set_output_shape(image_shape)
    out_param_shapes = self.output_dist.input_shapes
    logging.info("VAE: output parameter shapes: %s", out_param_shapes)
    self.decoder.set_output_shapes(*out_param_shapes)

  def _get_image_for_iter(self, images, t):
    """Return current frame or first image."""
    if self.sequential:
      return images[:, t:t + 1]
    else:
      return images[:, :1]

  @staticmethod
  def _get_mask_posterior(out_dist, img):
    p_comp = out_dist.components_distribution.prob(img[..., tf.newaxis, :])
    posterior = p_comp / (tf.reduce_sum(p_comp, axis=-1, keepdims=True) + 1e-6)
    return tf.transpose(posterior, [0, 4, 2, 3, 1])

  def _get_inputs_for(self, out_params, out_dist, img, z_dist, zp, loss):
    sg = self._sg
    # gradients of loss wrt z, components and mask
    dzp, dxp, dmp = tf.gradients(loss, [zp, out_params.pixel, out_params.mask])

    log_prob = sg.guard(
        out_dist.log_prob(img)[..., tf.newaxis], "B, 1, H, W, 1")

    counterfactual_log_probs = []
    for k in range(0, self.num_components):
      mask = tf.concat([out_params.mask[:, :k], out_params.mask[:, k + 1:]],
                       axis=1)
      pixel = tf.concat([out_params.pixel[:, :k], out_params.pixel[:, k + 1:]],
                        axis=1)
      out_dist_k = self.output_dist(pixel, mask)
      log_prob_k = out_dist_k.log_prob(img)[..., tf.newaxis]
      counterfactual_log_probs.append(log_prob_k)
    counterfactual = log_prob - tf.concat(counterfactual_log_probs, axis=1)

    pred_mask = tf.transpose(out_dist.mixture_distribution.probs,
                             [0, 4, 2, 3, 1])

    potential_inputs = {
        # spatial
        "image":
            sg.guard(img, "B, 1, H, W, C"),
        "log_prob":
            sg.guard(log_prob, "B, 1, H, W, 1"),
        "mask":
            sg.guard(out_params.mask, "B, K, H, W, 1"),
        "pred_mask":
            sg.guard(pred_mask, "B, K, H, W, 1"),
        "components":
            sg.guard(out_params.pixel, "B, K, H, W, Cp"),
        "dmask":
            sg.guard(dmp, "B, K, H, W, Mp"),
        "dcomponents":
            sg.guard(dxp, "B, K, H, W, Cp"),
        "posterior":
            sg.guard(self._get_mask_posterior(out_dist, img), "B, K, H, W, 1"),
        "capacity":
            0.5 *
            tf.ones(sg["B, K, H, W, 1"], dtype=tf.float32),  # TODO: legacy
        "coordinates":
            self._get_coord_channels(),
        "counterfactual":
            self._sg.guard(counterfactual, "B, K, H, W, 1"),
        # flat
        "zp":
            sg.guard(zp, "B, K, Zp"),
        "dzp":
            sg.guard(dzp, "B, K, Zp"),
        "flat_capacity":
            0.5 * tf.ones(sg["B, K, 1"], dtype=tf.float32),  # TODO: legacy
    }

    # collect used inputs, stop gradients and preprocess where needed
    final_inputs = {"spatial": {}, "flat": {}}
    for k, v in potential_inputs.items():
      # skip unused inputs
      if k not in self.inputs:
        continue
      # stop gradients
      if k in self.stop_gradients:
        v = tf.stop_gradient(v)
      # preprocess
      v = self._apply_preprocessing(k, v)
      # sort into flat / spatial according to their shape
      structure = "flat" if len(v.get_shape().as_list()) == 3 else "spatial"
      final_inputs[structure][k] = v

    return final_inputs

  def _apply_preprocessing(self, name, val):
    if name in self.preprocess:
      if self._sg.matches(val, "B, K, _z"):
        flat_val = tf.reshape(val, self._sg["B*K"] + [-1])
      elif self._sg.matches(val, "B, 1, _z"):
        flat_val = val[:, 0, :]
      elif self._sg.matches(val, "B, K, H, W, _c"):
        flat_val = tf.reshape(val, self._sg["B*K, H*W"] + [-1])
      elif self._sg.matches(val, "B, 1, H, W, _c"):
        flat_val = tf.reshape(val, self._sg["B, H*W"] + [-1])
      else:
        raise ValueError("Cannot handle shape {}".format(
            val.get_shape().as_list()))
      ln = self._layernorms[name]
      norm_val = ln(flat_val)
      return tf.reshape(norm_val, val.shape.as_list())
    else:
      return val

  def _get_coord_channels(self):
    if self.coord_type == "linear":
      x_coords = tf.linspace(-1.0, 1.0, self._sg.W)[None, None, None, :, None]
      y_coords = tf.linspace(-1.0, 1.0, self._sg.H)[None, None, :, None, None]
      x_coords = tf.tile(x_coords, self._sg["B, 1, H, 1, 1"])
      y_coords = tf.tile(y_coords, self._sg["B, 1, 1, W, 1"])
      return tf.concat([x_coords, y_coords], axis=-1)
    elif self.coord_type == "cos":
      freqs = self._sg.guard(tf.range(0.0, self.coord_freqs), "F")
      valx = tf.linspace(0.0, np.pi, self._sg.W)[None, None, None, :, None,
                                                 None]
      valy = tf.linspace(0.0, np.pi, self._sg.H)[None, None, :, None, None,
                                                 None]
      x_basis = tf.cos(valx * freqs[None, None, None, None, :, None])
      y_basis = tf.cos(valy * freqs[None, None, None, None, None, :])
      xy_basis = tf.reshape(x_basis * y_basis, self._sg["1, 1, H, W, F*F"])
      coords = tf.tile(xy_basis, self._sg["B, 1, 1, 1, 1"])[..., 1:]
      return coords
    else:
      raise KeyError('Unknown coord_type: "{}"'.format(self.coord_type))

  def _raw_kl(self, z_dist):
    return tfd.kl_divergence(z_dist, self.prior)

  def _reconstruction_error(self, x_dist, img):
    log_prob = self._sg.guard(x_dist.log_prob(img), "B, 1, H, W")
    return -tf.reduce_sum(log_prob, axis=[1, 2, 3])

  def _get_monitored_scalars(self, out_dist, data):
    self._sg.guard(out_dist, "B, 1, H, W, C")
    img = self._get_image_for_iter(data["image"], self.num_iters)
    scalars = {}
    with tf.name_scope("monitored_scalars"):
      # ######### Loss Monitoring #########
      scalars["loss/mse"] = tf.losses.mean_squared_error(
          img, out_dist.mean())

      # ########## Mask Monitoring #######
      if "mask" in data:
        true_mask = self._sg.guard(data["mask"], "B, T, L, H, W, 1")
        true_mask = tf.transpose(true_mask[:, -1, ..., 0], [0, 2, 3, 1])
        true_mask = self._sg.reshape(true_mask, "B, H*W, L")
      else:
        true_mask = None

      pred_mask = self._sg.guard(out_dist.mixture_distribution.probs,
                                 "B, 1, H, W, K")
      pred_mask = self._sg.reshape(pred_mask, "B, H*W, K")

      if pred_mask is not None and true_mask is not None:
        self._sg.guard(pred_mask, "B, H*W, K")
        self._sg.guard(true_mask, "B, H*W, L")
        scalars["loss/ari"] = tf.reduce_mean(
            adjusted_rand_index(true_mask, pred_mask))

        scalars["loss/ari_nobg"] = tf.reduce_mean(
            adjusted_rand_index(true_mask[..., 1:], pred_mask))

      return scalars
