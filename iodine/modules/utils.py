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
"""Utilities for IODINE."""
# pylint: disable=g-doc-bad-indent, g-doc-return-or-yield, g-doc-args
# pylint: disable=missing-docstring
import importlib
import math
from absl import logging
from matplotlib.colors import hsv_to_rgb
import numpy as np
import shapeguard
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


tfd = tfp.distributions

ACT_FUNCS = {
    "identity": tf.identity,
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh,
    "relu": tf.nn.relu,
    "elu": tf.nn.elu,
    "selu": tf.nn.selu,
    "softplus": tf.nn.softplus,
    "exp": tf.exp,
    "softmax": tf.nn.softmax,
}


def get_act_func(name_or_func):
  if name_or_func is None:
    return tf.identity
  if callable(name_or_func):
    return name_or_func
  elif isinstance(name_or_func, str):
    return ACT_FUNCS[name_or_func.lower()]
  else:
    raise KeyError(
        'Unknown activation function "{}" of type {}"'.format(
            name_or_func, type(name_or_func)
        )
    )


DISTS = {
    "normal": tfd.Normal,
    "log_normal": tfd.LogNormal,
    "laplace": tfd.Laplace,
    "logistic": tfd.Logistic,
}


def get_distribution(name_or_dist):
  if isinstance(name_or_dist, type(tfd.Normal)):
    return name_or_dist
  elif isinstance(name_or_dist, str):
    return DISTS[name_or_dist.lower()]
  raise KeyError(
      'Unknown distribution "{}" of type {}"'.format(name_or_dist,
                                                     type(name_or_dist)))


def get_mask_plot_colors(nr_colors):
  """Get nr_colors uniformly spaced hues to plot mask values."""
  hsv_colors = np.ones((nr_colors, 3), dtype=np.float32)
  hsv_colors[:, 0] = np.linspace(0, 1, nr_colors, endpoint=False)
  color_conv = hsv_to_rgb(hsv_colors)
  return color_conv


def color_transform(masks):
  with tf.name_scope("color_transform"):
    n_components = masks.shape.as_list()[-1]
    colors = tf.constant(get_mask_plot_colors(n_components), name="mask_colors")
    return tf.tensordot(masks, colors, axes=1)


def construct_diagnostic_image(
    images,
    recons,
    masks,
    components,
    border_width=2,
    nr_images=6,
    clip=True,
    mask_components=False,
    ):
  """Construct a single image containing image, recons., mask, and components.

    Args:
      images: (B, H, W, C)
      recons: (B, H, W, C)
      masks: (B, H, W, K)
      components: (B, H, W, K, C)
      border_width: int. width of the border in pixels. (default=2)
      nr_images: int. Number of images to include. (default=6)
      clip: bool. Whether to clip the final image to range [0, 1].

    Returns:
      diag_images: (nr, H+border_width*2, (W+border_width*2) * (K+3), 3)
    """
  with tf.name_scope("diagnostic_image"):
    # transform the masks into RGB images
    recolored_masks = color_transform(masks[:nr_images])
    if images.get_shape().as_list()[-1] == 1:
      # deal with grayscale images
      images = tf.tile(images[:nr_images], [1, 1, 1, 3])
      recons = tf.tile(recons[:nr_images], [1, 1, 1, 3])
      components = tf.tile(components[:nr_images], [1, 1, 1, 1, 3])

    if mask_components:
      components *= masks[:nr_images, ..., tf.newaxis]

    # Pad everything
    no_pad, pad = (0, 0), (border_width, border_width)
    paddings = tf.constant([no_pad, pad, pad, no_pad])
    paddings_components = tf.constant([no_pad, pad, pad, no_pad, no_pad])
    pad_images = tf.pad(images[:nr_images], paddings, constant_values=0.5)
    pad_recons = tf.pad(recons[:nr_images], paddings, constant_values=0.5)
    pad_masks = tf.pad(recolored_masks, paddings, constant_values=1.0)
    pad_components = tf.pad(
        components[:nr_images], paddings_components, constant_values=0.5
    )

    # reshape components into single wide image
    pad_components = tf.transpose(pad_components, [0, 1, 3, 2, 4])
    pc_shape = pad_components.shape.as_list()
    pc_shape[2] = pc_shape[2] * pc_shape.pop(3)
    pad_components = tf.reshape(pad_components, pc_shape)

    # concatenate all parts along width
    diag_imgs = tf.concat(
        [pad_images, pad_recons, pad_masks, pad_components], axis=2
    )
    # concatenate all images along height
    diag_shape = diag_imgs.shape.as_list()
    final_img = tf.reshape(diag_imgs, [1, -1, diag_shape[2], diag_shape[3]])
    if clip:
      final_img = tf.clip_by_value(final_img, 0.0, 1.0)
    return final_img


def construct_reconstr_image(images, recons, border_width=2,
                             nr_images=6, clip=True):
  """Construct a single image containing image, and recons.

    Args:
      images: (B, H, W, C)
      recons: (B, H, W, C)
      border_width: int. width of the border in pixels. (default=2)
      nr_images: int. Number of images to include. (default=6)
      clip: bool. Whether to clip the final image to range [0, 1].

    Returns:
      rec_images: (nr, H+border_width*2, (W+border_width*2) * 2, 3)
    """
  with tf.name_scope("diagnostic_image"):
    # Pad everything
    no_pad, pad = (0, 0), (border_width, border_width)
    paddings = tf.constant([no_pad, pad, pad, no_pad])
    pad_images = tf.pad(images[:nr_images], paddings, constant_values=0.5)
    pad_recons = tf.pad(recons[:nr_images], paddings, constant_values=0.5)
    # concatenate all parts along width
    diag_imgs = tf.concat([pad_images, pad_recons], axis=2)
    # concatenate all images along height
    diag_shape = diag_imgs.shape.as_list()
    final_img = tf.reshape(diag_imgs, [1, -1, diag_shape[2], diag_shape[3]])
    if clip:
      final_img = tf.clip_by_value(final_img, 0.0, 1.0)
    return final_img


def construct_iterations_image(
    images, recons, masks, border_width=2, nr_seqs=2, clip=True
):
  """Construct a single image containing image, and recons.

    Args:
      images: (B, T, 1, H, W, C)
      recons: (B, T, 1, H, W, C)
      masks:  (B, T, K, H, W, 1)
      border_width: int. width of the border in pixels. (default=2)
      nr_seqs: int. Number of sequences to include. (default=2)
      clip: bool. Whether to clip the final image to range [0, 1].

    Returns:
      rec_images: (nr, H+border_width*2, (W+border_width*2) * 2, 3)
    """
  sg = shapeguard.ShapeGuard()
  sg.guard(recons, "B, T, 1, H, W, C")
  if images.get_shape().as_list()[1] == 1:
    images = tf.tile(images, sg["1, T, 1, 1, 1, 1"])
  sg.guard(images, "B, T, 1, H, W, C")
  sg.guard(masks, " B, T, K, H, W, 1")
  if sg.C == 1:  # deal with grayscale
    images = tf.tile(images, [1, 1, 1, 1, 1, 3])
    recons = tf.tile(recons, [1, 1, 1, 1, 1, 3])
  sg.S = min(nr_seqs, sg.B)
  with tf.name_scope("diagnostic_image"):
    # convert masks to rgb
    masks_trans = tf.transpose(masks[:nr_seqs], [0, 1, 5, 3, 4, 2])
    recolored_masks = color_transform(masks_trans)
    # Pad everything
    no_pad, pad = (0, 0), (border_width, border_width)
    paddings = tf.constant([no_pad, no_pad, no_pad, pad, pad, no_pad])
    pad_images = tf.pad(images[:nr_seqs], paddings, constant_values=0.5)
    pad_recons = tf.pad(recons[:nr_seqs], paddings, constant_values=0.5)
    pad_masks = tf.pad(recolored_masks, paddings, constant_values=0.5)
    # concatenate all parts along width
    triples = tf.concat([pad_images, pad_recons, pad_masks], axis=3)
    triples = sg.guard(triples[:, :, 0], "S, T, 3*Hp, Wp, 3")
    # concatenate iterations along width and sequences along height
    final = tf.reshape(
        tf.transpose(triples, [0, 2, 1, 3, 4]), sg["1, S*3*Hp, Wp*T, 3"]
    )
    if clip:
      final = tf.clip_by_value(final, 0.0, 1.0)
    return final


def get_overview_image(image, output_dist, mask_components=False):
  recons = output_dist.mean()[:, 0]
  image = image[:, 0]
  if hasattr(output_dist, "mixture_distribution") and hasattr(
      output_dist, "components_distribution"
  ):
    mask = output_dist.mixture_distribution.probs[:, 0]
    components = output_dist.components_distribution.mean()[:, 0]
    return construct_diagnostic_image(
        image, recons, mask, components, mask_components=mask_components
    )
  else:
    return construct_reconstr_image(image, recons)


class OnlineMeanVarEstimator(snt.AbstractModule):
  """Online estimator for mean and variance using Welford's algorithm."""

  def __init__(self, axis=None, ddof=0.0, name="online_mean_var"):
    super().__init__(name=name)
    self._axis = axis
    self._ddof = ddof

  def _build(self, x, weights=None):
    if weights is None:
      weights = tf.ones_like(x)
    if weights.get_shape().as_list() != x.get_shape().as_list():
      weights = tf.broadcast_to(weights, x.get_shape().as_list())

    sum_weights = tf.reduce_sum(weights, axis=self._axis)
    shape = sum_weights.get_shape().as_list()

    total = tf.get_variable(
        "total",
        shape=shape,
        dtype=weights.dtype,
        initializer=tf.zeros_initializer(),
        trainable=False,
    )
    mean = tf.get_variable(
        "mean",
        shape=shape,
        dtype=x.dtype,
        initializer=tf.zeros_initializer(),
        trainable=False,
    )
    m2 = tf.get_variable(
        "M2",
        shape=shape,
        dtype=x.dtype,
        initializer=tf.zeros_initializer(),
        trainable=False,
    )

    total_update = tf.assign_add(total, sum_weights)

    with tf.control_dependencies([total_update]):
      delta = (x - mean) * weights
      mean_update = tf.assign_add(
          mean, tf.reduce_sum(delta, axis=self._axis) / total
      )

    with tf.control_dependencies([mean_update]):
      delta2 = x - mean
      m2_update = tf.assign_add(
          m2, tf.reduce_sum(delta * delta2, axis=self._axis)
      )

    with tf.control_dependencies([m2_update]):
      return tf.identity(mean), m2 / (total - self._ddof), tf.identity(total)


def print_shapes(name, value, indent=""):
  if isinstance(value, dict):
    print("{}{}:".format(indent, name))
    for k, v in sorted(value.items(),
                       key=lambda x: (isinstance(x[1], dict), x[0])):
      print_shapes(k, v, indent + "  ")
  elif isinstance(value, list):
    print(
        "{}{}[{}]: {} @ {}".format(
            indent, name, len(value), value[0].shape, value[0].dtype
        )
    )
  elif isinstance(value, np.ndarray):
    print("{}{}: {} @ {}".format(indent, name, value.shape, value.dtype))
  elif isinstance(value, tf.Tensor):
    print(
        "{}{}: {} @ {}".format(
            indent, name, value.get_shape().as_list(), value.dtype
        )
    )
  elif np.isscalar(value):
    print("{}{}: {}".format(indent, name, value))
  else:
    print("{}{}.type: {}".format(indent, name, type(value)))


def _pad_images(images, image_border_value=0.5, border_width=2):
  """Pad images to create gray borders.

    Args:
      images: Tensor of shape [B, H], [B, H, W], or [B, H, W, C].
      image_border_value: Scalar value of greyscale borderfor images.
      border_width: Int. Border width in pixels.

    Raises:
      ValueError: if the image provided is not {2,3,4} dimensional.

    Returns:
      Tensor of same shape as images, except H and W being H + border_width and
          W + border_width.
    """
  image_rank = len(images.get_shape())
  border_paddings = (border_width, border_width)
  if image_rank == 2:  # [B, H]
    paddings = [(0, 0), border_paddings]
  elif image_rank == 3:  # [B, H, W]
    paddings = [(0, 0), border_paddings, border_paddings]
  elif image_rank == 4:  # [B, H, W, C]
    paddings = [(0, 0), border_paddings, border_paddings, (0, 0)]
  else:
    raise ValueError("expected image to be 2D, 3D or 4D, got %d" % image_rank)
  paddings = tf.constant(paddings)
  return tf.pad(images, paddings, "CONSTANT",
                constant_values=image_border_value)


def images_to_grid(
    images,
    grid_height=None,
    grid_width=4,
    max_grid_height=4,
    max_grid_width=4,
    image_border_value=0.5,
):
  """Combine images and arrange them in a grid.

    Args:
      images: Tensor of shape [B, H], [B, H, W], or [B, H, W, C].
      grid_height: Height of the grid of images to output, or None. Either
          `grid_width` or `grid_height` must be set to an integer value.
          If None, `grid_height` is set to ceil(B/`grid_width`), and capped at
          `max_grid_height` when provided.
      grid_width: Width of the grid of images to output, or None. Either
          `grid_width` or `grid_height` must be set to an integer value.
          If None, `grid_width` is set to ceil(B/`grid_height`), and capped at
          `max_grid_width` when provided.
      max_grid_height: Maximum allowable height of the grid of images to
          output or None. Only used when `grid_height` is None.
      max_grid_width: Maximum allowable width of the grid of images to output,
          or None. Only used when `grid_width` is None.
      image_border_value: None or scalar value of greyscale borderfor images.
          If None, then no border is rendered.

    Raises:
      ValueError: if neither of grid_width or grid_height are set to a positive
          integer.

    Returns:
      images: Tensor of shape [height*H, width*W, C].
        C will be set to 1 if the input was provided with no channels.
        Contains all input images in a grid.
    """

  # If only one dimension is set, infer how big the other one should be.
  if grid_height is None:
    if not isinstance(grid_width, int) or grid_width <= 0:
      raise ValueError(
          "if `grid_height` is None, `grid_width` must be " "a positive integer"
      )
    grid_height = int(math.ceil(images.get_shape()[0].value / grid_width))
    if max_grid_height is not None:
      grid_height = min(max_grid_height, grid_height)
  if grid_width is None:
    if not isinstance(grid_height, int) or grid_height <= 0:
      raise ValueError(
          "if `grid_width` is None, `grid_height` must be " "a positive integer"
      )
    grid_width = int(math.ceil(images.get_shape()[0].value / grid_height))
    if max_grid_width is not None:
      grid_width = min(max_grid_width, grid_width)

  images = images[: grid_height * grid_width, ...]

  # Pad with extra blank frames if grid_height x grid_width is less than the
  # number of frames provided.
  pre_images_shape = images.get_shape().as_list()
  if pre_images_shape[0] < grid_height * grid_width:
    pre_images_shape[0] = grid_height * grid_width - pre_images_shape[0]
    if image_border_value is not None:
      dummy_frames = image_border_value * tf.ones(
          shape=pre_images_shape, dtype=images.dtype
      )
    else:
      dummy_frames = tf.zeros(shape=pre_images_shape, dtype=images.dtype)
    images = tf.concat([images, dummy_frames], axis=0)

  if image_border_value:
    images = _pad_images(images, image_border_value=image_border_value)
  images_shape = images.get_shape().as_list()
  images = tf.reshape(images, [grid_height, grid_width] + images_shape[1:])
  if len(images_shape) == 2:
    images = tf.expand_dims(images, -1)
  if len(images_shape) <= 3:
    images = tf.expand_dims(images, -1)
  image_height, image_width, channels = images.get_shape().as_list()[2:]
  images = tf.transpose(images, perm=[0, 2, 1, 3, 4])
  images = tf.reshape(
      images, [grid_height * image_height, grid_width * image_width, channels]
  )
  return images


def flatten_all_but_last(tensor, n_dims=1):
  shape = tensor.shape.as_list()
  batch_dims = shape[:-n_dims]
  flat_tensor = tf.reshape(tensor, [np.prod(batch_dims)] + shape[-n_dims:])

  def unflatten(other_tensor):
    other_shape = other_tensor.shape.as_list()
    return tf.reshape(other_tensor, batch_dims + other_shape[1:])

  return flat_tensor, unflatten


def ensure_3d(tensor):
  if tensor.shape.ndims == 2:
    return tensor[..., None]

  assert tensor.shape.ndims == 3
  return tensor


built_element_cache = {
    "none": None,
    "global_step": tf.train.get_or_create_global_step(),
}


def build(plan, identifier):
  logging.debug("building %s", identifier)

  if identifier in built_element_cache:
    logging.debug("%s is already built, returning", identifier)
    return built_element_cache[identifier]
  elif not isinstance(plan, dict):
    return plan
  elif "constructor" in plan:
    ctor = _resolve_constructor(plan)
    kwargs = {
        k: build(v, identifier=k) for k, v in plan.items() if k != "constructor"
    }
    with tf.variable_scope(identifier):
      built_element_cache[identifier] = ctor(**kwargs)
      return built_element_cache[identifier]
  else:
    return {k: build(v, identifier=k) for k, v in plan.items()}


def _resolve_constructor(plan_subsection):
  assert "constructor" in plan_subsection, plan_subsection
  if isinstance(plan_subsection["constructor"], str):
    module, _, ctor = plan_subsection["constructor"].rpartition(".")
    mod = importlib.import_module(module)
    return getattr(mod, ctor)
  else:
    return plan_subsection["constructor"]
