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
"""Plotting tools for IODINE."""
# pylint: disable=unused-import, missing-docstring, unused-variable
# pylint: disable=invalid-name, unexpected-keyword-arg
import functools
from iodine.modules.utils import get_mask_plot_colors
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np

__all__ = ("get_mask_plot_colors", "example_plot", "iterations_plot",
           "inputs_plot")


def clean_ax(ax, color=None, lw=4.0):
  ax.set_xticks([])
  ax.set_yticks([])
  if color is not None:
    for spine in ax.spines.values():
      spine.set_linewidth(lw)
      spine.set_color(color)


def optional_ax(fn):

  def _wrapped(*args, **kwargs):
    if kwargs.get("ax", None) is None:
      figsize = kwargs.pop("figsize", (4, 4))
      fig, ax = plt.subplots(figsize=figsize)
      kwargs["ax"] = ax
    return fn(*args, **kwargs)

  return _wrapped


def optional_clean_ax(fn):

  def _wrapped(*args, **kwargs):
    if kwargs.get("ax", None) is None:
      figsize = kwargs.pop("figsize", (4, 4))
      fig, ax = plt.subplots(figsize=figsize)
      kwargs["ax"] = ax
    color = kwargs.pop("color", None)
    lw = kwargs.pop("lw", 4.0)
    res = fn(*args, **kwargs)
    clean_ax(kwargs["ax"], color, lw)
    return res

  return _wrapped


@optional_clean_ax
def show_img(img, mask=None, ax=None, norm=False):
  if norm:
    vmin, vmax = np.min(img), np.max(img)
    img = (img - vmin) / (vmax - vmin)
  if mask is not None:
    img = img * mask + np.ones_like(img) * (1.0 - mask)

  return ax.imshow(img.clip(0.0, 1.0), interpolation="nearest")


@optional_clean_ax
def show_mask(m, ax):
  color_conv = get_mask_plot_colors(m.shape[0])
  color_mask = np.dot(np.transpose(m, [1, 2, 0]), color_conv)
  return ax.imshow(color_mask.clip(0.0, 1.0), interpolation="nearest")


@optional_clean_ax
def show_mat(m, ax, vmin=None, vmax=None, cmap="viridis"):
  return ax.matshow(
      m[..., 0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")


@optional_clean_ax
def show_coords(m, ax):
  vmin, vmax = np.min(m), np.max(m)
  m = (m - vmin) / (vmax - vmin)
  color_conv = get_mask_plot_colors(m.shape[-1])
  color_mask = np.dot(m, color_conv)
  return ax.imshow(color_mask, interpolation="nearest")


def example_plot(rinfo,
                 b=0,
                 t=-1,
                 mask_components=False,
                 size=2,
                 column_titles=True):
  image = rinfo["data"]["image"][b, 0]
  recons = rinfo["outputs"]["recons"][b, t, 0]
  pred_mask = rinfo["outputs"]["pred_mask"][b, t]
  components = rinfo["outputs"]["components"][b, t]

  K, H, W, C = components.shape
  colors = get_mask_plot_colors(K)

  nrows = 1
  ncols = 3 + K
  fig, axes = plt.subplots(ncols=ncols, figsize=(ncols * size, nrows * size))

  show_img(image, ax=axes[0], color="#000000")
  show_img(recons, ax=axes[1], color="#000000")
  show_mask(pred_mask[..., 0], ax=axes[2], color="#000000")
  for k in range(K):
    mask = pred_mask[k] if mask_components else None
    show_img(components[k], ax=axes[k + 3], color=colors[k], mask=mask)

  if column_titles:
    labels = ["Image", "Recons.", "Mask"
             ] + ["Component {}".format(k + 1) for k in range(K)]
    for ax, title in zip(axes, labels):
      ax.set_title(title)
  plt.subplots_adjust(hspace=0.03, wspace=0.035)
  return fig


def iterations_plot(rinfo, b=0, mask_components=False, size=2):
  image = rinfo["data"]["image"][b]
  true_mask = rinfo["data"]["true_mask"][b]
  recons = rinfo["outputs"]["recons"][b]
  pred_mask = rinfo["outputs"]["pred_mask"][b]
  pred_mask_logits = rinfo["outputs"]["pred_mask_logits"][b]
  components = rinfo["outputs"]["components"][b]

  T, K, H, W, C = components.shape
  colors = get_mask_plot_colors(K)
  nrows = T + 1
  ncols = 2 + K
  fig, axes = plt.subplots(
      nrows=nrows, ncols=ncols, figsize=(ncols * size, nrows * size))
  for t in range(T):
    show_img(recons[t, 0], ax=axes[t, 0])
    show_mask(pred_mask[t, ..., 0], ax=axes[t, 1])
    axes[t, 0].set_ylabel("iter {}".format(t))
    for k in range(K):
      mask = pred_mask[t, k] if mask_components else None
      show_img(components[t, k], ax=axes[t, k + 2], color=colors[k], mask=mask)

  axes[0, 0].set_title("Reconstruction")
  axes[0, 1].set_title("Mask")
  show_img(image[0], ax=axes[T, 0])
  show_mask(true_mask[0, ..., 0], ax=axes[T, 1])
  vmin = np.min(pred_mask_logits[T - 1])
  vmax = np.max(pred_mask_logits[T - 1])

  for k in range(K):
    axes[0, k + 2].set_title("Component {}".format(k + 1))  # , color=colors[k])
    show_mat(
        pred_mask_logits[T - 1, k], ax=axes[T, k + 2], vmin=vmin, vmax=vmax)
    axes[T, k + 2].set_xlabel(
        "Mask Logits for\nComponent {}".format(k + 1))  # , color=colors[k])
  axes[T, 0].set_xlabel("Input Image")
  axes[T, 1].set_xlabel("Ground Truth Mask")

  plt.subplots_adjust(wspace=0.05, hspace=0.05)
  return fig


def inputs_plot(rinfo, b=0, t=0, size=2):
  B, T, K, H, W, C = rinfo["outputs"]["components"].shape
  colors = get_mask_plot_colors(K)
  inputs = rinfo["inputs"]["spatial"]
  rows = [
      ("image", show_img, False),
      ("components", show_img, False),
      ("dcomponents", functools.partial(show_img, norm=True), False),
      ("mask", show_mat, True),
      ("pred_mask", show_mat, True),
      ("dmask", functools.partial(show_mat, cmap="coolwarm"), True),
      ("posterior", show_mat, True),
      ("log_prob", show_mat, True),
      ("counterfactual", show_mat, True),
      ("coordinates", show_coords, False),
  ]
  rows = [(n, f, mcb) for n, f, mcb in rows if n in inputs]
  nrows = len(rows)
  ncols = K + 1

  fig, axes = plt.subplots(
      nrows=nrows,
      ncols=ncols,
      figsize=(ncols * size - size * 0.9, nrows * size),
      gridspec_kw={"width_ratios": [1] * K + [0.1]},
  )
  for r, (name, plot_fn, make_cbar) in enumerate(rows):
    axes[r, 0].set_ylabel(name)
    if make_cbar:
      vmin = np.min(inputs[name][b, t])
      vmax = np.max(inputs[name][b, t])
      if np.abs(vmin - vmax) < 1e-6:
        vmin -= 0.1
        vmax += 0.1
      plot_fn = functools.partial(plot_fn, vmin=vmin, vmax=vmax)
      # print("range of {:<16}: [{:0.2f}, {:0.2f}]".format(name, vmin, vmax))
    for k in range(K):
      if inputs[name].shape[2] == 1:
        m = inputs[name][b, t, 0]
        color = (0.0, 0.0, 0.0)
      else:
        m = inputs[name][b, t, k]
        color = colors[k]
      mappable = plot_fn(m, ax=axes[r, k], color=color)
    if make_cbar:
      fig.colorbar(mappable, cax=axes[r, K])
    else:
      axes[r, K].set_visible(False)
  for k in range(K):
    axes[0, k].set_title("Component {}".format(k + 1))  # , color=colors[k])

  plt.subplots_adjust(hspace=0.05, wspace=0.05)
  return fig
