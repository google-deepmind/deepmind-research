# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Plots a cloth trajectory rollout."""

import pickle

import numpy as np
from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_string('rollout_path', None, 'Path to rollout pickle file')


def _equal_aspect_bounds(pos_array):
  """Compute cubic bounding box for a positions array with 10% padding."""
  bb_min = pos_array.min(axis=(0, 1))
  bb_max = pos_array.max(axis=(0, 1))
  center = (bb_min + bb_max) / 2.0
  half = (bb_max - bb_min).max() / 2.0 * 1.1
  return center - half, center + half


def main(unused_argv):
  with open(FLAGS.rollout_path, 'rb') as fp:
    rollout_data = pickle.load(fp)

  fig = plt.figure(figsize=(14, 7))
  ax_gt = fig.add_subplot(121, projection='3d')
  ax_pred = fig.add_subplot(122, projection='3d')
  skip = 10
  num_steps = rollout_data[0]['gt_pos'].shape[0]
  num_frames = len(rollout_data) * num_steps // skip

  # Use GT bounds for both panels: pred diverges at long horizons and would
  # otherwise dominate the bounding box, making early frames invisible.
  # With shared GT bounds the prediction simply drifts off-screen when it
  # diverges, which is much more informative.
  gt_bounds = []
  for trajectory in rollout_data:
    gt_bounds.append(_equal_aspect_bounds(trajectory['gt_pos']))

  def animate(num):
    step = (num * skip) % num_steps
    traj = (num * skip) // num_steps
    faces = rollout_data[traj]['faces'][step]
    gt_pos = rollout_data[traj]['gt_pos'][step]
    pred_pos = rollout_data[traj]['pred_pos'][step]

    ax_gt.cla()
    lo, hi = gt_bounds[traj]
    ax_gt.set_xlim([lo[0], hi[0]])
    ax_gt.set_ylim([lo[1], hi[1]])
    ax_gt.set_zlim([lo[2], hi[2]])
    ax_gt.plot_trisurf(gt_pos[:, 0], gt_pos[:, 1], faces, gt_pos[:, 2],
                       shade=True, color='steelblue')
    ax_gt.set_title('Ground Truth  step %d' % step)
    ax_gt.set_xlabel('X'); ax_gt.set_ylabel('Y'); ax_gt.set_zlabel('Z')

    ax_pred.cla()
    lo, hi = gt_bounds[traj]
    ax_pred.set_xlim([lo[0], hi[0]])
    ax_pred.set_ylim([lo[1], hi[1]])
    ax_pred.set_zlim([lo[2], hi[2]])
    ax_pred.plot_trisurf(pred_pos[:, 0], pred_pos[:, 1], faces, pred_pos[:, 2],
                         shade=True, color='darkorange')
    ax_pred.set_title('Predicted  step %d' % step)
    ax_pred.set_xlabel('X'); ax_pred.set_ylabel('Y'); ax_pred.set_zlabel('Z')

    return fig,

  _ = animation.FuncAnimation(fig, animate, frames=num_frames, interval=250)
  fig.canvas.manager.set_window_title('MeshGraphNets Cloth Rollout')
  plt.tight_layout()
  plt.show(block=True)


if __name__ == '__main__':
  app.run(main)
