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
"""Plots a sphere dynamics trajectory rollout."""

import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('rollout_path', None, 'Path to rollout pickle file')
flags.DEFINE_bool('save_gif', False, 'Save animation as GIF')
flags.DEFINE_string('output_path', 'sphere_animation.gif', 'Output path for GIF')


def main(unused_argv):
  if not FLAGS.rollout_path:
    raise ValueError('Must provide rollout_path')
    
  with open(FLAGS.rollout_path, 'rb') as fp:
    rollout_data = pickle.load(fp)

  fig = plt.figure(figsize=(12, 8))
  
  # Create subplots for comparison
  ax1 = fig.add_subplot(121, projection='3d')
  ax2 = fig.add_subplot(122, projection='3d')
  
  skip = 5  # Show every 5th frame for smoother animation
  num_steps = rollout_data[0]['gt_pos'].shape[0]
  num_frames = len(rollout_data) * num_steps // skip

  # Compute bounds for consistent scaling
  bounds = []
  for trajectory in rollout_data:
    bb_min_gt = trajectory['gt_pos'].min(axis=(0, 1))
    bb_max_gt = trajectory['gt_pos'].max(axis=(0, 1))
    bb_min_pred = trajectory['pred_pos'].min(axis=(0, 1))
    bb_max_pred = trajectory['pred_pos'].max(axis=(0, 1))
    
    bb_min = np.minimum(bb_min_gt, bb_min_pred)
    bb_max = np.maximum(bb_max_gt, bb_max_pred)
    bounds.append((bb_min, bb_max))

  def animate(num):
    step = (num * skip) % num_steps
    traj = (num * skip) // num_steps
    
    # Clear both axes
    ax1.cla()
    ax2.cla()
    
    # Set consistent bounds
    bound = bounds[traj]
    for ax in [ax1, ax2]:
      ax.set_xlim([bound[0][0], bound[1][0]])
      ax.set_ylim([bound[0][1], bound[1][1]])
      ax.set_zlim([bound[0][2], bound[1][2]])
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
    
    # Get positions and faces
    gt_pos = rollout_data[traj]['gt_pos'][step]
    pred_pos = rollout_data[traj]['pred_pos'][step]
    faces = rollout_data[traj]['faces'][step]
    
    # Plot ground truth (left)
    ax1.plot_trisurf(gt_pos[:, 0], gt_pos[:, 1], faces, gt_pos[:, 2], 
                     shade=True, color='blue', alpha=0.7)
    ax1.set_title(f'Ground Truth\nTrajectory {traj} Step {step}')
    
    # Plot prediction (right)
    ax2.plot_trisurf(pred_pos[:, 0], pred_pos[:, 1], faces, pred_pos[:, 2], 
                     shade=True, color='red', alpha=0.7)
    ax2.set_title(f'Prediction\nTrajectory {traj} Step {step}')
    
    # Calculate and display error
    mse = np.mean((gt_pos - pred_pos) ** 2)
    fig.suptitle(f'Sphere Dynamics - MSE: {mse:.6f}', fontsize=14)
    
    return fig,

  anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=200, blit=False)
  
  if FLAGS.save_gif:
    print(f'Saving animation to {FLAGS.output_path}...')
    anim.save(FLAGS.output_path, writer='pillow', fps=5)
    print('Animation saved!')
  
  plt.tight_layout()
  plt.show(block=True)


if __name__ == '__main__':
  app.run(main)
