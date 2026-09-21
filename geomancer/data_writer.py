# Copyright 2023 DeepMind Technologies Limited.
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

"""Data writer for Stanford Bunny experiments and other objects."""

# pylint: disable=unused-import
import copy
import io
import os
import time

from absl import app
from absl import flags
from absl import logging

from dm_control import mujoco

import numpy as np


_SHARD = flags.DEFINE_integer('shard', 0, 'Shard index')
_SIZE = flags.DEFINE_integer('size', 1000,
                             'Number of images to save to a shard')
_OBJECT = flags.DEFINE_string('object', 'dragon', 'Which object to render')
_PATH = flags.DEFINE_string('path', '', 'Path to folder with .stl files')


render_height = 1024
render_width = 1024

height = 256
width = 256


def get_normal(x):
  """Get vectors normal to a unit vector."""
  _, _, v = np.linalg.svd(x[None, :])
  return v[:, 1:]


def render(quat, light, mesh='bunny', meshdir='data'):
  """Script to render an image."""
  scale, pos = None, None
  if mesh == 'bunny':
    scale = 0.03
    pos = -1.0
  elif mesh == 'dragon':
    scale = 0.06
    pos = -0.3

  simple_world_mjcf_template = """
  <mujoco>
   <visual>
     <headlight active="0"/>
     <global offwidth="%s" offheight="%s"/>
   </visual>
   <compiler meshdir="%s"/>
   <asset>
     <mesh name="%s" file="%s.stl" scale="%g %g %g"/>
   </asset>
    <worldbody>
      <camera name="main" pos="0 0 5" xyaxes="1 0 0 0 1 0"/>
      <body name="obj" quat="{} {} {} {}">
        <geom name="%s" type="mesh" mesh="%s" pos="0 0 %g"/>
      </body>
      <light pos="{} {} {}" directional="true" dir="{} {} {}"/>
      <light pos="{} {} {}" directional="true" dir="{} {} {}"/>
    </worldbody>
  </mujoco>
  """ % (render_width, render_height, meshdir, mesh, mesh,
         scale, scale, scale, mesh, mesh, pos)

  light /= np.linalg.norm(light)
  quat /= np.linalg.norm(quat)

  simple_world_mjcf = simple_world_mjcf_template.format(
      *(np.concatenate((quat,
                        5*light, -5*light,
                        -5*light, 5*light)).tolist()))
  physics = mujoco.Physics.from_xml_string(simple_world_mjcf)
  data = physics.render(camera_id='main',
                        height=render_height,
                        width=render_width).astype(np.float32)
  data = data.reshape((width, int(render_width/width),
                       height, int(render_height/height), 3))
  return np.mean(np.mean(data, axis=1), axis=2)


def get_tangent(quat, light, mesh='bunny', meshdir='data',
                eps=0.03, use_light=True, use_quat=True):
  """Render image along with its tangent vectors by finite differences."""
  assert use_light or use_quat
  n = 0
  light_tangent = None
  quat_tangent = None
  if use_light:
    light_tangent = get_normal(light)
    n += 2
  if use_quat:
    quat_tangent = get_normal(quat)
    n += 3

  # a triple-wide pixel buffer
  try:
    data = render(quat, light, mesh=mesh, meshdir=meshdir)
    image_tangent = np.zeros((n, height, width, 3), dtype=np.float32)

    if use_quat:
      for i in range(3):
        perturbed = render(quat + eps * quat_tangent[:, i],
                           light, mesh=mesh, meshdir=meshdir)
        image_tangent[i] = (perturbed - data).astype(np.float32) / eps

    if use_light:
      j = 3 if use_quat else 0
      for i in range(2):
        perturbed = render(quat, light + eps * light_tangent[:, i],
                           mesh=mesh, meshdir=meshdir)
        image_tangent[i+j] = (perturbed - data) / eps

    image_tangent -= np.mean(image_tangent, axis=0)[None, ...]
    latent_tangent = np.block(
        [[quat_tangent, np.zeros((4, 2), dtype=np.float32)],
         [np.zeros((3, 3), dtype=np.float32), light_tangent]])

    return (np.mean(data, axis=-1),
            np.mean(image_tangent, axis=-1),
            latent_tangent)
  except:  # pylint: disable=bare-except
    logging.info('Failed with latents (quat: %s, light: %s)', quat, light)


def main(_):
  images = np.zeros((_SIZE.value, height, width), dtype=np.float32)
  latents = np.zeros((_SIZE.value, 7), dtype=np.float32)

  image_tangents = np.zeros((_SIZE.value, 5, height, width), dtype=np.float32)
  latent_tangents = np.zeros((_SIZE.value, 7, 5), dtype=np.float32)

  for i in range(_SIZE.value):
    light = np.random.randn(3)
    light /= np.linalg.norm(light)

    quat = np.random.randn(4)  # rotation represented as quaternion
    quat /= np.linalg.norm(quat)

    latents[i] = np.concatenate((light, quat))
    images[i], image_tangents[i], latent_tangents[i] = (
        get_tangent(quat, light, mesh=_OBJECT.value, meshdir=_PATH.value))
    logging.info('Rendered image %d of %d', i, _SIZE.value)

  os.makedirs(os.path.join(_PATH.value, _OBJECT.value), exist_ok=True)
  with open(os.path.join(
      _PATH.value, _OBJECT.value, 'shard_%03d.npz' % _SHARD.value), 'wb') as f:
    io_buffer = io.BytesIO()
    np.savez(io_buffer, images, latents, image_tangents, latent_tangents)
    f.write(io_buffer.getvalue())


if __name__ == '__main__':
  app.run(main)
