# Copyright 2020 Deepmind Technologies Limited.
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

"""Mesh data utilities."""
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # pylint: disable=unused-import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
import numpy as np
import six
from six.moves import range
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def random_shift(vertices, shift_factor=0.25):
  """Apply random shift to vertices."""
  max_shift_pos = tf.cast(255 - tf.reduce_max(vertices, axis=0), tf.float32)
  max_shift_pos = tf.maximum(max_shift_pos, 1e-9)

  max_shift_neg = tf.cast(tf.reduce_min(vertices, axis=0), tf.float32)
  max_shift_neg = tf.maximum(max_shift_neg, 1e-9)

  shift = tfd.TruncatedNormal(
      tf.zeros([1, 3]), shift_factor*255, -max_shift_neg,
      max_shift_pos).sample()
  shift = tf.cast(shift, tf.int32)
  vertices += shift
  return vertices


def make_vertex_model_dataset(ds, apply_random_shift=False):
  """Prepare dataset for vertex model training."""
  def _vertex_model_map_fn(example):
    vertices = example['vertices']

    # Randomly shift vertices
    if apply_random_shift:
      vertices = random_shift(vertices)

    # Re-order vertex coordinates as (z, y, x).
    vertices_permuted = tf.stack(
        [vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)

    # Flatten quantized vertices, reindex starting from 1, and pad with a
    # zero stopping token.
    vertices_flat = tf.reshape(vertices_permuted, [-1])
    example['vertices_flat'] = tf.pad(vertices_flat + 1, [[0, 1]])

    # Create mask to indicate valid tokens after padding and batching.
    example['vertices_flat_mask'] = tf.ones_like(
        example['vertices_flat'], dtype=tf.float32)
    return example
  return ds.map(_vertex_model_map_fn)


def make_face_model_dataset(
    ds, apply_random_shift=False, shuffle_vertices=True, quantization_bits=8):
  """Prepare dataset for face model training."""
  def _face_model_map_fn(example):
    vertices = example['vertices']

    # Randomly shift vertices
    if apply_random_shift:
      vertices = random_shift(vertices)
    example['num_vertices'] = tf.shape(vertices)[0]

    # Optionally shuffle vertices and re-order faces to match
    if shuffle_vertices:
      permutation = tf.random_shuffle(tf.range(example['num_vertices']))
      vertices = tf.gather(vertices, permutation)
      face_permutation = tf.concat(
          [tf.constant([0, 1], dtype=tf.int32), tf.argsort(permutation) + 2],
          axis=0)
      example['faces'] = tf.cast(
          tf.gather(face_permutation, example['faces']), tf.int64)

    def _dequantize_verts(verts, n_bits):
      min_range = -0.5
      max_range = 0.5
      range_quantize = 2**n_bits - 1
      verts = tf.cast(verts, tf.float32)
      verts = verts * (max_range - min_range) / range_quantize + min_range
      return verts

    # Vertices are quantized. So convert to floats for input to face model
    example['vertices'] = _dequantize_verts(vertices, quantization_bits)
    example['vertices_mask'] = tf.ones_like(
        example['vertices'][..., 0], dtype=tf.float32)
    example['faces_mask'] = tf.ones_like(example['faces'], dtype=tf.float32)
    return example
  return ds.map(_face_model_map_fn)


def read_obj_file(obj_file):
  """Read vertices and faces from already opened file."""
  vertex_list = []
  flat_vertices_list = []
  flat_vertices_indices = {}
  flat_triangles = []

  for line in obj_file:
    tokens = line.split()
    if not tokens:
      continue
    line_type = tokens[0]
    # We skip lines not starting with v or f.
    if line_type == 'v':
      vertex_list.append([float(x) for x in tokens[1:]])
    elif line_type == 'f':
      triangle = []
      for i in range(len(tokens) - 1):
        vertex_name = tokens[i + 1]
        if vertex_name in flat_vertices_indices:
          triangle.append(flat_vertices_indices[vertex_name])
          continue
        flat_vertex = []
        for index in six.ensure_str(vertex_name).split('/'):
          if not index:
            continue
          # obj triangle indices are 1 indexed, so subtract 1 here.
          flat_vertex += vertex_list[int(index) - 1]
        flat_vertex_index = len(flat_vertices_list)
        flat_vertices_list.append(flat_vertex)
        flat_vertices_indices[vertex_name] = flat_vertex_index
        triangle.append(flat_vertex_index)
      flat_triangles.append(triangle)

  return np.array(flat_vertices_list, dtype=np.float32), flat_triangles


def read_obj(obj_path):
  """Open .obj file from the path provided and read vertices and faces."""

  with open(obj_path) as obj_file:
    return read_obj_file(obj_file)


def write_obj(vertices, faces, file_path, transpose=True, scale=1.):
  """Write vertices and faces to obj."""
  if transpose:
    vertices = vertices[:, [1, 2, 0]]
  vertices *= scale
  if faces is not None:
    if min(min(faces)) == 0:
      f_add = 1
    else:
      f_add = 0
  with open(file_path, 'w') as f:
    for v in vertices:
      f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
    for face in faces:
      line = 'f'
      for i in face:
        line += ' {}'.format(i + f_add)
      line += '\n'
      f.write(line)


def quantize_verts(verts, n_bits=8):
  """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  verts_quantize = (verts - min_range) * range_quantize / (
      max_range - min_range)
  return verts_quantize.astype('int32')


def dequantize_verts(verts, n_bits=8, add_noise=False):
  """Convert quantized vertices to floats."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  verts = verts.astype('float32')
  verts = verts * (max_range - min_range) / range_quantize + min_range
  if add_noise:
    verts += np.random.uniform(size=verts.shape) * (1 / range_quantize)
  return verts


def face_to_cycles(face):
  """Find cycles in face."""
  g = nx.Graph()
  for v in range(len(face) - 1):
    g.add_edge(face[v], face[v + 1])
  g.add_edge(face[-1], face[0])
  return list(nx.cycle_basis(g))


def flatten_faces(faces):
  """Converts from list of faces to flat face array with stopping indices."""
  if not faces:
    return np.array([0])
  else:
    l = [f + [-1] for f in faces[:-1]]
    l += [faces[-1] + [-2]]
    return np.array([item for sublist in l for item in sublist]) + 2  # pylint: disable=g-complex-comprehension


def unflatten_faces(flat_faces):
  """Converts from flat face sequence to a list of separate faces."""
  def group(seq):
    g = []
    for el in seq:
      if el == 0 or el == -1:
        yield g
        g = []
      else:
        g.append(el - 1)
    yield g
  outputs = list(group(flat_faces - 1))[:-1]
  # Remove empty faces
  return [o for o in outputs if len(o) > 2]


def center_vertices(vertices):
  """Translate the vertices so that bounding box is centered at zero."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  vert_center = 0.5 * (vert_min + vert_max)
  return vertices - vert_center


def normalize_vertices_scale(vertices):
  """Scale the vertices so that the long diagonal of the bounding box is one."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  scale = np.sqrt(np.sum(extents**2))
  return vertices / scale


def quantize_process_mesh(vertices, faces, tris=None, quantization_bits=8):
  """Quantize vertices, remove resulting duplicates and reindex faces."""
  vertices = quantize_verts(vertices, quantization_bits)
  vertices, inv = np.unique(vertices, axis=0, return_inverse=True)

  # Sort vertices by z then y then x.
  sort_inds = np.lexsort(vertices.T)
  vertices = vertices[sort_inds]

  # Re-index faces and tris to re-ordered vertices.
  faces = [np.argsort(sort_inds)[inv[f]] for f in faces]
  if tris is not None:
    tris = np.array([np.argsort(sort_inds)[inv[t]] for t in tris])

  # Merging duplicate vertices and re-indexing the faces causes some faces to
  # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
  # sub-faces.
  sub_faces = []
  for f in faces:
    cliques = face_to_cycles(f)
    for c in cliques:
      c_length = len(c)
      # Only append faces with more than two verts.
      if c_length > 2:
        d = np.argmin(c)
        # Cyclically permute faces just that first index is the smallest.
        sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
  faces = sub_faces
  if tris is not None:
    tris = np.array([v for v in tris if len(set(v)) == len(v)])

  # Sort faces by lowest vertex indices. If two faces have the same lowest
  # index then sort by next lowest and so on.
  faces.sort(key=lambda f: tuple(sorted(f)))
  if tris is not None:
    tris = tris.tolist()
    tris.sort(key=lambda f: tuple(sorted(f)))
    tris = np.array(tris)

  # After removing degenerate faces some vertices are now unreferenced.
  # Remove these.
  num_verts = vertices.shape[0]
  vert_connected = np.equal(
      np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1)
  vertices = vertices[vert_connected]

  # Re-index faces and tris to re-ordered vertices.
  vert_indices = (
      np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
  faces = [vert_indices[f].tolist() for f in faces]
  if tris is not None:
    tris = np.array([vert_indices[t].tolist() for t in tris])

  return vertices, faces, tris


def process_mesh(vertices, faces, quantization_bits=8):
  """Process mesh vertices and faces."""

  # Transpose so that z-axis is vertical.
  vertices = vertices[:, [2, 0, 1]]

  # Translate the vertices so that bounding box is centered at zero.
  vertices = center_vertices(vertices)

  # Scale the vertices so that the long diagonal of the bounding box is equal
  # to one.
  vertices = normalize_vertices_scale(vertices)

  # Quantize and sort vertices, remove resulting duplicates, sort and reindex
  # faces.
  vertices, faces, _ = quantize_process_mesh(
      vertices, faces, quantization_bits=quantization_bits)

  # Flatten faces and add 'new face' = 1 and 'stop' = 0 tokens.
  faces = flatten_faces(faces)

  # Discard degenerate meshes without faces.
  return {
      'vertices': vertices,
      'faces': faces,
  }


def load_process_mesh(mesh_obj_path, quantization_bits=8):
  """Load obj file and process."""
  # Load mesh
  vertices, faces = read_obj(mesh_obj_path)
  return process_mesh(vertices, faces, quantization_bits)


def plot_meshes(mesh_list,
                ax_lims=0.3,
                fig_size=4,
                el=30,
                rot_start=120,
                vert_size=10,
                vert_alpha=0.75,
                n_cols=4):
  """Plots mesh data using matplotlib."""

  n_plot = len(mesh_list)
  n_cols = np.minimum(n_plot, n_cols)
  n_rows = np.ceil(n_plot / n_cols).astype('int')
  fig = plt.figure(figsize=(fig_size * n_cols, fig_size * n_rows))
  for p_inc, mesh in enumerate(mesh_list):

    for key in [
        'vertices', 'faces', 'vertices_conditional', 'pointcloud', 'class_name'
    ]:
      if key not in list(mesh.keys()):
        mesh[key] = None

    ax = fig.add_subplot(n_rows, n_cols, p_inc + 1, projection='3d')

    if mesh['faces'] is not None:
      if mesh['vertices_conditional'] is not None:
        face_verts = np.concatenate(
            [mesh['vertices_conditional'], mesh['vertices']], axis=0)
      else:
        face_verts = mesh['vertices']
      collection = []
      for f in mesh['faces']:
        collection.append(face_verts[f])
      plt_mesh = Poly3DCollection(collection)
      plt_mesh.set_edgecolor((0., 0., 0., 0.3))
      plt_mesh.set_facecolor((1, 0, 0, 0.2))
      ax.add_collection3d(plt_mesh)

    if mesh['vertices'] is not None:
      ax.scatter3D(
          mesh['vertices'][:, 0],
          mesh['vertices'][:, 1],
          mesh['vertices'][:, 2],
          lw=0.,
          s=vert_size,
          c='g',
          alpha=vert_alpha)

    if mesh['vertices_conditional'] is not None:
      ax.scatter3D(
          mesh['vertices_conditional'][:, 0],
          mesh['vertices_conditional'][:, 1],
          mesh['vertices_conditional'][:, 2],
          lw=0.,
          s=vert_size,
          c='b',
          alpha=vert_alpha)

    if mesh['pointcloud'] is not None:
      ax.scatter3D(
          mesh['pointcloud'][:, 0],
          mesh['pointcloud'][:, 1],
          mesh['pointcloud'][:, 2],
          lw=0.,
          s=2.5 * vert_size,
          c='b',
          alpha=1.)

    ax.set_xlim(-ax_lims, ax_lims)
    ax.set_ylim(-ax_lims, ax_lims)
    ax.set_zlim(-ax_lims, ax_lims)

    ax.view_init(el, rot_start)

    display_string = ''
    if mesh['faces'] is not None:
      display_string += 'Num. faces: {}\n'.format(len(collection))
    if mesh['vertices'] is not None:
      num_verts = mesh['vertices'].shape[0]
      if mesh['vertices_conditional'] is not None:
        num_verts += mesh['vertices_conditional'].shape[0]
      display_string += 'Num. verts: {}\n'.format(num_verts)
    if mesh['class_name'] is not None:
      display_string += 'Synset: {}'.format(mesh['class_name'])
    if mesh['pointcloud'] is not None:
      display_string += 'Num. pointcloud: {}\n'.format(
          mesh['pointcloud'].shape[0])
    ax.text2D(0.05, 0.8, display_string, transform=ax.transAxes)
  plt.subplots_adjust(
      left=0., right=1., bottom=0., top=1., wspace=0.025, hspace=0.025)
  plt.show()
