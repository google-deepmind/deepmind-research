# Copyright 2020 DeepMind Technologies Limited.
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

"""Run GEOMANCER on products of synthetic manifolds."""

import re

from absl import app
from absl import flags
from absl import logging

import geomancer

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import special_ortho_group
from tqdm import tqdm

SPECIFICATION = flags.DEFINE_list(
    name='specification', default=['S^2', 'S^2'], help='List of submanifolds')
NPTS = flags.DEFINE_integer(
    name='npts', default=1000, help='Number of data points')
ROTATE = flags.DEFINE_boolean(
    name='rotate', default=False, help='Apply random rotation to the data')
PLOT = flags.DEFINE_boolean(
    name='plot', default=True, help='Whether to enable plotting')


def make_so_tangent(q):
  """Given an n x n orthonormal matrix, return a basis for its tangent space."""
  n = q.shape[0]
  assert np.allclose(q.T @ q, np.eye(n), atol=1e-4, rtol=1e-4)
  a = np.zeros((n, n))
  ii = 0
  dq = np.zeros((n, n, n*(n-1)//2))
  for i in range(n):
    for j in range(i+1, n):
      a[i, j] = 1
      a[j, i] = -1
      dq[..., ii] = a @ q  # tangent vectors are skew-symmetric matrix times Q
      a[i, j] = 0
      a[j, i] = 0
      ii += 1
  # reshape and orthonormalize the result
  return np.linalg.qr(np.reshape(dq, (n**2, n*(n-1)//2)))[0]


def make_sphere_tangent(x):
  _, _, v = np.linalg.svd(x[None, :])
  return v[:, 1:]


def make_true_tangents(spec, data):
  """Return a set of orthonormal bases, one for each submanifold."""
  for i in range(spec.shape[1]):
    assert spec[0, i] == 0 or spec[1, i] == 0
  so_dim = sum(dim ** 2 for dim in spec[0])
  sphere_dim = sum(dim+1 if dim > 0 else 0 for dim in spec[1])
  assert so_dim + sphere_dim == data.shape[0]

  ii = 0
  tangents = []
  for i in range(spec.shape[1]):
    if spec[0, i] != 0:
      dim = spec[0, i]
      tangents.append(make_so_tangent(np.reshape(data[ii:ii+dim**2],
                                                 (dim, dim))))
      ii += dim ** 2
    else:
      dim = spec[1, i]
      tangents.append(make_sphere_tangent(data[ii:ii+dim+1]))
      ii += dim + 1

  tangents2 = []
  for i in range(len(tangents)):
    size1 = sum(x.shape[0] for x in tangents[:i])
    size2 = sum(x.shape[0] for x in tangents[i+1:])
    tangents2.append(np.concatenate(
        (np.zeros((size1, tangents[i].shape[1])),
         tangents[i],
         np.zeros((size2, tangents[i].shape[1]))), axis=0))
  return tangents2


def make_product_manifold(specification, npts):
  """Generate data from a product of manifolds with the given specification."""
  data = []
  tangents = []
  latent_dim = 0
  spec_array = np.zeros((2, len(specification)), dtype=np.int32)
  for i, spec in enumerate(specification):
    so_spec = re.search(r'SO\(([0-9]+)\)', spec)  # matches "SO(<numbers>)"
    sphere_spec = re.search(r'S\^([0-9]+)', spec)  # matches "S^<numbers>"

    if sphere_spec is not None:
      dim = int(sphere_spec.group(1))
      spec_array[1, i] = dim
      latent_dim += dim
      dat = np.random.randn(npts, dim+1)
      dat /= np.tile(np.sqrt(np.sum(dat**2, axis=1)[..., None]),
                     [1, dim+1])
    elif so_spec is not None:
      dim = int(so_spec.group(1))
      spec_array[0, i] = dim
      latent_dim += dim * (dim - 1) // 2
      dat = [np.ndarray.flatten(special_ortho_group.rvs(dim), order='C')
             for _ in range(npts)]
      dat = np.stack(dat)
    else:
      raise ValueError(f'Unrecognized manifold: {spec}')
    data.append(dat)
  data = np.concatenate(data, axis=1)

  for i in range(spec_array.shape[1]):
    if spec_array[0, i] != 0:
      dim = spec_array[0, i]
      tangents.append(np.zeros((npts, data.shape[1], dim * (dim - 1) // 2)))
    elif spec_array[1, i] != 0:
      dim = spec_array[1, i]
      tangents.append(np.zeros((npts, data.shape[1], dim)))

  for i in tqdm(range(npts)):
    true_tangent = make_true_tangents(spec_array, data[i])
    for j in range(len(specification)):
      tangents[j][i] = true_tangent[j]
  logging.info('Constructed data and true tangents for %s',
               ' x '.join(specification))
  return data, latent_dim, tangents


def main(_):
  # Generate data and run GEOMANCER
  data, dim, tangents = make_product_manifold(SPECIFICATION.value, NPTS.value)
  if ROTATE.value:
    rot, _ = np.linalg.qr(np.random.randn(data.shape[1], data.shape[1]))
    data_rot = data @ rot.T
    components, spectrum = geomancer.fit(data_rot, dim)
    errors = geomancer.eval_unaligned(data_rot, components, data, tangents)
  else:
    components, spectrum = geomancer.fit(data, dim)
    errors = geomancer.eval_aligned(components, tangents)

  logging.info('Error between subspaces: %.2f +/- %.2f radians',
               np.mean(errors),
               np.std(errors))

  if PLOT.value:

    # Plot spectrum
    plt.figure(figsize=(8, 6))
    plt.scatter(np.arange(len(spectrum)), spectrum, s=100)
    largest_gap = np.argmax(spectrum[1:]-spectrum[:-1]) + 1
    plt.axvline(largest_gap, linewidth=2, c='r')
    plt.xticks([])
    plt.yticks(fontsize=18)
    plt.xlabel('Index', fontsize=24)
    plt.ylabel('Eigenvalue', fontsize=24)
    plt.title('GeoManCEr Eigenvalue Spectrum', fontsize=24)

    # Plot subspace bases
    fig = plt.figure(figsize=(8, 6))
    bases = components[0]
    gs = gridspec.GridSpec(1, len(bases),
                           width_ratios=[b.shape[1] for b in bases])
    for i in range(len(bases)):
      ax = plt.subplot(gs[i])
      ax.imshow(bases[i])
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_title(r'$T_{\mathbf{x}_1}\mathcal{M}_%d$' % (i+1), fontsize=18)
    fig.canvas.set_window_title('GeoManCEr Results')

    # Plot ground truth
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, len(tangents),
                           width_ratios=[b.shape[2] for b in tangents])
    for i, spec in enumerate(SPECIFICATION.value):
      ax = plt.subplot(gs[i])
      ax.imshow(tangents[i][0])
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_title(r'$T_{\mathbf{x}_1}%s$' % spec, fontsize=18)
    fig.canvas.set_window_title('Ground Truth')

    plt.show()


if __name__ == '__main__':
  app.run(main)
