# Copyright 2021 DeepMind Technologies Limited.
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

"""Tests for compute_hfx_density."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from pyscf import dft
from pyscf import gto
from pyscf import lib
from pyscf import scf
import scipy
from density_functional_approximation_dm21 import compute_hfx_density


class ComputeHfxDensityTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    lib.param.TMPDIR = None
    lib.num_threads(1)

  @parameterized.named_parameters(
      {'testcase_name': 'local_hf', 'omega': 0.},
      {'testcase_name': 'range_separated_local_hf_0.5', 'omega': 0.5},
      {'testcase_name': 'range_separated_local_hf_1.0', 'omega': 1.0},
      {'testcase_name': 'range_separated_local_hf_2.0', 'omega': 2.0},
  )
  def test_closed_shell(self, omega):
    mol = gto.M(atom='He 0. 0. 0.', basis='3-21g')
    solver = dft.RKS(mol)
    solver.grids.level = 2
    solver.grids.build()
    solver.kernel()
    dm = solver.make_rdm1()
    with mol.with_range_coulomb(omega=omega):
      target_j, target_k = scf.hf.get_jk(mol, dm)
    target_hf = -0.25 * np.einsum('ij,ji', dm, target_k)
    target_coulomb = np.einsum('ij,ji', dm, target_j)

    coords = solver.grids.coords
    weights = solver.grids.weights

    results = compute_hfx_density.get_hf_density(
        mol, dm, coords, omega=omega, weights=weights)
    coulomb = np.einsum('ij,ji', dm, results.coulomb)
    hf = -0.25 * np.einsum('ij,ji', dm, results.exchange)
    predicted_hf = np.sum((results.exx[0] + results.exx[1]) * weights)

    with self.subTest('test_hf_density'):
      self.assertAlmostEqual(target_hf, predicted_hf)

    with self.subTest('test_get_jk'):
      np.testing.assert_allclose(results.coulomb, target_j)
      np.testing.assert_allclose(results.exchange, target_k)
      self.assertAlmostEqual(coulomb, target_coulomb)
      self.assertAlmostEqual(hf, target_hf)

  @parameterized.named_parameters(
      {'testcase_name': 'local_hf', 'omega': 0.},
      {'testcase_name': 'range_separated_local_hf_0.5', 'omega': 0.5},
      {'testcase_name': 'range_separated_local_hf_1.0', 'omega': 1.0},
      {'testcase_name': 'range_separated_local_hf_2.0', 'omega': 2.0},
  )
  def test_hf_density_on_open_shell(self, omega):
    mol = gto.M(atom='He 0. 0. 0.', basis='3-21g', charge=1, spin=1)
    solver = dft.UKS(mol)
    solver.grids.level = 2
    solver.grids.build()
    solver.kernel()
    dm = solver.make_rdm1()
    with mol.with_range_coulomb(omega=omega):
      target_j, target_k = scf.hf.get_jk(mol, dm)
    target_hf = -0.5 * (
        np.einsum('ij,ji', dm[0], target_k[0]) +
        np.einsum('ij,ji', dm[1], target_k[1]))
    target_coulomb = np.einsum('ij,ji', dm[0], target_j[0]) + np.einsum(
        'ij,ji', dm[1], target_j[1])

    coords = solver.grids.coords
    weights = solver.grids.weights

    results = compute_hfx_density.get_hf_density(
        mol, dm, coords, omega=omega, weights=weights)

    predicted_hf = np.sum((results.exx[0] + results.exx[1]) * weights)
    coulomb = (
        np.einsum('ij,ji', dm[0], results.coulomb[0]) +
        np.einsum('ij,ji', dm[1], results.coulomb[1]))
    hf = -0.5 * (
        np.einsum('ij,ji', dm[0], results.exchange[0]) +
        np.einsum('ij,ji', dm[1], results.exchange[1]))

    with self.subTest('test_hf_density'):
      self.assertAlmostEqual(target_hf, predicted_hf, places=3)

    with self.subTest('test_get_jk'):
      np.testing.assert_allclose(results.coulomb[0], target_j[0])
      np.testing.assert_allclose(results.coulomb[1], target_j[1])
      np.testing.assert_allclose(results.exchange[0], target_k[0])
      np.testing.assert_allclose(results.exchange[1], target_k[1])
      self.assertAlmostEqual(coulomb, target_coulomb)
      self.assertAlmostEqual(hf, target_hf)


def _nu_test_systems():
  systems = [
      {
          'atom': 'N 0 0 0; N 0 0 2.4',
          'charge': 0,
          'spin': 0,
          'basis': 'cc-pVDZ',
          'num_grids': -1
      },
      {
          'atom': 'N 0 0 0; N 0 0 2.4',
          'charge': 0,
          'spin': 0,
          'basis': 'cc-pVDZ',
          'num_grids': 1
      },
      {
          'atom': 'N 0 0 0; N 0 0 2.4',
          'charge': 0,
          'spin': 0,
          'basis': 'cc-pVDZ',
          'num_grids': 2
      },
      {
          'atom': 'N 0 0 0; N 0 0 2.4',
          'charge': 0,
          'spin': 0,
          'basis': 'cc-pVDZ',
          'num_grids': 10
      },
      {
          'atom': 'N 0 0 0; N 0 0 2.4',
          'charge': 0,
          'spin': 0,
          'basis': 'cc-pVDZ',
          'num_grids': 32
      },
      {
          'atom': 'N 0 0 0; N 0 0 2.4',
          'charge': 0,
          'spin': 0,
          'basis': 'cc-pVDZ',
          'num_grids': 33
      },
      {
          'atom': 'Li 0 0 0',
          'charge': 0,
          'spin': 1,
          'basis': 'cc-pVTZ',
          'num_grids': -1
      },
      {
          'atom': 'H 0 0 0',
          'charge': 0,
          'spin': 1,
          'basis': 'cc-pVQZ',
          'num_grids': -1
      },
  ]
  system_names = ['N2', 'N2_1', 'N2_2', 'N2_10', 'N2_32', 'N2_33', 'Li', 'H']
  for name, system in zip(system_names, systems):
    yield {'testcase_name': f'{name}_hermitian', 'hermi': 0, **system}
    yield {'testcase_name': f'{name}_non_hermitian', 'hermi': 1, **system}


class NuTest(parameterized.TestCase):

  def setUp(self):
    super(NuTest, self).setUp()
    lib.param.TMPDIR = None
    lib.num_threads(1)

  @parameterized.named_parameters(_nu_test_systems())
  def test_nu_integrals(self, atom, charge, spin, basis, num_grids, hermi):
    mol = gto.M(atom=atom, charge=charge, spin=spin, basis=basis)
    mf = dft.UKS(mol)
    mf.grids.build()
    if num_grids == -1:
      test_coords = mf.grids.coords
    else:
      test_coords = mf.grids.coords[0:num_grids]
    nu_slow = compute_hfx_density._evaluate_nu_slow(
        mol, test_coords, omega=0.0, hermi=hermi)
    nu_fast = compute_hfx_density._evaluate_nu(
        mol, test_coords, omega=0.0, hermi=hermi)
    np.testing.assert_allclose(nu_slow, nu_fast, atol=1E-13)

  def test_range_separated_nu(self):
    mol = gto.M(atom='He 0 0 0', basis='cc-pVDZ')
    r0 = np.array([[0.1, 0.2, 1.]])
    omega = 1.
    result = np.squeeze(compute_hfx_density._evaluate_nu(mol, r0, omega=omega))

    solver = dft.RKS(mol)
    solver.grids.level = 2
    solver.grids.build()
    coords = solver.grids.coords
    weights = solver.grids.weights
    ao_value = dft.numint.eval_ao(mol, coords, deriv=0)
    dist = np.linalg.norm(coords - r0, axis=1)
    erf = scipy.special.erf(omega * dist) / dist
    expected_result = np.squeeze(
        np.einsum('g,ga,gb->ab', weights * erf, ao_value, ao_value))

    np.testing.assert_allclose(result, expected_result)


if __name__ == '__main__':
  absltest.main()
