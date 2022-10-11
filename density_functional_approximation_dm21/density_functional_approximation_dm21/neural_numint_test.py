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
"""Tests for neural_numint."""

import os


from absl.testing import parameterized
import attr
from pyscf import dft
from pyscf import gto
from pyscf import lib
import tensorflow.compat.v1 as tf

from density_functional_approximation_dm21 import neural_numint


class NeuralNumintTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    lib.param.TMPDIR = None
    lib.num_threads(1)

  # Golden values were obtained using the version of PySCF (including integral
  # generation) reported in the DM21 paper.
  @parameterized.parameters(
      {
          'functional': neural_numint.Functional.DM21,
          'expected_energy': -126.898521
      },
      {
          'functional': neural_numint.Functional.DM21m,
          'expected_energy': -126.907332
      },
      {
          'functional': neural_numint.Functional.DM21mc,
          'expected_energy': -126.922127
      },
      {
          'functional': neural_numint.Functional.DM21mu,
          'expected_energy': -126.898178
      },
  )
  def test_rks(self, functional, expected_energy):
    ni = neural_numint.NeuralNumInt(functional)

    mol = gto.Mole()
    mol.atom = [['Ne', 0., 0., 0.]]
    mol.basis = 'sto-3g'
    mol.build()

    mf = dft.RKS(mol)
    mf.small_rho_cutoff = 1.e-20
    mf._numint = ni
    mf.run()
    self.assertAlmostEqual(mf.e_tot, expected_energy, delta=2.e-4)

  @parameterized.parameters(
      {
          'functional': neural_numint.Functional.DM21,
          'expected_energy': -37.34184876
      },
      {
          'functional': neural_numint.Functional.DM21m,
          'expected_energy': -37.3377766
      },
      {
          'functional': neural_numint.Functional.DM21mc,
          'expected_energy': -37.33489173
      },
      {
          'functional': neural_numint.Functional.DM21mu,
          'expected_energy': -37.34015315
      },
  )
  def test_uks(self, functional, expected_energy):
    ni = neural_numint.NeuralNumInt(functional)

    mol = gto.Mole()
    mol.atom = [['C', 0., 0., 0.]]
    mol.spin = 2
    mol.basis = 'sto-3g'
    mol.build()

    mf = dft.UKS(mol)
    mf.small_rho_cutoff = 1.e-20
    mf._numint = ni
    mf.run()
    self.assertAlmostEqual(mf.e_tot, expected_energy, delta=2.e-4)

  def test_exported_model(self):

    mol = gto.Mole()
    mol.atom = [['C', 0., 0., 0.]]
    mol.spin = 2
    mol.basis = 'sto-3g'
    mol.build()

    ni = neural_numint.NeuralNumInt(neural_numint.Functional.DM21)
    mf = dft.UKS(mol)
    mf.small_rho_cutoff = 1.e-20
    mf._numint = ni
    mf.run()

    dms = mf.make_rdm1()
    ao = ni.eval_ao(mol, mf.grids.coords, deriv=2)
    rho_a = ni.eval_rho(mol, ao, dms[0], xctype='MGGA')
    rho_b = ni.eval_rho(mol, ao, dms[1], xctype='MGGA')
    inputs, _ = ni.construct_functional_inputs(
        mol=mol,
        dms=dms,
        spin=1,
        coords=mf.grids.coords,
        weights=mf.grids.weights,
        rho=(rho_a, rho_b),
        ao=ao[0])

    feed_dict = dict(
        zip(
            attr.asdict(ni._placeholders).values(),
            attr.asdict(inputs).values(),
        ))
    with ni._graph.as_default():
      outputs = ni._session.run(
          {
              'vxc': ni._vxc,
              'vrho': ni._vrho,
              'vsigma': ni._vsigma,
              'vtau': ni._vtau,
              'vhf': ni._vhf
          },
          feed_dict=feed_dict)

    export_path = os.path.join(self.get_temp_dir(), 'export')
    ni.export_functional_and_derivatives(export_path)
    model = tf.saved_model.load_v2(export_path)
    tensor_inputs = {
        k: tf.constant(v, dtype=tf.float32)
        for k, v in attr.asdict(inputs).items()
    }
    exported_output_tensors = model.signatures['default'](**tensor_inputs)
    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      exported_outputs = session.run(exported_output_tensors)
    self.assertAllClose(outputs, exported_outputs, atol=5.e-5, rtol=1.e-5)


if __name__ == '__main__':
  tf.test.main()
