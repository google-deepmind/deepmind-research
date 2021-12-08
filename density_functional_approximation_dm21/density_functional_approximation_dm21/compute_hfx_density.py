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

r"""Computation of the Hartree-Fock exchange density.

We consider two types of potential:

  1. Coulomb potential v(r,r') = 1/|r-r'|, which results in the full HF exchange
     density and energy.
  2. Screened (long-range) Coulomb potential v(r,r') = erf(\omega|r-r'|)/|r-r'|,
     which results in the screened HF exchange density energy.

Note that PySCF and libcint treat a value of omega=0 to correspond to the
Coulomb potential. In the following, HF refers to full HF exchange if the
Coulomb potential is used and to screened HF exchange if the screened Coulomb
potential is used.

The Hartree-Fock (HF) exchange energy can be written as:

-2 HF_x = \sum_{a,b,c,d} D_{ab} D_{cd} \int dr \int dr'
   [ \chi_a(r) \chi_c(r) v(r, r') \chi_b(r') \chi_d(r') ]

where D is the density matrix, \chi_a the atomic basis functions and r, r' are
coordinates. For clarity we have dropped the spin-channel label of the density
matrix.

Defining the following intermediates:

\nu_{bd}(r) = \int dr' (\chi_b(r') v(r, r') \chi_d(r'))
E_b(r) = \sum_a D_{ab} \chi_a(r)

we get the following expression for HF:

-2 HF_x = \int dr \sum_{bd} E_b(r) E_d(r) \nu_{bd}(r)

Therefore the quantity

exx(r) = -0.5 sum_{bd} E_b(r) E_d(r) \nu_{bd}(r)

represents an energy density at location r which integrates to the HF exchange
energy.

The Fock matrix, F, is the derivative of the energy with respect to the density
matrix. If the energy depends upon the set of features {x}, then the Fock matrix
can be evaluated as \sum_x dE/dx dx/dD_{ab}. The derivatives with respect to the
features can be easily evaluated using automatic differentiation. We hence
require the derivative of exx with respect to the density matrix:

dexx(r)/dD_{ab} = -D_{cd} \chi_a(r) \chi_c(r) \nu_{bd}(r)

This is too large to store, so we instead compute the following intermediate,
and evaluate the derivative as required on the fly:

fxx_a(r) = D_{bd} \chi_a(r) \nu_{bd}(r)

Note: we compute exx and fxx for each spin channel for both restricted and
unrestricted calculations.
"""

from typing import Generator, Optional, Tuple, Union

import attr
import numpy as np
from pyscf.dft import numint
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.lib import numpy_helper


def _evaluate_nu_slow(mol: mole.Mole,
                      coords: np.ndarray,
                      omega: float,
                      hermi: int = 1) -> np.ndarray:
  """Computes nu integrals for given coordinates using a slow loop."""
  nu = []
  # Use the Gaussian nuclear model in int1e_rinv_sph to evaluate the screened
  # integrals.
  with mol.with_rinv_zeta(zeta=omega * omega):
    # This is going to be slow...
    for coord in coords:
      with mol.with_rinv_origin(coord):
        nu.append(mol.intor('int1e_rinv_sph', hermi=hermi))
  return np.asarray(nu)


def _evaluate_nu(mol: mole.Mole,
                 coords: np.ndarray,
                 omega: float,
                 hermi: int = 1) -> np.ndarray:
  """Computes nu integrals for given coordinates."""
  try:
    with mol.with_range_coulomb(omega=omega):
      # grids keyword argument supported in pyscf 2.0.0-alpha.
      nu = mol.intor('int1e_grids_sph', hermi=hermi, grids=coords)  # pytype: disable=wrong-keyword-args
  except TypeError:
    logger.info(
        mol, 'Support for int1e_grids not found (requires libcint 4.4.1 and '
        'pyscf 2.0.0a or later. Falling back to slow loop over individual grid '
        'points.')
    nu = _evaluate_nu_slow(mol, coords, omega)
  return nu


def _nu_chunk(mol: mole.Mole,
              coords: np.ndarray,
              omega: float,
              chunk_size: int = 1000
             ) -> Generator[Tuple[int, int, np.ndarray], None, None]:
  r"""Yields chunks of nu integrals over the grid.

  Args:
    mol: pyscf Mole object.
    coords: coordinates, r', at which to evaluate the nu integrals, shape (N,3).
    omega: range separation parameter. A value of 0 disables range-separation
      (i.e. uses the kernel v(r,r') = 1/|r-r'| instead of
      v(r,r') = erf(\omega |r-r'|) / |r-r'|)
    chunk_size: number of coordinates to evaluate the integrals at a time.

  Yields:
    start_index, end_index, nu_{ab}(r) where
      start_index, end_index are indices into coords,
      nu is an array of shape (end_index-start_index, nao, nao), where nao is
      the number of atomic orbitals and contains
      nu_{ab}(r) = <a(r')|v(r,r')| b(r')>, where a,b are atomic
      orbitals and r' are the grid coordinates in coords[start_index:end_index].

  Raises:
    ValueError: if omega is negative.
  """
  if omega < 0:
    raise ValueError('Range-separated parameter omega must be non-negative!')
  ncoords = len(coords)
  for chunk_index in range(0, ncoords, chunk_size):
    end_index = min(chunk_index + chunk_size, ncoords)
    coords_chunk = coords[chunk_index:end_index]
    nu_chunk = _evaluate_nu(mol, coords_chunk, omega=omega)
    yield chunk_index, end_index, nu_chunk


def _compute_exx_block(nu: np.ndarray,
                       e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  r"""Computes exx and fxx.

  Args:
    nu: batch of <i|v(r,r_k)|j> integrals, in format (k,i,j) where r_k is the
      position of the k-th grid point, i and j label atomic orbitals.
    e: density matrix in the AO basis at each grid point.

  Returns:
    exx and fxx, where
    fxx_{gb} =\sum_c nu_{gbc} e_{gc} and
    exx_{g} = -0.5 \sum_b e_{gb} fxx_{gb}.
  """
  fxx = np.einsum('gbc,gc->gb', nu, e)
  exx = -0.5 * np.einsum('gb,gb->g', e, fxx)
  return exx, fxx


def _compute_jk_block(nu: np.ndarray, fxx: np.ndarray, dm: np.ndarray,
                      ao_value: np.ndarray,
                      weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Computes J and K contributions from the given block of nu integrals."""
  batch_size = nu.shape[0]
  vj = numpy_helper.dot(nu.reshape(batch_size, -1), dm.reshape(-1, 1))
  vj = np.squeeze(vj)
  vj_ao = np.einsum('g,gb->gb', vj * weights, ao_value)
  j = numpy_helper.dot(ao_value.T, vj_ao)
  w_ao = np.einsum('g,gb->gb', weights, ao_value)
  k = numpy_helper.dot(fxx.T, w_ao)
  return j, k


@attr.s(auto_attribs=True)
class HFDensityResult:
  r"""Container for results returned by get_hf_density.

  Note that the kernel used in all integrals is defined by the omega input
  argument.

  Attributes:
    exx: exchange energy density at position r on the grid for the alpha, beta
      spin channels.  Each array is shape (N), where N is the number of grid
      points.
    fxx: intermediate for evaluating dexx/dD^{\sigma}_{ab}, where D is the
      density matrix and \sigma is the spin coordinate. See top-level docstring
      for details.  Each array is shape (N, nao), where nao is the number of
      atomic orbitals.
    coulomb: coulomb matrix (restricted calculations) or matrices (unrestricted
      calculations). Each array is shape (nao, nao).
      Restricted calculations: \sum_{} D_{cd} (ab|cd)
      Unrestricted calculations: \sum_{} D^{\sigma}_{cd} (ab|cd)
    exchange: exchange matrix (restricted calculations) or matrices
      (unrestricted calculations). Each array is shape (nao, nao).
      Restricted calculations: \sum_{} D_{cd} (ab|cd)
      Unrestricted calculations: \sum_{} D^{\sigma}_{cd} (ac|bd).
  """
  exx: Tuple[np.ndarray, np.ndarray]
  fxx: Optional[Tuple[np.ndarray, np.ndarray]] = None
  coulomb: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None
  exchange: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None


def get_hf_density(
    mol: mole.Mole,
    dm: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
    coords: np.ndarray,
    omega: float = 0.,
    deriv: int = 0,
    ao: Optional[np.ndarray] = None,
    chunk_size: int = 1000,
    weights: Optional[np.ndarray] = None,
) -> HFDensityResult:
  r"""Computes the (range-separated) HF energy density.

  Args:
    mol: PySCF molecule.
    dm: The density matrix. For restricted calculations, an array of shape
      (M, M), where M is the number of atomic orbitals. For unrestricted
      calculations, either an array of shape (2, M, M) or a tuple of arrays,
      each of shape (M, M), where dm[0] is the density matrix for the alpha
      electrons and dm[1] the density matrix for the beta electrons.
    coords: The coordinates to compute the HF density at, shape (N, 3), where N
      is the number of grid points.
    omega: The inverse width of the error function. An omega of 0. means range
      separation and a 1/|r-R| kernel is used in the nu integrals. Otherwise,
      the kernel erf(\omega|r-R|)/|r-R|) is used. Must be non-negative.
    deriv: The derivative order. Only first derivatives (deriv=1) are currently
      implemented. deriv=0 indicates no derivatives are required.
    ao: The atomic orbitals evaluated on the grid, shape (N, M). These are
      computed if not supplied.
    chunk_size: The number of coordinates to compute the HF density for at once.
      Reducing this saves memory since we don't have to keep as many Nus (nbasis
      x nbasis) in memory at once.
    weights: weight of each grid point, shape (N). If present, the Coulomb and
      exchange matrices are also computed semi-numerically, otherwise only the
      HF density and (if deriv=1) its first derivative are computed.

  Returns:
    HFDensityResult object with the HF density (exx), the derivative of the HF
    density with respect to the density (fxx) if deriv is 1, and the Coulomb and
    exchange matrices if the weights argument is provided.

  Raises:
    NotImplementedError: if a Cartesian basis set is used or if deriv is greater
    than 1.
    ValueError: if omega or deriv are negative.
  """
  if mol.cart:
    raise NotImplementedError('Local HF exchange is not implmented for basis '
                              'sets with Cartesian functions!')
  if deriv < 0:
    raise ValueError(f'`deriv` must be non-negative, got {deriv}')
  if omega < 0:
    raise ValueError(f'`omega` must be non-negative, got {omega}')
  if deriv > 1:
    raise NotImplementedError('Higher order derivatives are not implemented.')

  if isinstance(dm, tuple) or dm.ndim == 3:
    dma, dmb = dm
    restricted = False
  else:
    dma = dm / 2
    dmb = dm / 2
    restricted = True

  logger.info(mol, 'Computing contracted density matrix ...')
  if ao is None:
    ao = numint.eval_ao(mol, coords, deriv=0)
  e_a = np.dot(ao, dma)
  e_b = np.dot(ao, dmb)

  exxa = []
  exxb = []
  fxxa = []
  fxxb = []
  ja = np.zeros_like(dma)
  jb = np.zeros_like(dmb)
  ka = np.zeros_like(dma)
  kb = np.zeros_like(dmb)

  for start, end, nu in _nu_chunk(mol, coords, omega, chunk_size=chunk_size):
    logger.info(mol, 'Computing exx %s / %s ...', end, len(e_a))
    exxa_block, fxxa_block = _compute_exx_block(nu, e_a[start:end])
    exxa.extend(exxa_block)
    if not restricted:
      exxb_block, fxxb_block = _compute_exx_block(nu, e_b[start:end])
      exxb.extend(exxb_block)
    if deriv == 1:
      fxxa.extend(fxxa_block)
      if not restricted:
        fxxb.extend(fxxb_block)

    if weights is not None:
      ja_block, ka_block = _compute_jk_block(nu, fxxa_block, dma, ao[start:end],
                                             weights[start:end])
      ja += ja_block
      ka += ka_block
      if not restricted:
        jb_block, kb_block = _compute_jk_block(nu, fxxb_block, dmb,
                                               ao[start:end],
                                               weights[start:end])
        jb += jb_block
        kb += kb_block

  exxa = np.asarray(exxa)
  fxxa = np.asarray(fxxa)
  if restricted:
    exxb = exxa
    fxxb = fxxa
  else:
    exxb = np.asarray(exxb)
    fxxb = np.asarray(fxxb)

  result = HFDensityResult(exx=(exxa, exxb))
  if deriv == 1:
    result.fxx = (fxxa, fxxb)
  if weights is not None:
    if restricted:
      result.coulomb = 2 * ja
      result.exchange = 2 * ka
    else:
      result.coulomb = (ja, jb)
      result.exchange = (ka, kb)
  return result
