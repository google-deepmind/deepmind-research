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

"""An interface to DM21 family of exchange-correlation functionals for PySCF."""

import enum
import os
from typing import Generator, Optional, Sequence, Tuple, Union

import attr
import numpy as np
from pyscf import dft
from pyscf import gto
from pyscf.dft import numint
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

from density_functional_approximation_dm21 import compute_hfx_density

tf.disable_v2_behavior()

# TODO(b/196260242): avoid depending upon private function
_dot_ao_ao = numint._dot_ao_ao  # pylint: disable=protected-access


@enum.unique
class Functional(enum.Enum):
  """Enum for exchange-correlation functionals in the DM21 family.

  Attributes:
    DM21: trained on molecules dataset, and fractional charge, and fractional
      spin constraints.
    DM21m: trained on molecules dataset.
    DM21mc: trained on molecules dataset, and fractional charge constraints.
    DM21mu: trained on molecules dataset, and electron gas constraints.
  """
  # Break pylint's preferred naming pattern to match the functional names used
  # in the paper.
  # pylint: disable=invalid-name
  DM21 = enum.auto()
  DM21m = enum.auto()
  DM21mc = enum.auto()
  DM21mu = enum.auto()
  # pylint: enable=invalid-name


# We use attr.s instead of here instead of dataclasses.dataclass as
# dataclasses.asdict returns a deepcopy of the attributes. This is wasteful in
# memory if they are large and breaks (as in the case of tf.Tensors) if they are
# not serializable. attr.asdict does not perform this copy and so works with
# both np.ndarrays and tf.Tensors.
@attr.s(auto_attribs=True)
class FunctionalInputs:
  r""""Inputs required for DM21 functionals.

  Depending upon the context, this is either a set of numpy arrays (feature
  construction) or TF tensors (constructing placeholders/running functionals).

  Attributes:
    rho_a: Density information for the alpha electrons.
      PySCF for meta-GGAs supplies a single array for the total density
      (restricted calculations) and a pair of arrays, one for each spin channel
      (unrestricted calculations).
      Each array/tensor is of shape (6, N) and contains the density and density
      derivatives, where:
       rho(0, :) - density at each grid point
       rho(1, :) - norm of the derivative of the density at each grid point
                   along x
       rho(2, :) - norm of the derivative of the density at each grid point
                   along y
       rho(3, :) - norm of the derivative of the density at each grid point
                   along z
       rho(4, :) - \nabla^2 \rho [not used]
       rho(5, :) - tau (1/2 (\nabla \rho)^2) at each grid point.
      See pyscf.dft.numint.eval_rho for more details.
      We require separate inputs for both alpha- and beta-spin densities, even
      in restricted calculations (where rho_a = rho_b = rho/2, where rho is the
      total density).
    rho_b: as for rho_a for the beta electrons.
    hfx_a: local Hartree-Fock energy density at each grid point for the alpha-
      spin density for each value of omega.  Shape [N, len(omega_values)].
      See compute_hfx_density for more details.
    hfx_b: as for hfx_a for the beta-spin density.
    grid_coords: grid coordinates at which to evaluate the density. Shape
      (N, 3), where N is the number of grid points. Note that this is currently
      unused by the functional, but is still a required input.
    grid_weights: weight of each grid point. Shape (N).
  """
  rho_a: Union[tf.Tensor, np.ndarray]
  rho_b: Union[tf.Tensor, np.ndarray]
  hfx_a: Union[tf.Tensor, np.ndarray]
  hfx_b: Union[tf.Tensor, np.ndarray]
  grid_coords: Union[tf.Tensor, np.ndarray]
  grid_weights: Union[tf.Tensor, np.ndarray]


@attr.s(auto_attribs=True)
class _GridState:
  """Internal state required for the numerical grid.

  Attributes:
    coords: coordinates of the grid. Shape (N, 3), where N is the number of grid
      points.
    weight: weight associated with each grid point. Shape (N).
    mask: mask indicating whether a shell is zero at a grid point. Shape
      (N, nbas) where nbas is the number of shells in the basis set. See
      pyscf.dft.gen_grids.make_mask.
    ao: atomic orbitals evaluated on the grid. Shape (N, nao), where nao is the
      number of atomic orbitals, or shape (:, N, nao), where the 0-th element
      contains the ao values, the next three elements contain the first
      derivatives, and so on.
  """
  coords: np.ndarray
  weight: np.ndarray
  mask: np.ndarray
  ao: np.ndarray


@attr.s(auto_attribs=True)
class _SystemState:
  """Internal state required for system of interest.

  Attributes:
    mol: PySCF molecule
    dms: density matrix or matrices (unrestricted calculations only).
      Restricted calculations: shape (nao, nao), where nao is the number of
      atomic orbitals.
      Unrestricted calculations: shape (2, nao, nao) or a sequence (length 2) of
      arrays of shape (nao, nao), and dms[0] and dms[1] are the density matrices
      of the alpha and beta electrons respectively.
  """
  mol: gto.Mole
  dms: Union[np.ndarray, Sequence[np.ndarray]]


def _get_number_of_density_matrices(dms):
  """Returns the number of density matrices in dms."""
  # See pyscf.numint.NumInt._gen_rho_evaluator
  if isinstance(dms, np.ndarray) and dms.ndim == 2:
    return 1
  return len(dms)


class NeuralNumInt(numint.NumInt):
  """A wrapper around pyscf.dft.numint.NumInt for the DM21 functionals.

  In order to supply the local Hartree-Fock features required for the DM21
  functionals, we lightly wrap the NumInt class. The actual evaluation of the
  exchange-correlation functional is performed in NeuralNumInt.eval_xc.

  Usage:
    mf = dft.RKS(...)  # dft.ROKS and dft.UKS are also supported.
    # Specify the functional by monkey-patching mf._numint rather than using
    # mf._xc or mf._define_xc_.
    mf._numint = NeuralNumInt(Functional.DM21)
    mf.kernel()
  """

  def __init__(self,
               functional: Functional,
               *,
               checkpoint_path: Optional[str] = None):
    """Constructs a NeuralNumInt object.

    Args:
      functional: member of Functional enum giving the name of the
        functional.
      checkpoint_path: Optional path to specify the directory containing the
        checkpoints of the DM21 family of functionals. If not specified, attempt
        to find the checkpoints using a path relative to the source code.
    """

    self._functional_name = functional.name
    if checkpoint_path:
      self._model_path = os.path.join(checkpoint_path, self._functional_name)
    else:
      self._model_path = os.path.join(
          os.path.dirname(__file__), 'checkpoints', self._functional_name)

    # All DM21 functionals use local Hartree-Fock features with a non-range
    # separated 1/r kernel and a range-seperated kernel with \omega = 0.4.
    # Note an omega of 0.0 is interpreted by PySCF and libcint to indicate no
    # range-separation.
    self._omega_values = [0.0, 0.4]
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._build_graph()
      self._session = tf.Session()
      self._session.run(tf.global_variables_initializer())

    self._grid_state = None
    self._system_state = None
    self._vmat_hf = None
    super().__init__()

  def _build_graph(self, batch_dim: Optional[int] = None):
    """Builds the TensorFlow graph for evaluating the functional.

    Args:
      batch_dim: the batch dimension of the grid to use in the model. Default:
        None (determine at runtime). This should only be set if building a model
        in order to export and ahead-of-time compile it into a standalone
        library.
    """

    self._functional = hub.Module(spec=self._model_path)

    grid_coords = tf.placeholder(
        tf.float32, shape=[batch_dim, 3], name='grid_coords')
    grid_weights = tf.placeholder(
        tf.float32, shape=[batch_dim], name='grid_weights')

    # Density information.
    rho_a = tf.placeholder(tf.float32, shape=[6, batch_dim], name='rho_a')
    rho_b = tf.placeholder(tf.float32, shape=[6, batch_dim], name='rho_b')

    # Split into corresponding terms.
    rho_only_a, grad_a_x, grad_a_y, grad_a_z, _, tau_a = tf.unstack(
        rho_a, axis=0)
    rho_only_b, grad_b_x, grad_b_y, grad_b_z, _, tau_b = tf.unstack(
        rho_b, axis=0)

    # Evaluate |\del \rho|^2 for each spin density and for the total density.
    norm_grad_a = (grad_a_x**2 + grad_a_y**2 + grad_a_z**2)
    norm_grad_b = (grad_b_x**2 + grad_b_y**2 + grad_b_z**2)
    grad_x = grad_a_x + grad_b_x
    grad_y = grad_a_y + grad_b_y
    grad_z = grad_a_z + grad_b_z
    norm_grad = (grad_x**2 + grad_y**2 + grad_z**2)

    # The local Hartree-Fock energy densities at each grid point for the alpha-
    # and beta-spin densities for each value of omega.
    # Note an omega of 0 indicates no screening of the Coulomb potential.
    hfxa = tf.placeholder(
        tf.float32, shape=[batch_dim, len(self._omega_values)], name='hfxa')
    hfxb = tf.placeholder(
        tf.float32, shape=[batch_dim, len(self._omega_values)], name='hfxb')

    # Make all features 2D arrays on input for ease of handling inside the
    # functional.
    features = {
        'grid_coords': grid_coords,
        'grid_weights': tf.expand_dims(grid_weights, 1),
        'rho_a': tf.expand_dims(rho_only_a, 1),
        'rho_b': tf.expand_dims(rho_only_b, 1),
        'tau_a': tf.expand_dims(tau_a, 1),
        'tau_b': tf.expand_dims(tau_b, 1),
        'norm_grad_rho_a': tf.expand_dims(norm_grad_a, 1),
        'norm_grad_rho_b': tf.expand_dims(norm_grad_b, 1),
        'norm_grad_rho': tf.expand_dims(norm_grad, 1),
        'hfxa': hfxa,
        'hfxb': hfxb,
    }
    tensor_dict = {f'tensor_dict${k}': v for k, v in features.items()}

    predictions = self._functional(tensor_dict, as_dict=True)
    local_xc = predictions['grid_contribution']
    weighted_local_xc = local_xc * grid_weights
    unweighted_xc = tf.reduce_sum(local_xc, axis=0)
    xc = tf.reduce_sum(weighted_local_xc, axis=0)

    # The potential is the local exchange correlation divided by the
    # total density. Add a small constant to deal with zero density.
    self._vxc = local_xc / (rho_only_a + rho_only_b + 1E-12)

    # The derivatives of the exchange-correlation (XC) energy with respect to
    # input features.  PySCF weights the (standard) derivatives by the grid
    # weights, so we need to compute this with respect to the unweighted sum
    # over grid points.
    self._vrho = tf.gradients(
        unweighted_xc, [features['rho_a'], features['rho_b']],
        name='GRAD_RHO',
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    self._vsigma = tf.gradients(
        unweighted_xc, [
            features['norm_grad_rho_a'], features['norm_grad_rho_b'],
            features['norm_grad_rho']
        ],
        name='GRAD_SIGMA',
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    self._vtau = tf.gradients(
        unweighted_xc, [features['tau_a'], features['tau_b']],
        name='GRAD_TAU',
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # Standard meta-GGAs do not have a dependency on local HF, so we need to
    # compute the contribution to the Fock matrix ourselves. Just use the
    # weighted XC energy to avoid having to weight this later.
    self._vhf = tf.gradients(
        xc, [features['hfxa'], features['hfxb']],
        name='GRAD_HFX',
        unconnected_gradients=tf.UnconnectedGradients.ZERO)

    self._placeholders = FunctionalInputs(
        rho_a=rho_a,
        rho_b=rho_b,
        hfx_a=hfxa,
        hfx_b=hfxb,
        grid_coords=grid_coords,
        grid_weights=grid_weights)

    outputs = {
        'vxc': self._vxc,
        'vrho': tf.stack(self._vrho),
        'vsigma': tf.stack(self._vsigma),
        'vtau': tf.stack(self._vtau),
        'vhf': tf.stack(self._vhf),
    }
    # Create the signature for TF-Hub, including both the energy and functional
    # derivatives.
    # This is a no-op if _build_graph is called outside of
    # hub.create_module_spec.
    hub.add_signature(
        inputs=attr.asdict(self._placeholders), outputs=outputs)

  def export_functional_and_derivatives(
      self,
      export_path: str,
      batch_dim: Optional[int] = None,
  ):
    """Exports the TensorFlow graph containing the functional and derivatives.

    The hub modules supplied contain the TensorFlow operations for the
    evaluation of the exchange-correlation energy. Evaluation of the functional
    derivatives, required for a self-consistent calculation, are added in
    _build_graph. The module created by export_functional_and_derivatives
    contains the evaluation of the functional and the functional derivatives.
    This is much simpler to use from languages other than Python, e.g. using the
    C or C++ TensorFlow API, or using tfcompile to create a standalone C++
    library.

    Args:
      export_path: path to write the Hub model to. The exported model can be
        loaded using either TF-Hub or SavedModel APIs.
      batch_dim: the batch dimension of the grid to use in the model. Default:
        None (determine at runtime). This should only be set if the exported
        model is to be ahead-of-time compiled into a standalone library.
    """
    with tf.Graph().as_default():
      spec = hub.create_module_spec(
          self._build_graph, tags_and_args=[(set(), {'batch_dim': batch_dim})])
      functional_and_derivatives = hub.Module(spec=spec)
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        functional_and_derivatives.export(export_path, session)

  # DM21* functionals include the hybrid term directly, so set the
  # range-separated and hybrid parameters expected by PySCF to 0 so PySCF
  # doesn't also add these contributions in separately.
  def rsh_coeff(self, *args):
    """Returns the range separated parameters, omega, alpha, beta."""
    return [0.0, 0.0, 0.0]

  def hybrid_coeff(self, *args, **kwargs):
    """Returns the fraction of Hartree-Fock exchange to include."""
    return 0.0

  def _xc_type(self, *args, **kwargs):
    return 'MGGA'

  def nr_rks(self,
             mol: gto.Mole,
             grids: dft.Grids,
             xc_code: str,
             dms: Union[np.ndarray, Sequence[np.ndarray]],
             relativity: int = 0,
             hermi: int = 0,
             max_memory: float = 20000,
             verbose=None) -> Tuple[float, float, np.ndarray]:
    """Calculates RKS XC functional and potential matrix on a given grid.

    Args:
      mol: PySCF molecule.
      grids: grid on which to evaluate the functional.
      xc_code: XC code. Unused. NeuralNumInt hard codes the XC functional
        based upon the functional argument given to the constructor.
      dms: the density matrix or sequence of density matrices. Multiple density
        matrices are not currently supported. Shape (nao, nao), where nao is the
        number of atomic orbitals.
      relativity: Unused. (pyscf.numint.NumInt.nr_rks does not currently use
        this argument.)
      hermi: 0 if the density matrix is Hermitian, 1 if the density matrix is
        non-Hermitian.
      max_memory: the maximum cache to use, in MB.
      verbose: verbosity level. Unused. (PySCF currently does not handle the
        verbosity level passed in here.)

    Returns:
      nelec, excsum, vmat, where
        nelec is the number of electrons obtained by numerical integration of
        the density matrix.
        excsum is the functional's XC energy.
        vmat is the functional's XC potential matrix, shape (nao, nao).

    Raises:
      NotImplementedError: if multiple density matrices are supplied.
    """
    # Wrap nr_rks so we can store internal variables required to evaluate the
    # contribution to the XC potential from local Hartree-Fock features.
    # See pyscf.dft.numint.nr_rks for more details.
    ndms = _get_number_of_density_matrices(dms)
    if ndms > 1:
      raise NotImplementedError(
          'NeuralNumInt does not support multiple density matrices. '
          'Only ground state DFT calculations are currently implemented.')
    nao = mol.nao_nr()
    self._vmat_hf = np.zeros((nao, nao))
    self._system_state = _SystemState(mol=mol, dms=dms)
    nelec, excsum, vmat = super().nr_rks(
        mol=mol,
        grids=grids,
        xc_code=xc_code,
        dms=dms,
        relativity=relativity,
        hermi=hermi,
        max_memory=max_memory,
        verbose=verbose)
    vmat += self._vmat_hf + self._vmat_hf.T

    # Clear internal state to prevent accidental re-use.
    self._system_state = None
    self._grid_state = None
    return nelec, excsum, vmat

  def nr_uks(self,
             mol: gto.Mole,
             grids: dft.Grids,
             xc_code: str,
             dms: Union[Sequence[np.ndarray], Sequence[Sequence[np.ndarray]]],
             relativity: int = 0,
             hermi: int = 0,
             max_memory: float = 20000,
             verbose=None) -> Tuple[np.ndarray, float, np.ndarray]:
    """Calculates UKS XC functional and potential matrix on a given grid.

    Args:
      mol: PySCF molecule.
      grids: grid on which to evaluate the functional.
      xc_code: XC code. Unused. NeuralNumInt hard codes the XC functional
        based upon the functional argument given to the constructor.
      dms: the density matrix or sequence of density matrices for each spin
        channel. Multiple density matrices for each spin channel are not
        currently supported. Each density matrix is shape (nao, nao), where nao
        is the number of atomic orbitals.
      relativity: Unused. (pyscf.dft.numint.NumInt.nr_rks does not currently use
        this argument.)
      hermi: 0 if the density matrix is Hermitian, 1 if the density matrix is
        non-Hermitian.
      max_memory: the maximum cache to use, in MB.
      verbose: verbosity level. Unused. (PySCF currently does not handle the
        verbosity level passed in here.)

    Returns:
      nelec, excsum, vmat, where
        nelec is the number of alpha, beta electrons obtained by numerical
        integration of the density matrix as an array of size 2.
        excsum is the functional's XC energy.
        vmat is the functional's XC potential matrix, shape (2, nao, nao), where
        vmat[0] and vmat[1] are the potential matrices for the alpha and beta
        spin channels respectively.

    Raises:
      NotImplementedError: if multiple density matrices for each spin channel
      are supplied.
    """
    # Wrap nr_uks so we can store internal variables required to evaluate the
    # contribution to the XC potential from local Hartree-Fock features.
    # See pyscf.dft.numint.nr_uks for more details.
    if isinstance(dms, np.ndarray) and dms.ndim == 2:  # RHF DM
      ndms = _get_number_of_density_matrices(dms)
    else:
      ndms = _get_number_of_density_matrices(dms[0])
    if ndms > 1:
      raise NotImplementedError(
          'NeuralNumInt does not support multiple density matrices. '
          'Only ground state DFT calculations are currently implemented.')

    nao = mol.nao_nr()
    self._vmat_hf = np.zeros((2, nao, nao))
    self._system_state = _SystemState(mol=mol, dms=dms)
    nelec, excsum, vmat = super().nr_uks(
        mol=mol,
        grids=grids,
        xc_code=xc_code,
        dms=dms,
        relativity=relativity,
        hermi=hermi,
        max_memory=max_memory,
        verbose=verbose)
    vmat[0] += self._vmat_hf[0] + self._vmat_hf[0].T
    vmat[1] += self._vmat_hf[1] + self._vmat_hf[1].T

    # Clear internal state to prevent accidental re-use.
    self._system_state = None
    self._grid_state = None
    self._vmat_hf = None
    return nelec, excsum, vmat

  def block_loop(
      self,
      mol: gto.Mole,
      grids: dft.Grids,
      nao: Optional[int] = None,
      deriv: int = 0,
      max_memory: float = 2000,
      non0tab: Optional[np.ndarray] = None,
      blksize: Optional[int] = None,
      buf: Optional[np.ndarray] = None
  ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None,
                 None]:
    """Loops over the grid by blocks. See pyscf.dft.numint.NumInt.block_loop.

    Args:
      mol: PySCF molecule.
      grids: grid on which to evaluate the functional.
      nao: number of basis functions. If None, obtained from mol.
      deriv: unused. The first functional derivatives are always computed.
      max_memory: the maximum cache to use for the information on the grid, in
        MB. Determines the size of each block if blksize is None.
      non0tab: mask determining if a shell in the basis set is zero at a grid
        point. Shape (N, nbas), where N is the number of grid points and nbas
        the number of shells in the basis set. Obtained from grids if not
        supplied.
      blksize: size of each block. Calculated from max_memory if None.
      buf: buffer to use for storing ao. If None, a new array for ao is created
        for each block.

    Yields:
      ao, mask, weight, coords: information on a block of the grid containing N'
      points, where
        ao: atomic orbitals evaluated on the grid. Shape (N', nao), where nao is
        the number of atomic orbitals.
        mask: mask indicating whether a shell in the basis set is zero at a grid
        point. Shape (N', nbas).
        weight: weight associated with each grid point. Shape (N').
        coords: coordinates of the grid. Shape (N', 3).
    """
    # Wrap block_loop so we can store internal variables required to evaluate
    # the contribution to the XC potential from local Hartree-Fock features.
    for ao, mask, weight, coords in super().block_loop(
        mol=mol,
        grids=grids,
        nao=nao,
        deriv=deriv,
        max_memory=max_memory,
        non0tab=non0tab,
        blksize=blksize,
        buf=buf):
      # Cache the curent block so we can access it in eval_xc.
      self._grid_state = _GridState(
          ao=ao, mask=mask, weight=weight, coords=coords)
      yield ao, mask, weight, coords

  def construct_functional_inputs(
      self,
      mol: gto.Mole,
      dms: Union[np.ndarray, Sequence[np.ndarray]],
      spin: int,
      coords: np.ndarray,
      weights: np.ndarray,
      rho: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
      ao: Optional[np.ndarray] = None,
  ) -> Tuple[FunctionalInputs, Tuple[np.ndarray, np.ndarray]]:
    """Constructs the input features required for the functional.

    Args:
      mol: PySCF molecule.
      dms: density matrix of shape (nao, nao) (restricted calculations) or of
        shape (2, nao, nao) (unrestricted calculations) or tuple of density
        matrices for each spin channel, each of shape (nao, nao) (unrestricted
        calculations).
      spin: 0 for a spin-unpolarized (restricted Kohn-Sham) calculation, and
        spin-polarized (unrestricted) otherwise.
      coords: coordinates of the grid. Shape (N, 3), where N is the number of
        grid points.
      weights: weight associated with each grid point. Shape (N).
      rho: density and density derivatives at each grid point. Single array
        containing the total density for restricted calculations, tuple of
        arrays for each spin channel for unrestricted calculations. Each array
        has shape (6, N). See pyscf.dft.numint.eval_rho and comments in
        FunctionalInputs for more details.
      ao: The atomic orbitals evaluated on the grid, shape (N, nao). Computed if
        not supplied.

    Returns:
      inputs, fxx, where
        inputs: FunctionalInputs object containing the inputs (as np.ndarrays)
        for the functional.
        fxx: intermediates, shape (N, nao) for the alpha- and beta-spin
        channels, required for computing the first derivative of the local
        Hartree-Fock density with respect to the density matrices. See
        compute_hfx_density for more details.
    """
    if spin == 0:
      # RKS
      rhoa = rho / 2
      rhob = rho / 2
    else:
      # UKS
      rhoa, rhob = rho

    # Local HF features.
    exxa, exxb = [], []
    fxxa, fxxb = [], []
    for omega in sorted(self._omega_values):
      hfx_results = compute_hfx_density.get_hf_density(
          mol,
          dms,
          coords=coords,
          omega=omega,
          deriv=1,
          ao=ao)
      exxa.append(hfx_results.exx[0])
      exxb.append(hfx_results.exx[1])
      fxxa.append(hfx_results.fxx[0])
      fxxb.append(hfx_results.fxx[1])
    exxa = np.stack(exxa, axis=-1)
    fxxa = np.stack(fxxa, axis=-1)
    if spin == 0:
      exx = (exxa, exxa)
      fxx = (fxxa, fxxa)
    else:
      exxb = np.stack(exxb, axis=-1)
      fxxb = np.stack(fxxb, axis=-1)
      exx = (exxa, exxb)
      fxx = (fxxa, fxxb)

    return FunctionalInputs(
        rho_a=rhoa,
        rho_b=rhob,
        hfx_a=exx[0],
        hfx_b=exx[1],
        grid_coords=coords,
        grid_weights=weights), fxx

  def eval_xc(
      self,
      xc_code: str,
      rho: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
      spin: int = 0,
      relativity: int = 0,
      deriv: int = 1,
      omega: Optional[float] = None,
      verbose=None
  ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
             None, None]:
    """Evaluates the XC energy and functional derivatives.

    See pyscf.dft.libxc.eval_xc for more details on the interface.

    Note: this also sets self._vmat_extra, which contains the contribution the
    the potential matrix from the local Hartree-Fock terms in the functional.

    Args:
      xc_code: unused.
      rho: density and density derivatives at each grid point. Single array
        containing the total density for restricted calculations, tuple of
        arrays for each spin channel for unrestricted calculations. Each array
        has shape (6, N), where N is the number of grid points. See
        pyscf.dft.numint.eval_rho and comments in FunctionalInputs for more
        details.
      spin: 0 for a spin-unpolarized (restricted Kohn-Sham) calculation, and
        spin-polarized (unrestricted) otherwise.
      relativity: Not supported.
      deriv: unused. The first functional derivatives are always computed.
      omega: RSH parameter. Not supported.
      verbose: unused.

    Returns:
      exc, vxc, fxc, kxc, where:
        exc is the exchange-correlation potential matrix evaluated at each grid
        point, shape (N).
        vxc is (vrho, vgamma, vlapl, vtau), the first-order functional
        derivatives evaluated at each grid point, each shape (N).
        fxc is set to None. (The second-order functional derivatives are not
        computed.)
        kxc is set to None. (The third-order functional derivatives are not
        computed.)
    """
    del xc_code, verbose, deriv  # unused

    if relativity != 0:
      raise NotImplementedError('Relatistic calculations are not implemented '
                                'for DM21 functionals.')
    if omega is not None:
      raise NotImplementedError('User-specifed range seperation parameters are '
                                'not implemented for DM21 functionals.')

    # Retrieve cached state.
    ao = self._grid_state.ao
    if ao.ndim == 3:
      # Just need the AO values, not the gradients.
      ao = ao[0]
    if self._grid_state.weight is None:
      weights = np.array([1.])
    else:
      weights = self._grid_state.weight
    mask = self._grid_state.mask

    inputs, (fxxa, fxxb) = self.construct_functional_inputs(
        mol=self._system_state.mol,
        dms=self._system_state.dms,
        spin=spin,
        rho=rho,
        weights=weights,
        coords=self._grid_state.coords,
        ao=ao)

    with self._graph.as_default():
      feed_dict = dict(
          zip(
              attr.asdict(self._placeholders).values(),
              attr.asdict(inputs).values(),
          ))
      tensor_list = [
          self._vxc,
          self._vrho,
          self._vsigma,
          self._vtau,
          self._vhf,
      ]
      exc, vrho, vsigma, vtau, vhf = (
          self._session.run(tensor_list, feed_dict=feed_dict))

    mol = self._system_state.mol
    shls_slice = (0, mol.nbas)
    ao_loc_nr = mol.ao_loc_nr()
    # Note: tf.gradients returns a list of gradients.
    # vrho, vsigma, vtau are derivatives of objects that had
    # tf.expand_dims(..., 1) applied. The [:, 0] indexing undoes this by
    # selecting the 0-th (and only) element from the second dimension.
    if spin == 0:
      vxc_0 = (vrho[0][:, 0] + vrho[1][:, 0]) / 2.
      # pyscf expects derivatives with respect to:
      # grad_rho . grad_rho.
      # The functional uses the first and last as inputs, but then has
      # grad_(rho_a + rho_b) . grad_(rho_a + rho_b)
      # as input. The following computes the correct total derivatives.
      vxc_1 = (vsigma[0][:, 0] / 4. + vsigma[1][:, 0] / 4. + vsigma[2][:, 0])
      vxc_3 = (vtau[0][:, 0] + vtau[1][:, 0]) / 2.
      vxc_2 = np.zeros_like(vxc_3)
      vhfs = (vhf[0] + vhf[1]) / 2.
      # Local Hartree-Fock terms
      for i in range(len(self._omega_values)):
        # Factor of 1/2 is to account for adding vmat_hf + vmat_hf.T to vmat,
        # which we do to match existing PySCF style. Unlike other terms, vmat_hf
        # is already symmetric though.
        aow = np.einsum('pi,p->pi', fxxa[:, :, i], -0.5 * vhfs[:, i])
        self._vmat_hf += _dot_ao_ao(mol, ao, aow, mask, shls_slice,
                                    ao_loc_nr)
    else:
      vxc_0 = np.stack([vrho[0][:, 0], vrho[1][:, 0]], axis=1)
      # pyscf expects derivatives with respect to:
      # grad_rho_a . grad_rho_a
      # grad_rho_a . grad_rho_b
      # grad_rho_b . grad_rho_b
      # The functional uses the first and last as inputs, but then has
      # grad_(rho_a + rho_b) . grad_(rho_a + rho_b)
      # as input. The following computes the correct total derivatives.
      vxc_1 = np.stack([
          vsigma[0][:, 0] + vsigma[2][:, 0], 2. * vsigma[2][:, 0],
          vsigma[1][:, 0] + vsigma[2][:, 0]
      ],
                       axis=1)
      vxc_3 = np.stack([vtau[0][:, 0], vtau[1][:, 0]], axis=1)
      vxc_2 = np.zeros_like(vxc_3)
      vhfs = np.stack([vhf[0], vhf[1]], axis=2)
      for i in range(len(self._omega_values)):
        # Factors of 1/2 are due to the same reason as in the spin=0 case.
        aow = np.einsum('pi,p->pi', fxxa[:, :, i], -0.5 * vhfs[:, i, 0])
        self._vmat_hf[0] += _dot_ao_ao(mol, ao, aow, mask, shls_slice,
                                       ao_loc_nr)
        aow = np.einsum('pi,p->pi', fxxb[:, :, i], -0.5 * vhfs[:, i, 1])
        self._vmat_hf[1] += _dot_ao_ao(mol, ao, aow, mask, shls_slice,
                                       ao_loc_nr)

    fxc = None  # Second derivative not implemented
    kxc = None  # Second derivative not implemented
    # PySCF C routines expect float64.
    exc = exc.astype(np.float64)
    vxc = tuple(v.astype(np.float64) for v in (vxc_0, vxc_1, vxc_2, vxc_3))
    return exc, vxc, fxc, kxc
