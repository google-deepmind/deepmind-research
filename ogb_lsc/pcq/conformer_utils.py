# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conformer utilities."""

import copy
from typing import List, Optional
from absl import logging
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import tensorflow.compat.v2 as tf


def generate_conformers(
    molecule: Chem.rdchem.Mol,
    max_num_conformers: int,
    *,
    random_seed: int = -1,
    prune_rms_thresh: float = -1.0,
    max_iter: int = -1,
    fallback_to_random: bool = False,
) -> Chem.rdchem.Mol:
  """Generates conformers for a given molecule.

  Args:
    molecule: molecular representation of the compound.
    max_num_conformers: maximum number of conformers to generate. If pruning is
      done, the returned number of conformers is not guaranteed to match
      max_num_conformers.
    random_seed: random seed to use for conformer generation.
    prune_rms_thresh: RMSD threshold which allows to prune conformers that are
      too similar.
    max_iter: Maximum number of iterations to perform when optimising MMFF force
      field. If set to <= 0, energy optimisation is not performed.
    fallback_to_random: if conformers cannot be obtained, use random coordinates
      to initialise.

  Returns:
    Copy of a `molecule` with added hydrogens. The returned molecule contains
    force field-optimised conformers. The number of conformers is guaranteed to
    be <= max_num_conformers.
  """
  mol = copy.deepcopy(molecule)
  mol = Chem.AddHs(mol)
  mol = _embed_conformers(
      mol,
      max_num_conformers,
      random_seed,
      prune_rms_thresh,
      fallback_to_random,
      use_random=False)

  if max_iter > 0:
    mol_with_conformers = _minimize_by_mmff(mol, max_iter)
    if mol_with_conformers is None:
      mol_with_conformers = _minimize_by_uff(mol, max_iter)
  else:
    mol_with_conformers = mol
  # Aligns conformations in a molecule to each other using the first
  # conformation as the reference.
  AllChem.AlignMolConformers(mol_with_conformers)

  # We remove hydrogens to keep the number of atoms consistent with the graph
  # nodes.
  mol_with_conformers = Chem.RemoveHs(mol_with_conformers)

  return mol_with_conformers


def atom_to_feature_vector(
    atom: rdkit.Chem.rdchem.Atom,
    conformer: Optional[np.ndarray] = None,
) -> List[float]:
  """Converts rdkit atom object to feature list of indices.

  Args:
    atom: rdkit atom object.
    conformer: Generated conformers. Returns -1 values if set to None.

  Returns:
    List containing positions (x, y, z) of each atom from the conformer.
  """
  if conformer:
    pos = conformer.GetAtomPosition(atom.GetIdx())
    return [pos.x, pos.y, pos.z]
  return [np.nan, np.nan, np.nan]


def compute_conformer(smile: str, max_iter: int = -1) -> np.ndarray:
  """Computes conformer.

  Args:
    smile: Smile string.
    max_iter: Maximum number of iterations to perform when optimising MMFF force
      field. If set to <= 0, energy optimisation is not performed.

  Returns:
    A tuple containing index, fingerprint and conformer.
  Raises:
    RuntimeError: If unable to convert smile string to RDKit mol.
  """
  mol = rdkit.Chem.MolFromSmiles(smile)
  if not mol:
    raise RuntimeError('Unable to convert smile to molecule: %s' % smile)
  conformer_failed = False
  try:
    mol = generate_conformers(
        mol,
        max_num_conformers=1,
        random_seed=45,
        prune_rms_thresh=0.01,
        max_iter=max_iter)
  except IOError as e:
    logging.exception('Failed to generate conformers for %s . IOError %s.',
                      smile, e)
    conformer_failed = True
  except ValueError:
    logging.error('Failed to generate conformers for %s . ValueError', smile)
    conformer_failed = True
  except:  # pylint: disable=bare-except
    logging.error('Failed to generate conformers for %s.', smile)
    conformer_failed = True

  atom_features_list = []
  conformer = None if conformer_failed else list(mol.GetConformers())[0]
  for atom in mol.GetAtoms():
    atom_features_list.append(atom_to_feature_vector(atom, conformer))
  conformer_features = np.array(atom_features_list, dtype=np.float32)
  return conformer_features


def get_random_rotation_matrix(include_mirror_symmetry: bool) -> tf.Tensor:
  """Returns a single random rotation matrix."""
  rotation_matrix = _get_random_rotation_3d()
  if include_mirror_symmetry:
    random_mirror_symmetry = _get_random_mirror_symmetry()
    rotation_matrix = tf.matmul(rotation_matrix, random_mirror_symmetry)

  return rotation_matrix


def rotate(vectors: tf.Tensor, rotation_matrix: tf.Tensor) -> tf.Tensor:
  """Batch of vectors on a single rotation matrix."""
  return tf.matmul(vectors, rotation_matrix)


def _embed_conformers(
    molecule: Chem.rdchem.Mol,
    max_num_conformers: int,
    random_seed: int,
    prune_rms_thresh: float,
    fallback_to_random: bool,
    *,
    use_random: bool = False,
) -> Chem.rdchem.Mol:
  """Embeds conformers into a copy of a molecule.

  If random coordinates allowed, tries not to use random coordinates at first,
  and uses random only if fails.

  Args:
    molecule: molecular representation of the compound.
    max_num_conformers: maximum number of conformers to generate. If pruning is
      done, the returned number of conformers is not guaranteed to match
      max_num_conformers.
    random_seed: random seed to use for conformer generation.
    prune_rms_thresh: RMSD threshold which allows to prune conformers that are
      too similar.
    fallback_to_random: if conformers cannot be obtained, use random coordinates
      to initialise.
    *:
    use_random: Use random coordinates. Shouldn't be set by any caller except
      this function itself.

  Returns:
  A copy of a molecule with embedded conformers.

  Raises:
    ValueError: if conformers cannot be obtained for a given molecule.
  """
  mol = copy.deepcopy(molecule)

  # Obtains parameters for conformer generation.
  # In particular, ETKDG is experimental-torsion basic knowledge distance
  # geometry, which allows to randomly generate an initial conformation that
  # satisfies various geometric constraints such as lower and upper bounds on
  # the distances between atoms.
  params = AllChem.ETKDGv3()

  params.randomSeed = random_seed
  params.pruneRmsThresh = prune_rms_thresh
  params.numThreads = -1
  params.useRandomCoords = use_random

  conf_ids = AllChem.EmbedMultipleConfs(mol, max_num_conformers, params)

  if not conf_ids:
    if not fallback_to_random or use_random:
      raise ValueError('Cant get conformers')
    return _embed_conformers(
        mol,
        max_num_conformers,
        random_seed,
        prune_rms_thresh,
        fallback_to_random,
        use_random=True)
  return mol


def _minimize_by_mmff(
    molecule: Chem.rdchem.Mol,
    max_iter: int,
) -> Optional[Chem.rdchem.Mol]:
  """Minimizes forcefield for conformers using MMFF algorithm.

  Args:
    molecule: a datastructure containing conformers.
    max_iter: number of maximum iterations to use when optimising force field.

  Returns:
    A copy of a `molecule` containing optimised conformers; or None if MMFF
    cannot be performed.
  """
  molecule_props = AllChem.MMFFGetMoleculeProperties(molecule)
  if molecule_props is None:
    return None

  mol = copy.deepcopy(molecule)
  for conf_id in range(mol.GetNumConformers()):
    ff = AllChem.MMFFGetMoleculeForceField(
        mol, molecule_props, confId=conf_id, ignoreInterfragInteractions=False)
    ff.Initialize()
    # minimises a conformer within a mol in place.
    ff.Minimize(max_iter)
  return mol


def _minimize_by_uff(
    molecule: Chem.rdchem.Mol,
    max_iter: int,
) -> Chem.rdchem.Mol:
  """Minimizes forcefield for conformers using UFF algorithm.

  Args:
    molecule: a datastructure containing conformers.
    max_iter: number of maximum iterations to use when optimising force field.

  Returns:
    A copy of a `molecule` containing optimised conformers.
  """
  mol = copy.deepcopy(molecule)
  conf_ids = range(mol.GetNumConformers())
  for conf_id in conf_ids:
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    ff.Initialize()
    # minimises a conformer within a mol in place.
    ff.Minimize(max_iter)
  return mol


def _get_symmetry_rotation_matrix(sign: tf.Tensor) -> tf.Tensor:
  """Returns the 2d/3d matrix for mirror symmetry."""
  zero = tf.zeros_like(sign)
  one = tf.ones_like(sign)
  # pylint: disable=bad-whitespace,bad-continuation
  rot = [sign,  zero,  zero,
         zero,  one,   zero,
         zero,  zero,   one]
  # pylint: enable=bad-whitespace,bad-continuation
  shape = (3, 3)
  rot = tf.stack(rot, axis=-1)
  rot = tf.reshape(rot, shape)
  return rot


def _quaternion_to_rotation_matrix(quaternion: tf.Tensor) -> tf.Tensor:
  """Converts a batch of quaternions to a batch of rotation matrices."""
  q0 = quaternion[0]
  q1 = quaternion[1]
  q2 = quaternion[2]
  q3 = quaternion[3]

  r00 = 2 * (q0 * q0 + q1 * q1) - 1
  r01 = 2 * (q1 * q2 - q0 * q3)
  r02 = 2 * (q1 * q3 + q0 * q2)
  r10 = 2 * (q1 * q2 + q0 * q3)
  r11 = 2 * (q0 * q0 + q2 * q2) - 1
  r12 = 2 * (q2 * q3 - q0 * q1)
  r20 = 2 * (q1 * q3 - q0 * q2)
  r21 = 2 * (q2 * q3 + q0 * q1)
  r22 = 2 * (q0 * q0 + q3 * q3) - 1

  matrix = tf.stack([r00, r01, r02,
                     r10, r11, r12,
                     r20, r21, r22], axis=-1)
  return tf.reshape(matrix, [3, 3])


def _get_random_rotation_3d() -> tf.Tensor:
  random_quaternions = tf.random.normal(
      shape=[4], dtype=tf.float32)
  random_quaternions /= tf.linalg.norm(
      random_quaternions, axis=-1, keepdims=True)
  return _quaternion_to_rotation_matrix(random_quaternions)


def _get_random_mirror_symmetry() -> tf.Tensor:
  random_0_1 = tf.random.uniform(
      shape=(), minval=0, maxval=2, dtype=tf.int32)
  random_signs = tf.cast((2 * random_0_1) - 1, tf.float32)
  return _get_symmetry_rotation_matrix(random_signs)
