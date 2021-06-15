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

"""PCQM4M-LSC datasets."""

import functools
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
from ogb import lsc

NUM_VALID_SAMPLES = 380_670
NUM_TEST_SAMPLES = 377_423

NORMALIZE_TARGET_MEAN = 5.690944545356371
NORMALIZE_TARGET_STD = 1.1561347795107815


def load_splits() -> Dict[str, List[int]]:
  """Loads dataset splits."""
  dataset = _get_pcq_dataset(only_smiles=True)
  return dataset.get_idx_split()


def load_kth_fold_indices(data_root: str, k_fold_split_id: int) -> List[int]:
  """Loads k-th fold indices."""
  fname = f"{data_root}/k_fold_splits/{k_fold_split_id}.pkl"
  return list(map(int, _load_pickle(fname)))


def load_all_except_kth_fold_indices(data_root: str, k_fold_split_id: int,
                                     num_k_fold_splits: int) -> List[int]:
  """Loads indices except for the kth fold."""
  if k_fold_split_id is None:
    raise ValueError("Expected integer value for `k_fold_split_id`.")
  indices = []
  for index in range(num_k_fold_splits):
    if index != k_fold_split_id:
      indices += load_kth_fold_indices(data_root, index)
  return indices


def load_smile_strings(
    with_labels=False) -> List[Union[str, Tuple[str, np.ndarray]]]:
  """Loads the smile strings in the PCQ dataset."""
  dataset = _get_pcq_dataset(only_smiles=True)
  smiles = []
  for i in range(len(dataset)):
    smile, label = dataset[i]
    if with_labels:
      smiles.append((smile, label))
    else:
      smiles.append(smile)

  return smiles


@functools.lru_cache()
def load_cached_conformers(cached_fname: str) -> Dict[str, np.ndarray]:
  """Returns cached dict mapping smile strings to conformer features."""
  return _load_pickle(cached_fname)


@functools.lru_cache()
def _get_pcq_dataset(only_smiles: bool):
  return lsc.PCQM4MDataset(only_smiles=only_smiles)


def _load_pickle(fname: str):
  with open(fname, "rb") as f:
    return pickle.load(f)
