# Lint as: python2, python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""nest utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest


def _nest_apply_over_list(list_of_nests, fn):
  """Equivalent to fn, but works on list-of-nests.

  Transforms a list-of-nests to a nest-of-lists, then applies `fn`
  to each of the inner lists.

  It is assumed that all nests have the same structure. Elements of the nest may
  be None, in which case they are ignored, i.e. they do not form part of the
  stack. This is useful when stacking agent states where parts of the state nest
  have been filtered.

  Args:
    list_of_nests: A Python list of nests.
    fn: the function applied on the list of leaves.

  Returns:
    A nest-of-arrays, where the arrays are formed by `fn`ing a list.
  """
  list_of_flat_nests = [nest.flatten(n) for n in list_of_nests]
  flat_nest_of_stacks = []
  for position in range(len(list_of_flat_nests[0])):
    new_list = [flat_nest[position] for flat_nest in list_of_flat_nests]
    new_list = [x for x in new_list if x is not None]
    flat_nest_of_stacks.append(fn(new_list))
  return nest.pack_sequence_as(
      structure=list_of_nests[0], flat_sequence=flat_nest_of_stacks)


def _take_indices(inputs, indices):
  return nest.map_structure(lambda t: np.take(t, indices, axis=0), inputs)


def nest_stack(list_of_nests, axis=0):
  """Equivalent to np.stack, but works on list-of-nests.

  Transforms a list-of-nests to a nest-of-lists, then applies `np.stack`
  to each of the inner lists.

  It is assumed that all nests have the same structure. Elements of the nest may
  be None, in which case they are ignored, i.e. they do not form part of the
  stack. This is useful when stacking agent states where parts of the state nest
  have been filtered.

  Args:
    list_of_nests: A Python list of nests.
    axis: Optional, the `axis` argument for `np.stack`.

  Returns:
    A nest-of-arrays, where the arrays are formed by `np.stack`ing a list.
  """
  return _nest_apply_over_list(list_of_nests, lambda l: np.stack(l, axis=axis))


def nest_unstack(batched_inputs, batch_size):
  """Splits a sequence of numpy arrays along 0th dimension."""
  return [_take_indices(batched_inputs, idx) for idx in range(batch_size)]
