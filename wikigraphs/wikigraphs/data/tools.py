# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
#
# WikiGraphs is licensed under the terms of the Creative Commons
# Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.
#
# WikiText-103 data (unchanged) is licensed by Salesforce.com, Inc. under the
# terms of the Creative Commons Attribution-ShareAlike 4.0 International
# (CC BY-SA 4.0) license. You can find details about CC BY-SA 4.0 at:
#
#     https://creativecommons.org/licenses/by-sa/4.0/legalcode
#
# Freebase data is licensed by Google LLC under the terms of the Creative
# Commons CC BY 4.0 license. You may obtain a copy of the License at:
#
#     https://creativecommons.org/licenses/by/4.0/legalcode
#
# ==============================================================================
"""Some tools for processing data."""

from typing import Any, Iterator

from absl import logging
import numpy as np


def pad_to(x: np.array, size: int, axis: int = -1, pad_value: float = 0.):
  """Pad an array to the specified size along a specified axis."""
  if x.shape[axis] > size:
    raise ValueError(f'Data item has size {x.shape[axis]} larger than {size}'
                     f' in axis {axis} already.')
  elif x.shape[axis] == size:
    return x
  else:
    pad_amount = [(0, 0)] * x.ndim
    pad_amount[axis] = (0, size - x.shape[axis])
    return np.pad(x, pad_amount, mode='constant', constant_values=pad_value)


def dynamic_batch(
    iterable: Iterator[Any],
    batch_size: int,
    timesteps: int,
    return_incomplete_batch: bool = False,
    pad: bool = False,
    pad_value: float = 0.) -> Iterator[Any]:
  """Batches up values in iterable to [batch_size, timesteps].

  This function takes items from the iterable and pack them into the batch.
  Sequence #i in the batch is a continuation from the sequence #i in the
  previous batch, i.e. it will start from where the previous sequence left over.
  When an item is finished, a new item is taken from the iterable to append to
  the sequence and fill the batch.

  This function is designed for language modeling, where the input and the
  target sequences are offset by one.  We take that into account by making sure
  neighboring batches have one token overlap.

  Example:
    If the iterable contains [[0, 1, 2], [10, 11, 12, 13, 14], [20, 21, 22]] and
    batch size is 2, timesteps is 3, then the first batch would be:
      [[0, 1, 2],
       [10, 11, 12]]
    then the second batch:
      [[2, 20, 21],    # seq 0 finished, continuing from seq 2
       [12, 13, 14]]
    Note the overlap of 1 token between these two batches, and the continuation
    of sequences across batches.

  Args:
    iterable: the iterable that yields sequences of integer token IDs.
    batch_size: number of examples in a batch.
    timesteps: length of each sequence in a batch.
    return_incomplete_batch: if True return the incomplete batches, which
      typically appears at the end of the dataset.
    pad: set to True to pad the incomplete batches.
    pad_value: the value to use for padding.

  Yields:
    batches: where batches['obs'] are the observations of size
      [batch_size, timesteps], and batches['should_reset'] is a 0/1 mask of
      the same size that marks sequence boundaries, e.g. the entries in this
      mask are all 0 except at locations where a new sequence is starting.
  """
  if return_incomplete_batch and not pad:
    raise ValueError(
        f'If return_incomplete_batch, then pad must be True, currently {pad}.')

  iterator = iter(iterable)
  elems = []
  for _ in range(batch_size):
    item = next(iterator)
    elems.append(item)
  start_batch = [True] * batch_size

  iter_finished = False
  loaded_finished = False
  while not (iter_finished and loaded_finished):
    batch = []
    for i in range(batch_size):
      # should_reset value is 1 when a new sequence begins.
      # [old[-3], old[-2], old[-1], new[0], new[1], new[2]]
      # [0, 0, 0, 1, 0, 0]
      should_reset = np.zeros(timesteps, np.float32)
      if start_batch[i]:
        should_reset[0] = 1

      # Pack new examples in the sequence until they go beyond the required
      # timesteps.
      while len(elems[i]) < timesteps:
        should_reset[len(elems[i])] = 1
        try:
          item = next(iterator)
        except StopIteration:
          iter_finished = True
          break
        elems[i] = np.concatenate([elems[i], item])

      batch.append(dict(obs=elems[i][:timesteps], should_reset=should_reset))
      # Shift and make sure we have a 1 token overlap.
      elems[i] = elems[i][timesteps - 1:]
      # Since the last token is shifted to be the first token of the next batch,
      # We need to make sure reset is handled properly as well.
      start_batch[i] = (should_reset[-1] == 1)

    # If any loaded data is not yet consumed in the output we should keep
    # generating.
    loaded_finished = all(e.size == 0 for e in elems)

    if not return_incomplete_batch:
      elem_len = len(batch[0]['obs'])
      if (elem_len != timesteps or
          not all(len(x['obs']) == elem_len for x in batch[1:])):
        logging.info('Dropping the (last?) incomplete batch.')
        break

    if pad:
      for x in batch:
        x['obs'] = pad_to(x['obs'], timesteps, axis=0, pad_value=pad_value)

    yield dict(
        obs=np.stack([x['obs'] for x in batch], axis=0),
        should_reset=np.stack([x['should_reset'] for x in batch], axis=0))


def batch_graph_text_pairs(
    iterable: Iterator[Any],
    batch_size: int,
    timesteps: int,
    pad_value: float = 0.,
    seq_and_graph_id: bool = False) -> Iterator[Any]:
  """Batch graph and text pairs.

  This method pairs text with graphs, each text sequence is split into chunks
  (with an overlap of 1) of size `timesteps`, and the graph associated with the
  text is used and associated with each chunk as well.  The last incomplete
  chunk of each text sequence is padded with the `pad_value`.

  Args:
    iterable: Iterable that returns (graph, sequence) pairs, graph can be
      anything, and sequence is a list of tokenized token IDs.
    batch_size: Number of examples in a batch.
    timesteps: Window size for the sequences.
    pad_value: Value to use for padding.
    seq_and_graph_id: whether the `iterable` contains `seq_id` and `graph_id`.

  Yields:
    batch: a batch of text sequence paired with graphs.
  """
  iterator = iter(iterable)
  seqs = [None] * batch_size
  graphs = [None] * batch_size
  graph_ids = [None] * batch_size
  seq_ids = [None] * batch_size

  iter_finished = False
  loaded_finished = False
  while not (iter_finished and loaded_finished):
    batch = []
    for idx in range(batch_size):
      should_reset = np.zeros(timesteps, np.float32)
      # pylint: disable=g-explicit-length-test
      if seqs[idx] is None or len(seqs[idx]) == 0:
        should_reset[0] = 1
        # One sequence exhausted, get the next example.
        try:
          if seq_and_graph_id:
            (graph, seq), (graph_id, seq_id) = next(iterator)
            graph_ids[idx] = graph_id
            seq_ids[idx] = seq_id
          else:
            graph, seq = next(iterator)
          seqs[idx] = seq
          graphs[idx] = graph
        except StopIteration:
          iter_finished = True
          seqs[idx] = np.array([pad_value], dtype=np.int32)
          graphs[idx] = None

      example = dict(obs=seqs[idx][:timesteps], graph=graphs[idx],
                     should_reset=should_reset)
      if seq_and_graph_id:
        example['seq_id'] = seq_ids[idx]
        example['graph_id'] = graph_ids[idx]

      batch.append(example)
      # Make sure that there is an overlap, as we generate targets by shifting
      # the tensor by 1 timestep. So the next element should be shifted by
      # `timesteps - 1' timesteps.
      seqs[idx] = seqs[idx][timesteps - 1:]

    # Make sure all loaded data are consumed in the output
    loaded_finished = all(s.size == 0 for s in seqs)

    # Also check for the last batch to avoid returning a fully empty batch
    if iter_finished and all([np.all(b['obs'] == pad_value) for b in batch]):
      break

    # pad sequences to specified length
    for e in batch:
      e['obs'] = pad_to(e['obs'], timesteps, axis=0, pad_value=pad_value)
    stacked_batch = dict(
        obs=np.stack([e['obs'] for e in batch], axis=0),
        graphs=[e['graph'] for e in batch],
        should_reset=np.stack([e['should_reset'] for e in batch], axis=0))
    if seq_and_graph_id:
      stacked_batch['seq_id'] = np.stack(
          [e['seq_id'] for e in batch], axis=0)
      stacked_batch['graph_id'] = np.stack(
          [e['graph_id'] for e in batch], axis=0)
    yield stacked_batch
