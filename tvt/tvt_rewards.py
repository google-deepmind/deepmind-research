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
"""Temporal Value Transport implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent import futures
import numpy as np
from six.moves import range
from six.moves import zip


def _unstack(array, axis):
  """Opposite of np.stack."""
  split_array = np.split(array, array.shape[axis], axis=axis)
  return [np.squeeze(a, axis=axis) for a in split_array]


def _top_k_args(array, k):
  """Return top k arguments or all arguments if array size is less than k."""
  if len(array) <= k:
    return np.arange(len(array))
  return np.argpartition(array, kth=-k)[-k:]


def _threshold_read_event_times(read_strengths, threshold):
  """Return the times of max read strengths within one threshold read event."""
  chosen_times = []
  over_threshold = False
  max_read_strength = 0.
  # Wait until the threshold is crossed then keep track of max read strength and
  # time of max read strength until the read strengths go back under the
  # threshold, then add that max read strength time to the chosen times. Wait
  # until threshold is crossed again and then repeat the process.
  for time, strength in enumerate(read_strengths):
    if strength > threshold:
      over_threshold = True
      if strength > max_read_strength:
        max_read_strength = strength
        max_read_strength_time = time
    else:
      # If coming back under threshold, add the time of the last max read.
      if over_threshold:
        chosen_times.append(max_read_strength_time)
        max_read_strength = 0.
      over_threshold = False
  # Add max read strength time if episode finishes before going under threshold.
  if over_threshold:
    chosen_times.append(max_read_strength_time)
  return np.array(chosen_times)


def _tvt_rewards_single_head(read_weights, read_strengths, read_times,
                             baselines, alpha, top_k_t1,
                             read_strength_threshold, no_transport_period):
  """Compute TVT rewards for a single read head, no batch dimension.

  This performs the updates for one read head.
  `t1` and `t2` refer to times to where and from where the value is being
  transported, respectively. I.e. the rewards at `t1` times are being modified
  based on values at times `t2`.

  Args:
    read_weights: shape (ep_length, top_k).
    read_strengths: shape (ep_length,).
    read_times: shape (ep_length, top_k).
    baselines: shape (ep_length,).
    alpha: The multiplier for the temporal value transport rewards.
    top_k_t1: For each read event time, this determines how many time points
      to send tvt reward to.
    read_strength_threshold: Read strengths below this value are ignored.
    no_transport_period: Length of no_transport_period.

  Returns:
    An array of TVT rewards with shape (ep_length,).
  """
  tvt_rewards = np.zeros_like(baselines)

  # Mask read_weights for reads that read back to times within
  # no_transport_period of current time.
  ep_length = read_times.shape[0]
  times = np.arange(ep_length)
  # Expand dims for correct broadcasting when subtracting read_times.
  times = np.expand_dims(times, -1)
  read_past_no_transport_period = (times - read_times) > no_transport_period
  read_weights_masked = np.where(read_past_no_transport_period,
                                 read_weights,
                                 np.zeros_like(read_weights))

  # Find t2 times with maximum read weights. Ignore t2 times whose maximum
  # read weights fall inside the no_transport_period.
  max_read_weight_args = np.argmax(read_weights, axis=1)  # (ep_length,)
  times = np.arange(ep_length)
  max_read_weight_times = read_times[times,
                                     max_read_weight_args]  # (ep_length,)
  read_strengths_cut = np.where(
      times - max_read_weight_times > no_transport_period,
      read_strengths,
      np.zeros_like(read_strengths))

  # Filter t2 candidates to perform value transport on local maximums
  # above a threshold.
  t2_times_with_largest_reads = _threshold_read_event_times(
      read_strengths_cut, read_strength_threshold)

  # Loop through all t2 candidates and transport value to top_k_t1 read times.
  for t2 in t2_times_with_largest_reads:
    try:
      baseline_value_when_reading = baselines[t2]
    except IndexError:
      raise RuntimeError("Attempting to access baselines array with length {}"
                         " at index {}. Make sure output_baseline is set in"
                         " the agent config.".format(len(baselines), t2))
    read_times_from_t2 = read_times[t2]
    read_weights_from_t2 = read_weights_masked[t2]

    # Find the top_k_t1 read times for this t2 and their corresponding read
    # weights. The call to _top_k_args() here gives the array indices for the
    # times and weights of the top_k_t1 reads from t2.
    top_t1_indices = _top_k_args(read_weights_from_t2, top_k_t1)
    top_t1_read_times = np.take(read_times_from_t2, top_t1_indices)
    top_t1_read_weights = np.take(read_weights_from_t2, top_t1_indices)

    # For each of the top_k_t1 read times t and corresponding read weight w,
    # find the trajectory that contains step_num (t + shift) and modify the
    # reward at step_num (t + shift) using w and the baseline value at t2.
    # We ignore any read times t >= t2. These can emerge because if nothing
    # in memory matches positively with the read query, the top reads may be
    # in the empty region of the memory.
    for step_num, read_weight in zip(top_t1_read_times, top_t1_read_weights):
      if step_num >= t2:
        # Skip this step_num as it is not really a memory time.
        continue

      # Compute the tvt reward and add it on.
      tvt_reward = alpha * read_weight * baseline_value_when_reading
      tvt_rewards[step_num] += tvt_reward

  return tvt_rewards


def _compute_tvt_rewards_from_read_info(
    read_weights, read_strengths, read_times, baselines, gamma,
    alpha=0.9, top_k_t1=50,
    read_strength_threshold=2.,
    no_transport_period_when_gamma_1=25):
  """Compute TVT rewards given supplied read information, no batch dimension.

  Args:
    read_weights: shape (ep_length, num_read_heads, top_k).
    read_strengths: shape (ep_length, num_read_heads).
    read_times: shape (ep_length, num_read_heads, top_k).
    baselines: shape (ep_length,).
    gamma: Scalar discount factor used to calculate the no_transport_period.
    alpha: The multiplier for the temporal value transport rewards.
    top_k_t1: For each read event time, this determines how many time points
      to send tvt reward to.
    read_strength_threshold: Read strengths below this value are ignored.
    no_transport_period_when_gamma_1: no transport period when gamma == 1.

  Returns:
    An array of TVT rewards with shape (ep_length,).
  """

  if gamma < 1:
    no_transport_period = int(1 / (1 - gamma))
  else:
    if no_transport_period_when_gamma_1 is None:
      raise ValueError("No transport period must be defined when gamma == 1.")
    no_transport_period = no_transport_period_when_gamma_1

  # Split read infos by read head.
  num_read_heads = read_weights.shape[1]
  read_weights = _unstack(read_weights, axis=1)
  read_strengths = _unstack(read_strengths, axis=1)
  read_times = _unstack(read_times, axis=1)

  # Calcuate TVT rewards for each read head separately and add to total.
  tvt_rewards = np.zeros_like(baselines)
  for i in range(num_read_heads):
    tvt_rewards += _tvt_rewards_single_head(
        read_weights[i], read_strengths[i], read_times[i],
        baselines, alpha, top_k_t1, read_strength_threshold,
        no_transport_period)

  return tvt_rewards


def compute_tvt_rewards(read_infos, baselines, gamma=.96):
  """Compute TVT rewards from EpisodeOutputs.

  Args:
    read_infos: A memory_reader.ReadInformation namedtuple, where each element
      has shape (ep_length, batch_size, num_read_heads, ...).
    baselines: A numpy float array with shape (ep_length, batch_size).
    gamma: Discount factor.

  Returns:
    An array of TVT rewards with shape (ep_length,).
  """
  if not read_infos:
    return np.zeros_like(baselines)

  # TVT reward computation is without batch dimension. so we need to process
  # read_infos and baselines into batchwise components.
  batch_size = baselines.shape[1]

  # Split each element of read info on batch dim.
  read_weights = _unstack(read_infos.weights, axis=1)
  read_strengths = _unstack(read_infos.strengths, axis=1)
  read_indices = _unstack(read_infos.indices, axis=1)
  # Split baselines on batch dim.
  baselines = _unstack(baselines, axis=1)

  # Comute TVT rewards for each element in the batch (threading over batch).
  tvt_rewards = []
  with futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
    for i in range(batch_size):
      tvt_rewards.append(
          executor.submit(
              _compute_tvt_rewards_from_read_info,
              read_weights[i],
              read_strengths[i],
              read_indices[i],
              baselines[i],
              gamma)
          )
    tvt_rewards = [f.result() for f in tvt_rewards]

  # Process TVT rewards back into an array of shape (ep_length, batch_size).
  return np.stack(tvt_rewards, axis=1)
