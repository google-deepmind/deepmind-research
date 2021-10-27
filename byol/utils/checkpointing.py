# Copyright 2020 DeepMind Technologies Limited.
#
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
"""Checkpoint saving and restoring utilities."""

import os
import time
from typing import Mapping, Text, Tuple, Union

from absl import logging
import dill
import jax
import jax.numpy as jnp

from byol.utils import helpers


class Checkpointer:
  """A checkpoint saving and loading class."""

  def __init__(
      self,
      use_checkpointing: bool,
      checkpoint_dir: Text,
      save_checkpoint_interval: int,
      filename: Text):
    if (not use_checkpointing or
        checkpoint_dir is None or
        save_checkpoint_interval <= 0):
      self._checkpoint_enabled = False
      return

    self._checkpoint_enabled = True
    self._checkpoint_dir = checkpoint_dir
    os.makedirs(self._checkpoint_dir, exist_ok=True)
    self._filename = filename
    self._checkpoint_path = os.path.join(self._checkpoint_dir, filename)
    self._last_checkpoint_time = 0
    self._checkpoint_every = save_checkpoint_interval

  def maybe_save_checkpoint(
      self,
      experiment_state: Mapping[Text, jnp.ndarray],
      step: int,
      rng: jnp.ndarray,
      is_final: bool):
    """Saves a checkpoint if enough time has passed since the previous one."""
    current_time = time.time()
    if (not self._checkpoint_enabled or
        jax.host_id() != 0 or  # Only checkpoint the first worker.
        (not is_final and
         current_time - self._last_checkpoint_time < self._checkpoint_every)):
      return
    checkpoint_data = dict(
        experiment_state=jax.tree_map(
            lambda x: jax.device_get(x[0]), experiment_state),
        step=step,
        rng=rng)
    with open(self._checkpoint_path + '_tmp', 'wb') as checkpoint_file:
      dill.dump(checkpoint_data, checkpoint_file, protocol=2)
    try:
      os.rename(self._checkpoint_path, self._checkpoint_path + '_old')
      remove_old = True
    except FileNotFoundError:
      remove_old = False  # No previous checkpoint to remove
    os.rename(self._checkpoint_path + '_tmp', self._checkpoint_path)
    if remove_old:
      os.remove(self._checkpoint_path + '_old')
    self._last_checkpoint_time = current_time

  def maybe_load_checkpoint(
      self) -> Union[Tuple[Mapping[Text, jnp.ndarray], int, jnp.ndarray], None]:
    """Loads a checkpoint if any is found."""
    checkpoint_data = load_checkpoint(self._checkpoint_path)
    if checkpoint_data is None:
      logging.info('No existing checkpoint found at %s', self._checkpoint_path)
      return None
    step = checkpoint_data['step']
    rng = checkpoint_data['rng']
    experiment_state = jax.tree_map(
        helpers.bcast_local_devices, checkpoint_data['experiment_state'])
    del checkpoint_data
    return experiment_state, step, rng


def load_checkpoint(checkpoint_path):
  try:
    with open(checkpoint_path, 'rb') as checkpoint_file:
      checkpoint_data = dill.load(checkpoint_file)
      logging.info('Loading checkpoint from %s, saved at step %d',
                   checkpoint_path, checkpoint_data['step'])
      return checkpoint_data
  except FileNotFoundError:
    return None
