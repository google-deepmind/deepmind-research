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

"""Checkpointing functionality."""

import os
from typing import Any, Mapping, Optional

from absl import logging
import dill
import jax
import jax.numpy as jnp


class Checkpointer:
  """A checkpoint saving and loading class."""

  def __init__(self, checkpoint_dir: str, filename: str):
    """Class initializer.

    Args:
     checkpoint_dir: Checkpoint directory.
     filename: Filename of checkpoint in checkpoint directory.
    """
    self._checkpoint_dir = checkpoint_dir
    if not os.path.isdir(self._checkpoint_dir):
      os.mkdir(self._checkpoint_dir)

    self._checkpoint_path = os.path.join(self._checkpoint_dir, filename)

  def save_checkpoint(
      self,
      experiment_state: Mapping[str, jnp.ndarray],
      opt_state: Mapping[str, jnp.ndarray],
      step: int,
      extra_checkpoint_info: Optional[Mapping[str, Any]] = None) -> None:
    """Save checkpoint with experiment state and step information.

    Args:
     experiment_state: Experiment params to be stored.
     opt_state: Optimizer state to be stored.
     step: Training iteration step.
     extra_checkpoint_info: Extra information to be stored.
    """
    if jax.host_id() != 0:
      return

    checkpoint_data = dict(
        experiment_state=jax.tree_map(jax.device_get, experiment_state),
        opt_state=jax.tree_map(jax.device_get, opt_state),
        step=step)
    if extra_checkpoint_info is not None:
      for key in extra_checkpoint_info:
        checkpoint_data[key] = extra_checkpoint_info[key]

    with open(self._checkpoint_path, 'wb') as checkpoint_file:
      dill.dump(checkpoint_data, checkpoint_file, protocol=2)

  def load_checkpoint(
      self) -> Optional[Mapping[str, Mapping[str, jnp.ndarray]]]:
    """Load and return checkpoint data.

    Returns:
     Loaded checkpoint if it exists else returns None.
    """

    if os.path.exists(self._checkpoint_path):
      with open(self._checkpoint_path, 'rb') as checkpoint_file:
        checkpoint_data = dill.load(checkpoint_file)
        logging.info('Loading checkpoint from %s, saved at step %d',
                     self._checkpoint_path, checkpoint_data['step'])
        return checkpoint_data
    else:
      logging.warning('No pre-saved checkpoint found at %s',
                      self._checkpoint_path)
      return None
