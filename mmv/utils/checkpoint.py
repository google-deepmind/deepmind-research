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

"""Checkpoint restoring utilities."""

import pickle

from absl import logging


class _SafeUnpickler(pickle.Unpickler):
  """Restricts unpickling to numpy arrays and basic Python types.

  Replaces dill.load() to prevent arbitrary code execution when loading
  checkpoints from untrusted sources. Only the types present in MMV
  parameter/state dicts (nested dicts of numpy arrays) are allowed.
  """

  _ALLOWED = {
      "numpy": {"ndarray", "dtype"},
      "numpy.core.multiarray": {"_reconstruct", "scalar"},
      "numpy.core": {"multiarray"},
      "numpy.dtypes": {"Float32DType", "Float64DType", "Int32DType",
                       "Int64DType"},
      "builtins": {
          "dict", "list", "tuple", "set", "str", "int", "float",
          "bool", "bytes", "complex",
      },
  }

  def find_class(self, module, name):
    allowed_names = self._ALLOWED.get(module, set())
    if name in allowed_names:
      return super().find_class(module, name)
    raise pickle.UnpicklingError(
        f"Refusing to unpickle {module}.{name}: not in allowlist.")


def load_checkpoint(checkpoint_path):
  try:
    with open(checkpoint_path, "rb") as checkpoint_file:
      checkpoint_data = _SafeUnpickler(checkpoint_file).load()
      logging.info("Loading checkpoint from %s", checkpoint_path)
      return checkpoint_data
  except FileNotFoundError:
    return None
