#!/usr/bin/env python3
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

"""Helper for exporting a functional and its derivatives to a saved_model."""

from typing import Sequence

from absl import app
from absl import flags

from density_functional_approximation_dm21 import neural_numint

_OUT_DIR = flags.DEFINE_string(
    'out_dir', None, 'Output directory.', required=True)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    1000,
    'Number of grid points exported functional will process in a single call.',
    lower_bound=0)
_FUNCTIONAL = flags.DEFINE_enum_class('functional',
                                      neural_numint.Functional.DM21,
                                      neural_numint.Functional,
                                      'Functional to export.')


def export(
    functional: neural_numint.Functional,
    export_path: str,
    batch_dim: int,
) -> None:
  """Export a functional and its derivatives to a single saved_model.

  Args:
    functional: functional to export.
    export_path: path to saved the model to.
    batch_dim: number of grid points to process in a single call.
  """
  ni = neural_numint.NeuralNumInt(functional)
  ni.export_functional_and_derivatives(
      export_path=export_path, batch_dim=batch_dim)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  export(_FUNCTIONAL.value, _OUT_DIR.value, _BATCH_SIZE.value)


if __name__ == '__main__':
  app.run(main)
