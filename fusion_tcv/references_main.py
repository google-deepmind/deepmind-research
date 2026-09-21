# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Print the values of references."""

from typing import Sequence

from absl import app
from absl import flags

from fusion_tcv import named_array
from fusion_tcv import references
from fusion_tcv import tcv_common


_refs = flags.DEFINE_enum("refs", None, references.REFERENCES.keys(),
                          "Which references to print")
_count = flags.DEFINE_integer("count", 100, "How many timesteps to print.")
_freq = flags.DEFINE_integer("freq", 1, "Print only every so often.")
_fields = flags.DEFINE_multi_enum(
    "field", None, tcv_common.REF_RANGES.names(),
    "Which reference fields to print, default of all.")
flags.mark_flag_as_required("refs")


def print_ref(step: int, ref: named_array.NamedArray):
  print(f"Step: {step}")
  for k in (_fields.value or ref.names.names()):
    print(f"  {k}: [{', '.join(f'{v:.3f}' for v in ref[k])}]")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if _freq.value <= 0:
    raise app.UsageError("`freq` must be >0.")

  ref = references.REFERENCES[_refs.value]()
  print_ref(0, ref.reset())
  for i in range(1, _count.value + 1):
    for _ in range(_freq.value - 1):
      ref.step()
    print_ref(i * _freq.value, ref.step())
  print(f"Stopped after {_count.value * _freq.value} steps.")


if __name__ == "__main__":
  app.run(main)
