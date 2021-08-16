# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenizer implementation mapping strings to their UTF-8 bytes."""

from typing import Union
import numpy as np


class BytesTokenizer:
  """Tokenizes string to utf-8 bytes."""

  def __init__(self):
    self._num_reserved_tokens = 6  # PAD, BOS, EOS, MASK, CLS, SEP

  def to_string(self, inputs: np.ndarray) -> str:
    inputs_no_special = (
        inputs[inputs >= self._num_reserved_tokens] - self._num_reserved_tokens)
    decoded_bytes = inputs_no_special.astype(np.uint8).tobytes()
    return decoded_bytes.decode('utf-8', errors='replace')

  def to_int(self, inputs: Union[str, bytes]) -> np.ndarray:
    if isinstance(inputs, str):
      inputs = inputs.encode('utf-8')
    encoded = np.frombuffer(inputs, np.uint8).astype(np.int32)
    encoded = encoded + self._num_reserved_tokens
    return encoded.astype(np.int32)

  @property
  def vocab_size(self) -> int:
    return 256 + self._num_reserved_tokens

  @property
  def pad_token(self) -> int:
    return 0

  @property
  def bos_token(self) -> int:
    return 1

  @property
  def eos_token(self) -> int:
    return 2

  @property
  def mask_token(self) -> int:
    return 3

  @property
  def cls_token(self) -> int:
    return 4

  @property
  def sep_token(self) -> int:
    return 5
