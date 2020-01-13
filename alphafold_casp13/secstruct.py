# Lint as: python3.
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Layer for modelling and scoring secondary structure."""

import os

from absl import logging
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


# 8-class classes (Q8)
SECONDARY_STRUCTURES = '-HETSGBI'

# Equivalence classes for 3-class (Q3) from Li & Yu 2016.
# See http://www.cmbi.ru.nl/dssp.html for letter explanations.
Q3_MAP = ['-TSGIB', 'H', 'E']


def make_q3_matrices():
  """Generate mapping matrices for secstruct Q8:Q3 equivalence classes."""
  dimension = len(SECONDARY_STRUCTURES)
  q3_map_matrix = np.zeros((dimension, len(Q3_MAP)))
  q3_lookup = np.zeros((dimension,), dtype=np.int32)
  for i, eclass in enumerate(Q3_MAP):  # equivalence classes
    for m in eclass:  # Members of the class.
      ss_type = SECONDARY_STRUCTURES.index(m)
      q3_map_matrix[ss_type, i] = 1.0
      q3_lookup[ss_type] = i
  return q3_map_matrix, q3_lookup


class Secstruct(object):
  """Make a layer that computes hierarchical secstruct."""
  # Build static, shared structures:
  q3_map_matrix, q3_lookup = make_q3_matrices()
  static_dimension = len(SECONDARY_STRUCTURES)

  def __init__(self, name='secstruct'):
    self.name = name
    self._dimension = Secstruct.static_dimension

  def make_layer_new(self, activations):
    """Make the layer."""
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      logging.info('Creating secstruct %s', activations)
      self.logits = tf.contrib.layers.linear(activations, self._dimension)
      self.ss_q8_probs = tf.nn.softmax(self.logits)
      self.ss_q3_probs = tf.matmul(
          self.ss_q8_probs, tf.constant(self.q3_map_matrix, dtype=tf.float32))

  def get_q8_probs(self):
    return self.ss_q8_probs


def save_secstructs(dump_dir_path, name, index, sequence, probs,
                    label='Deepmind secstruct'):
  """Write secstruct prob distributions to an ss2 file.

  Can be overloaded to write out asa values too.

  Args:
    dump_dir_path: directory where to write files.
    name: name of domain
    index: index number of multiple samples. (or None for no index)
    sequence: string of L residue labels
    probs: L x D matrix of probabilities. L is length of sequence,
      D is probability dimension (usually 3).
    label: A label for the file.
  """
  filename = os.path.join(dump_dir_path, '%s.ss2' % name)
  if index is not None:
    filename = os.path.join(dump_dir_path, '%s_%04d.ss2' % (name, index))
  with tf.io.gfile.GFile(filename, 'w') as gf:
    logging.info('Saving secstruct to %s', filename)
    gf.write('# %s CLASSES [%s] %s sample %s\n\n' % (
        label, ''.join(SECONDARY_STRUCTURES[:probs.shape[1]]), name, index))
    for l in range(probs.shape[0]):
      ss = SECONDARY_STRUCTURES[np.argmax(probs[l, :])]
      gf.write('%4d %1s %1s %s\n' % (l + 1, sequence[l], ss, ''.join(
          [('%6.3f' % p) for p in probs[l, :]])))
