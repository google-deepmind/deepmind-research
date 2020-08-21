# Copyright 2020 Deepmind Technologies Limited.
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

"""Tests for the PolyGen open-source version."""
from modules import FaceModel
from modules import VertexModel
import numpy as np
import tensorflow as tf

_BATCH_SIZE = 4
_TRANSFORMER_CONFIG = {
    'num_layers': 2,
    'hidden_size': 64,
    'fc_size': 256
}
_CLASS_CONDITIONAL = True
_NUM_CLASSES = 4
_NUM_INPUT_VERTS = 50
_NUM_PAD_VERTS = 10
_NUM_INPUT_FACE_INDICES = 200
_QUANTIZATION_BITS = 8
_VERTEX_MODEL_USE_DISCRETE_EMBEDDINGS = True
_FACE_MODEL_DECODER_CROSS_ATTENTION = True
_FACE_MODEL_DISCRETE_EMBEDDINGS = True
_MAX_SAMPLE_LENGTH_VERTS = 10
_MAX_SAMPLE_LENGTH_FACES = 10


def _get_vertex_model_batch():
  """Returns batch with placeholders for vertex model inputs."""
  return {
      'class_label': tf.range(_BATCH_SIZE),
      'vertices_flat': tf.placeholder(
          dtype=tf.int32, shape=[_BATCH_SIZE, None]),
  }


def _get_face_model_batch():
  """Returns batch with placeholders for face model inputs."""
  return {
      'vertices': tf.placeholder(
          dtype=tf.float32, shape=[_BATCH_SIZE, None, 3]),
      'vertices_mask': tf.placeholder(
          dtype=tf.float32, shape=[_BATCH_SIZE, None]),
      'faces': tf.placeholder(
          dtype=tf.int32, shape=[_BATCH_SIZE, None]),
  }


class VertexModelTest(tf.test.TestCase):

  def setUp(self):
    """Defines a vertex model."""
    super(VertexModelTest, self).setUp()
    self.model = VertexModel(
        decoder_config=_TRANSFORMER_CONFIG,
        class_conditional=_CLASS_CONDITIONAL,
        num_classes=_NUM_CLASSES,
        max_num_input_verts=_NUM_INPUT_VERTS,
        quantization_bits=_QUANTIZATION_BITS,
        use_discrete_embeddings=_VERTEX_MODEL_USE_DISCRETE_EMBEDDINGS)

  def test_model_runs(self):
    """Tests if the model runs without crashing."""
    batch = _get_vertex_model_batch()
    pred_dist = self.model(batch, is_training=False)
    logits = pred_dist.logits
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      vertices_flat = np.random.randint(
          2**_QUANTIZATION_BITS + 1,
          size=[_BATCH_SIZE, _NUM_INPUT_VERTS * 3 + 1])
      sess.run(logits, {batch['vertices_flat']: vertices_flat})

  def test_sample_outputs_range(self):
    """Tests if the model produces samples in the correct range."""
    context = {'class_label': tf.zeros((_BATCH_SIZE,), dtype=tf.int32)}
    sample_dict = self.model.sample(
        _BATCH_SIZE, max_sample_length=_MAX_SAMPLE_LENGTH_VERTS,
        context=context)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      sample_dict_np = sess.run(sample_dict)
      in_range = np.logical_and(
          0 <= sample_dict_np['vertices'],
          sample_dict_np['vertices'] <= 2**_QUANTIZATION_BITS).all()
      self.assertTrue(in_range)


class FaceModelTest(tf.test.TestCase):

  def setUp(self):
    """Defines a face model."""
    super(FaceModelTest, self).setUp()
    self.model = FaceModel(
        encoder_config=_TRANSFORMER_CONFIG,
        decoder_config=_TRANSFORMER_CONFIG,
        class_conditional=False,
        max_seq_length=_NUM_INPUT_FACE_INDICES,
        decoder_cross_attention=_FACE_MODEL_DECODER_CROSS_ATTENTION,
        use_discrete_vertex_embeddings=_FACE_MODEL_DISCRETE_EMBEDDINGS,
        quantization_bits=_QUANTIZATION_BITS)

  def test_model_runs(self):
    """Tests if the model runs without crashing."""
    batch = _get_face_model_batch()
    pred_dist = self.model(batch, is_training=False)
    logits = pred_dist.logits
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      vertices = np.random.rand(_BATCH_SIZE, _NUM_INPUT_VERTS, 3) - 0.5
      vertices_mask = np.ones([_BATCH_SIZE, _NUM_INPUT_VERTS])
      faces = np.random.randint(
          _NUM_INPUT_VERTS + 2, size=[_BATCH_SIZE, _NUM_INPUT_FACE_INDICES])
      sess.run(
          logits,
          {batch['vertices']: vertices,
           batch['vertices_mask']: vertices_mask,
           batch['faces']: faces}
          )

  def test_sample_outputs_range(self):
    """Tests if the model produces samples in the correct range."""
    context = _get_face_model_batch()
    del context['faces']
    sample_dict = self.model.sample(
        context, max_sample_length=_MAX_SAMPLE_LENGTH_FACES)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      # Pad the vertices in order to test that the face model only outputs
      # vertex indices in the unpadded range
      vertices = np.pad(
          np.random.rand(_BATCH_SIZE, _NUM_INPUT_VERTS, 3) - 0.5,
          [[0, 0], [0, _NUM_PAD_VERTS], [0, 0]], mode='constant')
      vertices_mask = np.pad(
          np.ones([_BATCH_SIZE, _NUM_INPUT_VERTS]),
          [[0, 0], [0, _NUM_PAD_VERTS]], mode='constant')
      sample_dict_np = sess.run(
          sample_dict,
          {context['vertices']: vertices,
           context['vertices_mask']: vertices_mask})
      in_range = np.logical_and(
          0 <= sample_dict_np['faces'],
          sample_dict_np['faces'] <= _NUM_INPUT_VERTS + 1).all()
      self.assertTrue(in_range)

if __name__ == '__main__':
  tf.test.main()
