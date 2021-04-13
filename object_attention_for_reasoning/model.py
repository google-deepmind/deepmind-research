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
"""Model code. Provided settings are identical to what was used in the paper."""

import sonnet as snt
import tensorflow.compat.v1 as tf

from object_attention_for_reasoning import transformer


QUESTION_VOCAB_SIZE = 82
ANSWER_VOCAB_SIZE = 22

MAX_QUESTION_LENGTH = 20
MAX_CHOICE_LENGTH = 12

NUM_CHOICES = 4
EMBED_DIM = 16

PRETRAINED_MODEL_CONFIG = dict(
    use_relative_positions=True,
    shuffle_objects=True,
    transformer_layers=28,
    head_size=128,
    num_heads=10,
    embed_dim=EMBED_DIM,
)


def append_ids(tensor, id_vector, axis):
  id_vector = tf.constant(id_vector, tf.float32)
  for a in range(len(tensor.shape)):
    if a != axis:
      id_vector = tf.expand_dims(id_vector, axis=a)
  tiling_vector = [s if i != axis else 1 for i, s in enumerate(tensor.shape)]
  id_tensor = tf.tile(id_vector, tiling_vector)
  return tf.concat([tensor, id_tensor], axis=axis)


class ClevrerTransformerModel(object):
  """Model from Ding et al. 2020 (https://arxiv.org/abs/2012.08508)."""

  def __init__(self, use_relative_positions, shuffle_objects,
               transformer_layers, num_heads, head_size, embed_dim):
    """Instantiate Sonnet modules."""
    self._embed_dim = embed_dim
    self._embed = snt.Embed(QUESTION_VOCAB_SIZE, embed_dim - 2)
    self._shuffle_objects = shuffle_objects
    self._memory_transformer = transformer.TransformerTower(
        value_size=embed_dim + 2,
        num_heads=num_heads,
        num_layers=transformer_layers,
        use_relative_positions=use_relative_positions,
        causal=False)

    self._final_layer_mc = snt.Sequential(
        [snt.Linear(head_size), tf.nn.relu, snt.Linear(1)])
    self._final_layer_descriptive = snt.Sequential(
        [snt.Linear(head_size), tf.nn.relu,
         snt.Linear(ANSWER_VOCAB_SIZE)])

    self._dummy = tf.get_variable("dummy", [embed_dim + 2], tf.float32,
                                  initializer=tf.zeros_initializer)
    self._infill_linear = snt.Linear(embed_dim + 2)
    self._mask_embedding = tf.get_variable(
        "mask", [embed_dim + 2], tf.float32, initializer=tf.zeros_initializer)

  def _apply_transformers(self, lang_embedding, vision_embedding):
    """Applies transformer to language and vision input.

    Args:
      lang_embedding: tensor,
      vision_embedding: tensor, "validation", or "test".

    Returns:
      tuple, output at dummy token, all output embeddings, infill loss
    """
    def _unroll(tensor):
      """Unroll the time dimension into the object dimension."""
      return tf.reshape(
          tensor, [tensor.shape[0], -1, tensor.shape[3]])

    words = append_ids(lang_embedding, [1, 0], axis=2)
    dummy_word = tf.tile(self._dummy[None, None, :], [tf.shape(words)[0], 1, 1])
    vision_embedding = append_ids(vision_embedding, [0, 1], axis=3)
    vision_over_time = _unroll(vision_embedding)
    transformer_input = tf.concat([dummy_word, words, vision_over_time], axis=1)

    output, _ = self._memory_transformer(transformer_input,
                                         is_training=False)
    return output[:, 0, :]

  def apply_model_descriptive(self, inputs):
    """Applies model to CLEVRER descriptive questions.

    Args:
      inputs: dict of form: {
        "question": tf.int32 tensor of shape [batch, MAX_QUESTION_LENGTH],
        "monet_latents": tf.float32 tensor of shape [batch, frames, 8, 16],
      }
    Returns:
      Tensor of shape [batch, ANSWER_VOCAB_SIZE], representing logits for each
      possible answer word.
    """
    question = inputs["question"]

    # Shape: [batch, question_len, embed_dim-2]
    question_embedding = self._embed(question)
    # Shape: [batch, question_len, embed_dim]
    question_embedding = append_ids(question_embedding, [0, 1], 2)
    choices_embedding = self._embed(
        tf.zeros([question.shape[0], MAX_CHOICE_LENGTH], tf.int64))
    choices_embedding = append_ids(choices_embedding, [0, 1], 2)
    # Shape: [batch, choices, question_len + choice_len, embed_dim]
    lang_embedding = tf.concat([question_embedding, choices_embedding], axis=1)

    # Shape: [batch, frames, num_objects, embed_dim]
    vision_embedding = inputs["monet_latents"]

    if self._shuffle_objects:
      vision_embedding = tf.transpose(vision_embedding, [2, 1, 0, 3])
      vision_embedding = tf.random.shuffle(vision_embedding)
      vision_embedding = tf.transpose(vision_embedding, [2, 1, 0, 3])

    output = self._apply_transformers(lang_embedding, vision_embedding)
    output = self._final_layer_descriptive(output)
    return output

  def apply_model_mc(self, inputs):
    """Applies model to CLEVRER multiple-choice questions.

    Args:
      inputs: dict of form: {
        "question": tf.int32 tensor of shape [batch, MAX_QUESTION_LENGTH],
        "choices": tf.int32 tensor of shape [batch, 4, MAX_CHOICE_LENGTH],
        "monet_latents": tf.float32 tensor of shape [batch, frames, 8, 16],
      }
    Returns:
      Tensor of shape [batch, 4], representing logits for each choice
    """
    question = inputs["question"]
    choices = inputs["choices"]

    # Shape: [batch, question_len, embed_dim-2]
    question_embedding = self._embed(question)
    # Shape: [batch, question_len, embed_dim]
    question_embedding = append_ids(question_embedding, [1, 0], 2)
    # Shape: [batch, choices, choice_len, embed_dim-2]
    choices_embedding = snt.BatchApply(self._embed)(choices)
    # Shape: [batch, choices, choice_len, embed_dim]
    choices_embedding = append_ids(choices_embedding, [0, 1], 3)
    # Shape: [batch, choices, question_len + choice_len, embed_dim]
    lang_embedding = tf.concat([
        tf.tile(question_embedding[:, None],
                [1, choices_embedding.shape[1], 1, 1]),
        choices_embedding], axis=2)

    # Shape: [batch, frames, num_objects, embed_dim]
    vision_embedding = inputs["monet_latents"]

    if self._shuffle_objects:
      vision_embedding = tf.transpose(vision_embedding, [2, 1, 0, 3])
      vision_embedding = tf.random.shuffle(vision_embedding)
      vision_embedding = tf.transpose(vision_embedding, [2, 1, 0, 3])

    output_per_choice = []
    for c in range(NUM_CHOICES):
      output = self._apply_transformers(
          lang_embedding[:, c, :, :], vision_embedding)
      output_per_choice.append(output)

    output = tf.stack(output_per_choice, axis=1)
    output = tf.squeeze(snt.BatchApply(self._final_layer_mc)(output), axis=2)
    return output
