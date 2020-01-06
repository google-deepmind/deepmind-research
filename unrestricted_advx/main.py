# Copyright 2019 Deepmind Technologies Limited.
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

"""Submission to Unrestricted Adversarial Challenge."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from unrestricted_advex import eval_kit


def _preprocess_image(image):
  image = tf.image.central_crop(image, central_fraction=0.875)
  image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
  return image


def test_preprocess(image):
  image = _preprocess_image(image)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def main():
  g = tf.Graph()
  with g.as_default():
    input_tensor = tf.placeholder(tf.float32, (None, 224, 224, 3))
    x_np = test_preprocess(input_tensor)
    raw_module_1 = hub.Module(
        "https://tfhub.dev/deepmind/llr-pretrain-adv/latents/1")
    raw_module_2 = hub.Module(
        "https://tfhub.dev/deepmind/llr-pretrain-adv/linear/1")
    latents = raw_module_1(dict(inputs=x_np, decay_rate=0.1))
    logits = raw_module_2(dict(inputs=latents))
    logits = tf.squeeze(logits, axis=[1, 2])
    two_class_logits = tf.concat([tf.nn.relu(-logits[:, 1:]),
                                  tf.nn.relu(logits[:, 1:])], axis=1)
    sess = tf.train.SingularMonitoredSession()
    def model(x_np):
      return sess.run(two_class_logits, feed_dict={input_tensor: x_np})

    eval_kit.evaluate_bird_or_bicycle_model(model, model_name="llr_resnet")

if __name__ == "__main__":
  main()
