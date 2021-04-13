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

"""Single file script for doing a quick evaluation of a model.

This script is called by run.sh.
Usage:
  user@host:/path/to/deepmind_research$ unsupervised_adversarial_training/run.sh
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl import app
from absl import flags
import cleverhans
from cleverhans import attacks
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

UAT_HUB_URL = ('https://tfhub.dev/deepmind/unsupervised-adversarial-training/'
               'cifar10/wrn_106/1')

FLAGS = flags.FLAGS
flags.DEFINE_enum('attack_fn_name', 'fgsm', ['fgsm', 'none'],
                  'Name of the attack method to use.')
flags.DEFINE_float('epsilon_attack', 8.0 / 255,
                   'Maximum allowable perturbation size, between 0 and 1.')
flags.DEFINE_integer('num_steps', 20, 'Number of attack iterations.')
flags.DEFINE_integer('num_batches', 100, 'Number of batches to evaluate.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('skip_batches', 0,
                     'Controls index of start image. This can be used to '
                     'evaluate the model on different subsets of the test set.')
flags.DEFINE_float('learning_rate', 0.003, 'Attack optimizer learning rate.')


def _top_1_accuracy(logits, labels):
  return tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))


def make_classifier():
  model = hub.Module(UAT_HUB_URL)

  def classifier(x):
    x = _cifar_meanstd_normalize(x)
    model_input = dict(x=x, decay_rate=0.1, prefix='default')
    return model(model_input)

  return classifier


def eval_cifar():
  """Evaluate an adversarially trained model."""
  attack_fn_name = FLAGS.attack_fn_name
  total_batches = FLAGS.num_batches
  batch_size = FLAGS.batch_size

  # Note that a `classifier` is a function mapping [0,1]-scaled image Tensors
  # to a logit Tensor. In particular, it includes *both* the preprocessing
  # function, and the neural network.
  classifier = make_classifier()
  cleverhans_model = cleverhans.model.CallableModelWrapper(classifier, 'logits')

  _, data_test = tf.keras.datasets.cifar10.load_data()
  data = _build_dataset(data_test, batch_size=batch_size, shuffle=False)

  # Generate adversarial images.
  if attack_fn_name == 'fgsm':
    attack = attacks.MadryEtAl(cleverhans_model)
    num_cifar_classes = 10
    adv_x = attack.generate(data.image,
                            eps=FLAGS.epsilon_attack,
                            eps_iter=FLAGS.learning_rate,
                            nb_iter=FLAGS.num_steps,
                            y=tf.one_hot(data.label, depth=num_cifar_classes))
  elif attack_fn_name == 'none':
    adv_x = data.image

  logits = classifier(adv_x)
  probs = tf.nn.softmax(logits)
  adv_acc = _top_1_accuracy(logits, data.label)

  with tf.train.SingularMonitoredSession() as sess:
    total_acc = 0.
    for _ in range(FLAGS.skip_batches):
      sess.run(data.image)
    for _ in range(total_batches):
      _, _, adv_acc_val = sess.run([probs, data.label, adv_acc])
      total_acc += adv_acc_val
      print('Batch accuracy: {}'.format(adv_acc_val))
    print('Total accuracy against {}: {}'.format(
        FLAGS.attack_fn_name, total_acc / total_batches))


##########    Utilities    ##########


# Defines a dataset sample."""
Sample = collections.namedtuple('Sample', ['image', 'label'])


def _build_dataset(raw_data, batch_size=32, shuffle=False):
  """Builds a dataset from raw NumPy tensors.

  Args:
    raw_data: Pair (images, labels) of numpy arrays. `images` should have shape
      (N, H, W, C) with values in [0, 255], and `labels` should have shape
      (N,) or (N, 1) indicating class indices.
    batch_size: int, batch size
    shuffle: bool, whether to shuffle the data (default: True).

  Returns:
    (image_tensor, label_tensor), which iterate over the dataset, which are
      (batch_size, H, W, C) tf.float32 and (batch_size,) tf.int32 Tensors
      respectively
  """
  images, labels = raw_data
  labels = np.squeeze(labels)
  samples = Sample(images.astype(np.float32) / 255., labels.astype(np.int64))
  data = tf.data.Dataset.from_tensor_slices(samples)
  if shuffle:
    data = data.shuffle(1000)
  return data.repeat().batch(batch_size).make_one_shot_iterator().get_next()


def _cifar_meanstd_normalize(image):
  """Mean + stddev whitening for CIFAR-10 used in ResNets.

  Args:
    image: Numpy array or TF Tensor, with values in [0, 255]

  Returns:
    image: Numpy array or TF Tensor, shifted and scaled by mean/stdev on
      CIFAR-10 dataset.
  """
  # Channel-wise means and std devs calculated from the CIFAR-10 training set
  cifar_means = [125.3, 123.0, 113.9]
  cifar_devs = [63.0, 62.1, 66.7]
  rescaled_means = [x / 255. for x in cifar_means]
  rescaled_devs = [x / 255. for x in cifar_devs]
  image = (image - rescaled_means) / rescaled_devs
  return image


def main(unused_argv):
  eval_cifar()

if __name__ == '__main__':
  app.run(main)
