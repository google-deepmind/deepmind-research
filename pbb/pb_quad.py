# Copyright 2020 DeepMind Technologies Limited
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
"""PAC Bayesian bounds for Neural network.

Implements PAC Bayes quadratic bound on MNIST dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import sonnet.python.custom_getters.bayes_by_backprop as bbb
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def make_diagonal_gauss_posterior_builder(rho):
  """Makes diagonal gaussian posterior distributions.

  Args:
    rho: float, Variable used to parameterize standard deviation of the Normal
      distribution.

  Returns:
    A function which builds diagonal gaussian posterior distributions.
  """

  def diagonal_gauss_posterior_builder(getter, name, shape, *args, **kwargs):
    """Builds diagonal Gaussian posterior distributions.

    Given a true `getter` function and arguments forwarded from
    `tf.get_variable`, returns a distribution object for a diagonal posterior
    over a variable of the requisite shape.

    Args:
      getter: The `getter` passed to a `custom_getter`. Please see the
        documentation for `tf.get_variable`.
      name: The `name` argument passed to `tf.get_variable`.
      shape: The `shape` argument passed to `tf.get_variable`.
      *args: See positional arguments passed to `tf.get_variable`.
      **kwargs: See keyword arguments passed to `tf.get_variable`.

    Returns:
      An instance of `tfp.distributions.Normal` representing the posterior
      distribution over the variable in question.
    """
    # Please see the documentation for
    # `tfp.distributions.param_static_shapes`.
    parameter_shapes = tfp.distributions.Normal.param_static_shapes(shape)

    loc_var = getter(
        name + "/posterior_loc", shape=parameter_shapes["loc"], *args, **kwargs)
    kwargs["initializer"] = bbb.scale_variable_initializer(rho)
    scale_var = getter(
        name + "/posterior_scale",
        shape=parameter_shapes["scale"],
        *args,
        **kwargs)
    posterior = tfp.distributions.Normal(
        loc=loc_var,
        scale=tf.nn.softplus(scale_var),
        name="{}_posterior_dist".format(name))
    return posterior

  return diagonal_gauss_posterior_builder


def reinit_prior_to_posterior(posterior_prior_map):
  """Assigns prior values to the initial value of posterior weights.

  Note: This needs to be done once before the training iterations begin.

  Args:
    posterior_prior_map: Dictionary tf.Variable -> tf.Variable, A mapping from
      posterior variable to corresponding prior variable.

  Returns:
   An assign op which when executed will assign prior to posterior.
  """
  assign_ops = []
  for post_var_name, prior_var in posterior_prior_map.items():
    post_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 post_var_name)[0]
    assign_ops.append(prior_var.assign(post_var))
  return tf.group(*assign_ops)


def load_mnist_data(train_batch_size, test_batch_size):
  """Returns MNIST train and test data.

  Args:
    train_batch_size: int, Train Mini Batch size.
    test_batch_size: int, Test Mini Batch size.

  Returns:
   Train and test data tuples each containing
   ip - A  (batch_size, 784) tensor, label- A (batch_size, 1)tensor. of labels,
   label_onehot - A (batch_size, 10)tensor of one hot labels.
  """
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  # Preprocess the data (these are Numpy arrays)
  x_train = x_train.reshape(-1, 784).astype("float32") / 255
  x_test = x_test.reshape(-1, 784).astype("float32") / 255

  y_train = y_train.astype("float32")
  y_test = y_test.astype("float32")

  # We will use 50000 examples for training.
  x_train = x_train[:-10000]
  y_train = y_train[:-10000]
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  # Shuffle and slice the dataset.
  train_dataset = train_dataset.shuffle(
      buffer_size=1024).repeat().batch(train_batch_size)

  # Now we get a test dataset.
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  test_dataset = test_dataset.repeat().batch(test_batch_size)

  train_ip, train_label = train_dataset.make_one_shot_iterator().get_next()
  test_ip, test_label = test_dataset.make_one_shot_iterator().get_next()
  train_label = tf.cast(train_label, tf.int64)
  test_label = tf.cast(test_label, tf.int64)

  train_label_onehot = tf.one_hot(train_label, 10)
  test_label_onehot = tf.one_hot(test_label, 10)

  return (train_ip, train_label, train_label_onehot), (test_ip, test_label,
                                                       test_label_onehot)


def _compute_acc(logits, labels):
  """Computes accuracy.

  Args:
    logits: (batch_size, 10), Logits obtained from the forwards pass from the
      model.
    labels: (batch_size, int), True labels.

  Returns:
    float, Accuracy.
  """
  correct_prediction = tf.argmax(logits, 1)
  return tf.reduce_mean(
      tf.cast(tf.equal(correct_prediction, labels), tf.float32),
      name="accuracy")


def _compute_bounded_train_loss_acc(logits, label, label_onehot, p_min=1e-8):
  """Computes cross entropy loss and upper bounds by log(1./ p_min).

  Args:
    logits: (batch_size, 10), Logits obtained from the forwards pass from the
      model.
    label: (batch_size, int), True labels.
    label_onehot: (batch_size, 10), One hot true label.
    p_min: float, Minimum probability value to which the softmax will be
      clipped.

  Returns:
    (float, float), Loss and accuracy.
  """
  acc = _compute_acc(logits, label)
  logits = tf.nn.softmax(logits)
  predictions = tf.math.maximum(p_min, logits)

  with tf.control_dependencies([predictions, logits]):
    individual_loss = tf.reduce_sum(
        -1. * tf.math.multiply(label_onehot, tf.log(predictions)), axis=1)
    loss = tf.reduce_mean(individual_loss)
    return loss, acc


def _compute_loss_acc(logits, label, label_onehot):
  acc = _compute_acc(logits, label)
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
          labels=label_onehot, logits=logits),
      axis=0)

  return loss, acc


class PBQuad(object):
  """PAC Bayes Quadratic bound class."""

  def __init__(self,
               layer_spec,
               delta=0.05,
               learning_rate=5e-3,
               momentum=0.95,
               train_batch_size=256,
               test_batch_size=10000,
               train_data_size=50000,
               prior_rho=3e-2,
               loss_p_min=1e-4,
               prediction_mode="mean",
               name="pbb_trainer"):
    """Initializes the PAC Bayes Quad class.

    Args:
     layer_spec: Tuple of tuples, Model architecture specification. Check
        'mnist_config' config.core,layer_spec for further details.
      delta: float, PAC Bayes confidence parameter.
      learning_rate: float, Learning rate value for the Momentum optimizer.
      momentum: float, Momentum value for the SGD Momentum optimizer.
      train_batch_size: int, Train minibatch size.
      test_batch_size: int, Test minibatch size.
      train_data_size: int, Total number of training samples in the dataset.
      prior_rho: float, rho value used to initialize the standard deviation of
        prior.
      loss_p_min: float, Minimum probability value allowed in the cross entropy
        loss while training. log(1. / loss_p_min) is an upperbound on training
        loss function.
      prediction_mode: string, If set to "mean" the mean value of the
        distribution is used to make prediction on the test data.
      name: string, Name of the class.
    """

    self._delta = delta
    self._learning_rate = learning_rate
    self._momentum = momentum
    self._train_batch_size = train_batch_size
    self._train_data_size = train_data_size
    self._test_batch_size = test_batch_size
    self._prior_rho = prior_rho
    self._loss_p_min = loss_p_min
    self._prediction_mode = prediction_mode
    self._layer_spec = layer_spec
    self._name = name

    self._core_net_train = None
    self._core_net_test = None
    self._posterior_prior_map = {}
    self._update_ops = {}

  @property
  def update_ops(self):
    """Returns a dictionary of update ops for running and logging."""
    return self._update_ops

  @property
  def posterior_prior_map(self):
    return self._posterior_prior_map

  def _get_model(self, is_training):
    """Builds PAC Bayes (PB) training or test model.

    Args:
      is_training: boolean, For training, the network weights are sampled. For
        testing, if the prediction mode is set to "mean" then mean value of the
        posterior network weights is used to make prediction on test dataset.
        Otherwise the network weights are sampled.

    Returns:
       PB model which can used to train or test. The weights of the model are
       random variables. Any reference to a particular
       weight in the model will lead to sampling from the distribution.
    """

    def gauss_prior_getter(getter, name, *args, **kwargs):
      """Creates relevant prior variables.

        The actual values of the prior will be rewritten when we reinitialize
        the prior to the posterior.

      Args:
        getter: The `getter` passed to a `custom_getter`. Please see the
          documentation for `tf.get_variable`.
        name: The `name` argument passed to `tf.get_variable`.
        *args: See positional arguments passed to `tf.get_variable`.
        **kwargs: See keyword arguments passed to `tf.get_variable`.

      Returns:
        Instance of 'tfp.distributions.Normal'.
      """
      loc_var = getter(name + "_prior_loc", *args, **kwargs)
      self._posterior_prior_map[name + "/posterior_loc"] = loc_var
      return tfp.distributions.Normal(
          loc=loc_var, scale=self._prior_rho, name="{}_prior_dist".format(name))

    if is_training:
      sampling_mode_tensor = tf.constant(bbb.EstimatorModes.sample)
    else:
      if self._prediction_mode == "mean":
        sampling_mode_tensor = tf.constant(bbb.EstimatorModes.mean)
      else:
        sampling_mode_tensor = tf.constant(bbb.EstimatorModes.sample)

    prior_builder = gauss_prior_getter
    posterior_builder = make_diagonal_gauss_posterior_builder(self._prior_rho)

    getter = bbb.bayes_by_backprop_getter(
        prior_builder=prior_builder,
        posterior_builder=posterior_builder,
        kl_builder=bbb.analytic_kl_builder,
        sampling_mode_tensor=sampling_mode_tensor)

    with tf.variable_scope(
        self._name, custom_getter=getter, reuse=tf.AUTO_REUSE):
      core_net = snt.nets.MLP(
          output_sizes=self._layer_spec[0][1],
          activation=tf.nn.relu,
          activate_final=self._layer_spec[0][2])
    return core_net

  def _kl_cost(self):
    """Returns KL Divergence between all model prior and posterior variables."""
    return bbb.get_total_kl_cost(filter_by_name_substring=self._name)

  def _load_data(self):
    with tf.variable_scope("{}_data".format(self._name)):
      (self._train_ip, self._train_label,
       self._train_label_onehot), (self._test_ip, self._test_label,
                                   self._test_label_onehot) = load_mnist_data(
                                       self._train_batch_size,
                                       self._test_batch_size)

  def build_train_ops(self):
    """Builds PB Quadratic bound training graph."""
    tf.logging.info("Building graph for PBB trainer")

    self._load_data()
    with tf.variable_scope(self._name):
      self._core_net_train = self._get_model(is_training=True)
      self._core_net_test = self._get_model(is_training=False)

      with tf.name_scope("loss"):
        train_logits = self._core_net_train(self._train_ip)
        train_data_size = tf.cast(self._train_data_size, tf.float32)

        train_loss, train_acc = _compute_bounded_train_loss_acc(
            train_logits,
            self._train_label,
            self._train_label_onehot,
            p_min=self._loss_p_min)

        kl_div = self._kl_cost()
        pac_ub = (kl_div + tf.log(2. * tf.sqrt(train_data_size) /
                                  self._delta)) / (2. * train_data_size)

        self._update_ops["pac_ub"] = pac_ub
        self._update_ops["kl_div_n"] = kl_div / train_data_size
        bounded_train_loss = tf.sqrt(
            1. / tf.log(1. / self._loss_p_min)) * train_loss
        total_loss = tf.square(
            tf.sqrt(bounded_train_loss + pac_ub) + tf.sqrt(pac_ub))

        with tf.name_scope("optimizer"):
          opt = tf.train.MomentumOptimizer(
              self._learning_rate, momentum=self._momentum)

          self.var_list = tf.get_collection(
              tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
          train_op = opt.minimize(total_loss, var_list=self.var_list)

      self._update_ops["train_op"] = train_op
      self._update_ops["train_loss"] = train_loss
      self._update_ops["train_acc"] = train_acc
      self._update_ops["total_loss"] = total_loss

  def eval(self):
    """Evaluates the model on the test data.

    Returns:
      A dictionary of str->float with test loss and test accuracy as values.
    """
    with tf.name_scope(self._name):
      with tf.name_scope("test"):
        test_logits = self._core_net_test(self._test_ip)
        (test_loss, test_acc) = _compute_loss_acc(test_logits, self._test_label,
                                                  self._test_label_onehot)
        return {
            "loss": test_loss,
            "acc": test_acc,
        }
