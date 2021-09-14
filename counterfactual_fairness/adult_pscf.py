# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Training script for causal model for Adult dataset, using PSCF."""

import functools
import time
from typing import Any, List, Mapping, NamedTuple, Sequence

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections.config_flags import config_flags
import numpy as np
import optax
import pandas as pd
from sklearn import metrics
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_probability.substrates import jax as tfp
from counterfactual_fairness import adult
from counterfactual_fairness import causal_network
from counterfactual_fairness import utils
from counterfactual_fairness import variational

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config', 'adult_pscf_config.py', 'Training configuration.')

LOG_EVERY = 100

# These are all aliases to callables which will return instances of
# particular distribution modules, or a Node itself. This is used to make
# subsequent code more legible.
Node = causal_network.Node
Gaussian = causal_network.Gaussian
MLPMultinomial = causal_network.MLPMultinomial


def build_input(train_data: pd.DataFrame, batch_size: int,
                training_steps: int, shuffle_size: int = 10000):
  """See base class."""
  num_epochs = (training_steps // batch_size) + 1
  ds = utils.get_dataset(train_data, batch_size, shuffle_size,
                         num_epochs=num_epochs)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return iter(tfds.as_numpy(ds))


class CausalNetOutput(NamedTuple):
  q_hidden_obs: Sequence[tfp.distributions.Distribution]
  p_hidden: Sequence[tfp.distributions.Distribution]
  hidden_samples: Sequence[jnp.ndarray]
  log_p_obs_hidden: jnp.ndarray
  is_male: jnp.ndarray  # indicates which elements of the batch correspond to
                        # male individuals


def build_causal_graph(train_data: pd.DataFrame, column_names: List[str],
                       inputs: jnp.ndarray):
  """Build the causal graph of the model."""
  make_multinomial = functools.partial(
      causal_network.MLPMultinomial.from_frame, hidden_shape=(100,))
  make_gaussian = functools.partial(
      causal_network.Gaussian, hidden_shape=(100,))

  # Construct the graphical model. Each random variable is represented by an
  # instance of the `Node` class, as discussed in that class's docstring.
  # The following nodes have no parents, and thus the distribution modules
  # will not be conditional on anything -- they simply represent priors.
  node_a = Node(MLPMultinomial.from_frame(train_data, 'sex'))
  node_c1 = Node(MLPMultinomial.from_frame(train_data, 'native-country'))
  node_c2 = Node(Gaussian('age', column_names.index('age')))

  # These are all hidden nodes, that do not correspond to any actual data in
  # pandas dataframe loaded previously. We therefore are permitted to control
  # the dimensionality of these nodes as we wish (with the `dim` argument).
  # The distribution module here should be interpreted as saying that we are
  # imposing a multi-modal prior (a mixture of Gaussians) on each latent
  # variable.
  node_hm = Node(causal_network.GaussianMixture('hm', 10, dim=2), hidden=True)
  node_hl = Node(causal_network.GaussianMixture('hl', 10, dim=2), hidden=True)
  node_hr1 = Node(
      causal_network.GaussianMixture('hr1', 10, dim=2), hidden=True)
  node_hr2 = Node(
      causal_network.GaussianMixture('hr2', 10, dim=2), hidden=True)
  node_hr3 = Node(
      causal_network.GaussianMixture('hr3', 10, dim=2), hidden=True)

  # The rest of the graph is now constructed; the order of construction is
  # important, so we can inform each node of its parents.
  # Note that in the paper we simply have one node called "R", but here it is
  # separated into three separate `Node` instances. This is necessary since
  # each node can only represent a single quantity in the dataframe.
  node_m = Node(
      make_multinomial(train_data, 'marital-status'),
      [node_a, node_hm, node_c1, node_c2])
  node_l = Node(
      make_gaussian('education-num', column_names.index('education-num')),
      [node_a, node_hl, node_c1, node_c2, node_m])
  node_r1 = Node(
      make_multinomial(train_data, 'occupation'),
      [node_a, node_c1, node_c2, node_m, node_l])
  node_r2 = Node(
      make_gaussian('hours-per-week', column_names.index('hours-per-week')),
      [node_a, node_c1, node_c2, node_m, node_l])
  node_r3 = Node(
      make_multinomial(train_data, 'workclass'),
      [node_a, node_c1, node_c2, node_m, node_l])
  node_y = Node(
      MLPMultinomial.from_frame(train_data, 'income'),
      [node_a, node_c1, node_c2, node_m, node_l, node_r1, node_r2, node_r3])

  # We now construct several (self-explanatory) collections of nodes. These
  # will be used at various points later in the code, and serve to provide
  # greater semantic interpretability.
  observable_nodes = (node_a, node_c1, node_c2, node_l, node_m, node_r1,
                      node_r2, node_r3, node_y)

  # The nodes on which each latent variable is conditionally dependent.
  # Note that Y is not in this list, since all of its dependencies are
  # included below, and further it does not depend directly on Hm.
  nodes_on_which_hm_depends = (node_a, node_c1, node_c2, node_m)
  nodes_on_which_hl_depends = (node_a, node_c1, node_c2, node_m, node_l)
  nodes_on_which_hr1_depends = (node_a, node_c1, node_c2, node_m, node_l,
                                node_r1)
  nodes_on_which_hr2_depends = (node_a, node_c1, node_c2, node_m, node_l,
                                node_r2)
  nodes_on_which_hr3_depends = (node_a, node_c1, node_c2, node_m, node_l,
                                node_r3)
  hidden_nodes = (node_hm, node_hl, node_hr1, node_hr2, node_hr3)

  # Function to create the distribution needed for variational inference. This
  # is the same for each latent variable.
  def make_q_x_obs_module(node):
    """Make a Variational module for the given hidden variable."""
    assert node.hidden
    return variational.Variational(
        common_layer_sizes=(20, 20), output_dim=node.dim)

  # For each latent variable, we first construct a Haiku module (using the
  # function above), and then connect it to the graph using the node's
  # value. As described in more detail in the documentation for `Node`,
  # these values represent actual observed data. Therefore we will later
  # be connecting these same modules to the graph in different ways in order
  # to perform fair inference.
  q_hm_obs_module = make_q_x_obs_module(node_hm)
  q_hl_obs_module = make_q_x_obs_module(node_hl)
  q_hr1_obs_module = make_q_x_obs_module(node_hr1)
  q_hr2_obs_module = make_q_x_obs_module(node_hr2)
  q_hr3_obs_module = make_q_x_obs_module(node_hr3)

  causal_network.populate(observable_nodes, inputs)

  q_hm_obs = q_hm_obs_module(
      *(node.observed_value for node in nodes_on_which_hm_depends))
  q_hl_obs = q_hl_obs_module(
      *(node.observed_value for node in nodes_on_which_hl_depends))
  q_hr1_obs = q_hr1_obs_module(
      *(node.observed_value for node in nodes_on_which_hr1_depends))
  q_hr2_obs = q_hr2_obs_module(
      *(node.observed_value for node in nodes_on_which_hr2_depends))
  q_hr3_obs = q_hr3_obs_module(
      *(node.observed_value for node in nodes_on_which_hr3_depends))
  q_hidden_obs = (q_hm_obs, q_hl_obs, q_hr1_obs, q_hr2_obs, q_hr3_obs)

  return observable_nodes, hidden_nodes, q_hidden_obs


def build_forward_fn(train_data: pd.DataFrame, column_names: List[str],
                     likelihood_multiplier: float):
  """Create the model's forward pass."""

  def forward_fn(inputs: jnp.ndarray) -> CausalNetOutput:
    """Forward pass."""
    observable_nodes, hidden_nodes, q_hidden = build_causal_graph(
        train_data, column_names, inputs)
    (node_hm, node_hl, node_hr1, node_hr2, node_hr3) = hidden_nodes
    (node_a, _, _, _, _, _, _, _, node_y) = observable_nodes

    # Log-likelihood function.
    def log_p_obs_h(hm_value, hl_value, hr1_value, hr2_value, hr3_value):
      """Compute log P(A, C, M, L, R, Y | H)."""
      # In order to create distributions like P(M | H_m, A, C), we need
      # the value of H_m that we've been provided as an argument, rather than
      # the value stored on H_m (which, in fact, will never be populated
      # since H_m is unobserved).
      # For compactness, we first construct the complete list of replacements.
      node_to_replacement = {
          node_hm: hm_value,
          node_hl: hl_value,
          node_hr1: hr1_value,
          node_hr2: hr2_value,
          node_hr3: hr3_value,
      }

      def log_prob_for_node(node):
        """Given a node, compute it's log probability for the given latents."""
        log_prob = jnp.squeeze(
            node.make_distribution(node_to_replacement).log_prob(
                node.observed_value))
        return log_prob

      # We apply the likelihood multiplier to all likelihood terms except that
      # for Y, the target. This is then added on separately in the line below.
      sum_no_y = likelihood_multiplier * sum(
          log_prob_for_node(node)
          for node in observable_nodes
          if node is not node_y)

      return sum_no_y + log_prob_for_node(node_y)

    q_hidden_obs = tuple(q_hidden)
    p_hidden = tuple(node.distribution for node in hidden_nodes)
    rnd_key = hk.next_rng_key()
    hidden_samples = tuple(
        q_hidden.sample(seed=rnd_key) for q_hidden in q_hidden_obs)
    log_p_obs_hidden = log_p_obs_h(*hidden_samples)

    # We need to split our batch of data into male and female parts.
    is_male = jnp.equal(node_a.observed_value[:, 1], 1)

    return CausalNetOutput(
        q_hidden_obs=q_hidden_obs,
        p_hidden=p_hidden,
        hidden_samples=hidden_samples,
        log_p_obs_hidden=log_p_obs_hidden,
        is_male=is_male)

  def fair_inference_fn(inputs: jnp.ndarray, batch_size: int,
                        num_prediction_samples: int):
    """Get the fair and unfair predictions for the given input."""
    observable_nodes, hidden_nodes, q_hidden_obs = build_causal_graph(
        train_data, column_names, inputs)
    (node_hm, node_hl, node_hr1, node_hr2, node_hr3) = hidden_nodes
    (node_a, node_c1, node_c2, node_l, node_m, node_r1, node_r2, node_r3,
     node_y) = observable_nodes
    (q_hm_obs, q_hl_obs, q_hr1_obs, q_hr2_obs, q_hr3_obs) = q_hidden_obs
    rnd_key = hk.next_rng_key()

    # *** FAIR INFERENCE ***

    # To predict Y in a fair sense:
    #    * Infer Hm given observations.
    #    * Infer M using inferred Hm, baseline A, real C
    #    * Infer L using inferred Hl, M, real A, C
    #    * Infer Y using inferred M, baseline A, real C
    # This is done by numerical integration, i.e. draw samples from
    # p_fair(Y | A, C, M, L).
    a_all_male = jnp.concatenate(
        (jnp.zeros((batch_size, 1)), jnp.ones((batch_size, 1))),
        axis=1)

    # Here we take a num_samples per observation. This results to
    # an array of shape:
    #     (num_samples, batch_size, hm_dim).
    # However, forward pass is easier by reshaping to:
    #     (num_samples * batch_size, hm_dim).
    hm_dim = 2
    def expanded_sample(distribution):
      return distribution.sample(
          num_prediction_samples, seed=rnd_key).reshape(
              (batch_size * num_prediction_samples, hm_dim))

    hm_pred_sample = expanded_sample(q_hm_obs)
    hl_pred_sample = expanded_sample(q_hl_obs)
    hr1_pred_sample = expanded_sample(q_hr1_obs)
    hr2_pred_sample = expanded_sample(q_hr2_obs)
    hr3_pred_sample = expanded_sample(q_hr3_obs)

    # The values of the observed nodes need to be tiled to match the dims
    # of the above hidden samples. The `expand` function achieves this.
    def expand(observed_value):
      return jnp.tile(observed_value, (num_prediction_samples, 1))

    expanded_a = expand(node_a.observed_value)
    expanded_a_baseline = expand(a_all_male)
    expanded_c1 = expand(node_c1.observed_value)
    expanded_c2 = expand(node_c2.observed_value)

    # For M, and all subsequent variables, we only generate one sample. This
    # is because we already have *many* samples from the latent variables, and
    # all we require is an independent sample from the distribution.
    m_pred_sample = node_m.make_distribution({
        node_a: expanded_a_baseline,
        node_hm: hm_pred_sample,
        node_c1: expanded_c1,
        node_c2: expanded_c2}).sample(seed=rnd_key)

    l_pred_sample = node_l.make_distribution({
        node_a: expanded_a,
        node_hl: hl_pred_sample,
        node_c1: expanded_c1,
        node_c2: expanded_c2,
        node_m: m_pred_sample}).sample(seed=rnd_key)

    r1_pred_sample = node_r1.make_distribution({
        node_a: expanded_a,
        node_hr1: hr1_pred_sample,
        node_c1: expanded_c1,
        node_c2: expanded_c2,
        node_m: m_pred_sample,
        node_l: l_pred_sample}).sample(seed=rnd_key)
    r2_pred_sample = node_r2.make_distribution({
        node_a: expanded_a,
        node_hr2: hr2_pred_sample,
        node_c1: expanded_c1,
        node_c2: expanded_c2,
        node_m: m_pred_sample,
        node_l: l_pred_sample}).sample(seed=rnd_key)
    r3_pred_sample = node_r3.make_distribution({
        node_a: expanded_a,
        node_hr3: hr3_pred_sample,
        node_c1: expanded_c1,
        node_c2: expanded_c2,
        node_m: m_pred_sample,
        node_l: l_pred_sample}).sample(seed=rnd_key)

    # Finally, we sample from the distribution for Y. Like above, we only
    # draw one sample per element in the array.
    y_pred_sample = node_y.make_distribution({
        node_a: expanded_a_baseline,
        # node_a: expanded_a,
        node_c1: expanded_c1,
        node_c2: expanded_c2,
        node_m: m_pred_sample,
        node_l: l_pred_sample,
        node_r1: r1_pred_sample,
        node_r2: r2_pred_sample,
        node_r3: r3_pred_sample}).sample(seed=rnd_key)

    # Reshape back to (num_samples, batch_size, y_dim), undoing the expanding
    # operation used for sampling.
    y_pred_sample = y_pred_sample.reshape(
        (num_prediction_samples, batch_size, -1))

    # Now form an array of shape (batch_size, y_dim) by taking an expectation
    # over the sample dimension. This represents the probability that the
    # result is in each class.
    y_pred_expectation = jnp.mean(y_pred_sample, axis=0)

    # Find out the predicted y, for later use in a confusion matrix.
    predicted_class_y_fair = utils.multinomial_class(y_pred_expectation)

    # *** NAIVE INFERENCE ***
    predicted_class_y_unfair = utils.multinomial_class(node_y.distribution)

    return predicted_class_y_fair, predicted_class_y_unfair

  return forward_fn, fair_inference_fn


def _loss_fn(
    forward_fn,
    beta: float,
    mmd_sample_size: int,
    constraint_multiplier: float,
    constraint_ratio: float,
    params: hk.Params,
    rng: jnp.ndarray,
    inputs: jnp.ndarray,
) -> jnp.ndarray:
  """Loss function definition."""
  outputs = forward_fn(params, rng, inputs)
  loss = _loss_klqp(outputs, beta)

  # if (constraint_ratio * constraint_multiplier) > 0:
  constraint_loss = 0.

  # Create constraint penalty and add to overall loss term.
  for distribution in outputs.q_hidden_obs:
    constraint_loss += (constraint_ratio * constraint_multiplier *
                        utils.mmd_loss(distribution,
                                       outputs.is_male,
                                       mmd_sample_size,
                                       rng))

  # Optimisation - don't do the computation if the multiplier is set to zero.
  loss += constraint_loss

  return loss


def _evaluate(
    fair_inference_fn,
    params: hk.Params,
    rng: jnp.ndarray,
    inputs: jnp.ndarray,
    batch_size: int,
    num_prediction_samples: int,
):
  """Perform evaluation of fair inference."""
  output = fair_inference_fn(params, rng, inputs,
                             batch_size, num_prediction_samples)

  return output


def _loss_klqp(outputs: CausalNetOutput, beta: float) -> jnp.ndarray:
  """Compute the loss on data wrt params."""

  expected_log_q_hidden_obs = sum(
      jnp.sum(q_hidden_obs.log_prob(hidden_sample), axis=1) for q_hidden_obs,
      hidden_sample in zip(outputs.q_hidden_obs, outputs.hidden_samples))

  assert expected_log_q_hidden_obs.ndim == 1

  # For log probabilities computed from distributions, we need to sum along
  # the last axis, which takes the product of distributions for
  # multi-dimensional hidden variables.
  log_p_hidden = sum(
      jnp.sum(p_hidden.log_prob(hidden_sample), axis=1) for p_hidden,
      hidden_sample in zip(outputs.p_hidden, outputs.hidden_samples))

  assert outputs.log_p_obs_hidden.ndim == 1

  kl_divergence = (
      beta * (expected_log_q_hidden_obs - log_p_hidden) -
      outputs.log_p_obs_hidden)
  return jnp.mean(kl_divergence)


class Updater:
  """A stateless abstraction around an init_fn/update_fn pair.

  This extracts some common boilerplate from the training loop.
  """

  def __init__(self, net_init, loss_fn, eval_fn,
               optimizer: optax.GradientTransformation,
               constraint_turn_on_step):
    self._net_init = net_init
    self._loss_fn = loss_fn
    self._eval_fn = eval_fn
    self._opt = optimizer
    self._constraint_turn_on_step = constraint_turn_on_step

  @functools.partial(jax.jit, static_argnums=0)
  def init(self, init_rng, data):
    """Initializes state of the updater."""
    params = self._net_init(init_rng, data)
    opt_state = self._opt.init(params)
    out = dict(
        step=np.array(0),
        rng=init_rng,
        opt_state=opt_state,
        params=params,
    )
    return out

  @functools.partial(jax.jit, static_argnums=0)
  def update(self, state: Mapping[str, Any], data: jnp.ndarray):
    """Updates the state using some data and returns metrics."""
    rng = state['rng']
    params = state['params']

    constraint_ratio = (state['step'] > self._constraint_turn_on_step).astype(
        float)
    loss, g = jax.value_and_grad(self._loss_fn, argnums=1)(
        constraint_ratio, params, rng, data)
    updates, opt_state = self._opt.update(g, state['opt_state'])
    params = optax.apply_updates(params, updates)

    new_state = {
        'step': state['step'] + 1,
        'rng': rng,
        'opt_state': opt_state,
        'params': params,
    }

    new_metrics = {
        'step': state['step'],
        'loss': loss,
    }
    return new_state, new_metrics

  @functools.partial(jax.jit, static_argnums=(0, 3, 4))
  def evaluate(self, state: Mapping[str, Any], inputs: jnp.ndarray,
               batch_size: int, num_prediction_samples: int):
    """Evaluate fair inference."""
    rng = state['rng']
    params = state['params']
    fair_pred, unfair_pred = self._eval_fn(params, rng, inputs, batch_size,
                                           num_prediction_samples)
    return fair_pred, unfair_pred


def main(_):
  flags_config = FLAGS.config
  # Create the dataset.
  train_data, test_data = adult.read_all_data(FLAGS.dataset_dir)
  column_names = list(train_data.columns)
  train_input = build_input(train_data, flags_config.batch_size,
                            flags_config.num_steps)

  # Set up the model, loss, and updater.
  forward_fn, fair_inference_fn = build_forward_fn(
      train_data, column_names, flags_config.likelihood_multiplier)
  forward_fn = hk.transform(forward_fn)
  fair_inference_fn = hk.transform(fair_inference_fn)

  loss_fn = functools.partial(_loss_fn, forward_fn.apply,
                              flags_config.beta,
                              flags_config.mmd_sample_size,
                              flags_config.constraint_multiplier)
  eval_fn = functools.partial(_evaluate, fair_inference_fn.apply)

  optimizer = optax.adam(flags_config.learning_rate)

  updater = Updater(forward_fn.init, loss_fn, eval_fn,
                    optimizer, flags_config.constraint_turn_on_step)

  # Initialize parameters.
  logging.info('Initializing parameters...')
  rng = jax.random.PRNGKey(42)
  train_data = next(train_input)
  state = updater.init(rng, train_data)

  # Training loop.
  logging.info('Starting train loop...')
  prev_time = time.time()
  for step in range(flags_config.num_steps):
    train_data = next(train_input)
    state, stats = updater.update(state, train_data)
    if step % LOG_EVERY == 0:
      steps_per_sec = LOG_EVERY / (time.time() - prev_time)
      prev_time = time.time()
      stats.update({'steps_per_sec': steps_per_sec})
      logging.info({k: float(v) for k, v in stats.items()})

  # Evaluate.
  logging.info('Starting evaluation...')
  test_input = build_input(test_data, flags_config.batch_size,
                           training_steps=0,
                           shuffle_size=0)

  predicted_test_y = []
  corrected_test_y = []
  while True:
    try:
      eval_data = next(test_input)
      # Now run the fair prediction; this projects the input to the latent space
      # and then performs sampling.
      predicted_class_y_fair, predicted_class_y_unfair = updater.evaluate(
          state, eval_data, flags_config.batch_size,
          flags_config.num_prediction_samples)
      predicted_test_y.append(predicted_class_y_unfair)
      corrected_test_y.append(predicted_class_y_fair)
      # logging.info('Completed evaluation step %d', step)
    except StopIteration:
      logging.info('Finished evaluation')
      break

  # Join together the predictions from each batch.
  test_y = np.concatenate(predicted_test_y, axis=0)
  tweaked_test_y = np.concatenate(corrected_test_y, axis=0)

  # Note the true values for computing accuracy and confusion matrices.
  y_true = test_data['income'].cat.codes
  # Make sure y_true is the same size
  y_true = y_true[:len(test_y)]

  test_accuracy = metrics.accuracy_score(y_true, test_y)
  tweaked_test_accuracy = metrics.accuracy_score(
      y_true, tweaked_test_y)

  # Print out accuracy and confusion matrices.
  logging.info('Accuracy (full model):                %f', test_accuracy)
  logging.info('Confusion matrix:')
  logging.info(metrics.confusion_matrix(y_true, test_y))
  logging.info('')

  logging.info('Accuracy (tweaked with baseline: Male): %f',
               tweaked_test_accuracy)
  logging.info('Confusion matrix:')
  logging.info(metrics.confusion_matrix(y_true, tweaked_test_y))

if __name__ == '__main__':
  app.run(main)
