################################################################################
# Copyright 2019 DeepMind Technologies Limited
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
################################################################################
"""Script to train CURL."""

import collections
import functools
from absl import logging

import numpy as np
from sklearn import neighbors
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from curl import model
from curl import utils

tfc = tf.compat.v1

# pylint: disable=g-long-lambda

MainOps = collections.namedtuple('MainOps', [
    'elbo', 'll', 'log_p_x', 'kl_y', 'kl_z', 'elbo_supervised', 'll_supervised',
    'log_p_x_supervised', 'kl_y_supervised', 'kl_z_supervised',
    'cat_probs', 'confusion', 'purity', 'latents'
])

DatasetTuple = collections.namedtuple('DatasetTuple', [
    'train_data', 'train_iter_for_clf', 'train_data_for_clf',
    'valid_iter', 'valid_data', 'test_iter', 'test_data', 'ds_info'
])


def compute_purity(confusion):
  return np.sum(np.max(confusion, axis=0)).astype(float) / np.sum(confusion)


def process_dataset(iterator,
                    ops_to_run,
                    sess,
                    feed_dict=None,
                    aggregation_ops=np.stack,
                    processing_ops=None):
  """Process a dataset by computing ops and accumulating batch by batch.

  Args:
    iterator: iterator through the dataset.
    ops_to_run: dict, tf ops to run as part of dataset processing.
    sess: tf.Session to use.
    feed_dict: dict, required placeholders.
    aggregation_ops: fn or dict of fns, aggregation op to apply for each op.
    processing_ops: fn or dict of fns, extra processing op to apply for each op.

  Returns:
    Results accumulated over dataset.
  """

  if not isinstance(ops_to_run, dict):
    raise TypeError('ops_to_run must be specified as a dict')

  if not isinstance(aggregation_ops, dict):
    aggregation_ops = {k: aggregation_ops for k in ops_to_run}
  if not isinstance(processing_ops, dict):
    processing_ops = {k: processing_ops for k in ops_to_run}

  out_results = collections.OrderedDict()
  sess.run(iterator.initializer)
  while True:
    # Iterate over the whole dataset and append the results to a per-key list.
    try:
      outs = sess.run(ops_to_run, feed_dict=feed_dict)
      for key, value in outs.items():
        out_results.setdefault(key, []).append(value)

    except tf.errors.OutOfRangeError:  # end of dataset iterator
      break

  # Aggregate and process results.
  for key, value in out_results.items():
    if aggregation_ops[key]:
      out_results[key] = aggregation_ops[key](value)
    if processing_ops[key]:
      out_results[key] = processing_ops[key](out_results[key], axis=0)

  return out_results


def get_data_sources(dataset, dataset_kwargs, batch_size, test_batch_size,
                     training_data_type, n_concurrent_classes, image_key,
                     label_key):
  """Create and return data sources for training, validation, and testing.

  Args:
    dataset: str, name of dataset ('mnist', 'omniglot', etc).
    dataset_kwargs: dict, kwargs used in tf dataset constructors.
    batch_size: int, batch size used for training.
    test_batch_size: int, batch size used for evaluation.
    training_data_type: str, how training data is seen ('iid', or 'sequential').
    n_concurrent_classes: int, # classes seen at a time (ignored for 'iid').
    image_key: str, name if image key in dataset.
    label_key: str, name of label key in dataset.

  Returns:
    A namedtuple containing all of the dataset iterators and batches.

  """

  # Load training data sources
  ds_train, ds_info = tfds.load(
      name=dataset,
      split=tfds.Split.TRAIN,
      with_info=True,
      as_dataset_kwargs={'shuffle_files': False},
      **dataset_kwargs)

  # Validate assumption that data is in [0, 255]
  assert ds_info.features[image_key].dtype == tf.uint8

  n_classes = ds_info.features[label_key].num_classes
  num_train_examples = ds_info.splits['train'].num_examples

  def preprocess_data(x):
    """Convert images from uint8 in [0, 255] to float in [0, 1]."""
    x[image_key] = tf.image.convert_image_dtype(x[image_key], tf.float32)
    return x

  if training_data_type == 'sequential':
    c = None  # The index of the class number, None for now and updated later
    if n_concurrent_classes == 1:
      filter_fn = lambda v: tf.equal(v[label_key], c)
    else:
      # Define the lowest and highest class number at each data period.
      assert n_classes % n_concurrent_classes == 0, (
          'Number of total classes must be divisible by '
          'number of concurrent classes')
      cmin = []
      cmax = []
      for i in range(int(n_classes / n_concurrent_classes)):
        for _ in range(n_concurrent_classes):
          cmin.append(i * n_concurrent_classes)
          cmax.append((i + 1) * n_concurrent_classes)

      filter_fn = lambda v: tf.logical_and(
          tf.greater_equal(v[label_key], cmin[c]), tf.less(
              v[label_key], cmax[c]))

    # Set up data sources/queues (one for each class).
    train_datasets = []
    train_iterators = []
    train_data = []

    full_ds = ds_train.repeat().shuffle(num_train_examples, seed=0)
    full_ds = full_ds.map(preprocess_data)
    for c in range(n_classes):
      filtered_ds = full_ds.filter(filter_fn).batch(
          batch_size, drop_remainder=True)
      train_datasets.append(filtered_ds)
      train_iterators.append(train_datasets[-1].make_one_shot_iterator())
      train_data.append(train_iterators[-1].get_next())

  else:  # not sequential
    full_ds = ds_train.repeat().shuffle(num_train_examples, seed=0)
    full_ds = full_ds.map(preprocess_data)
    train_datasets = full_ds.batch(batch_size, drop_remainder=True)
    train_data = train_datasets.make_one_shot_iterator().get_next()

  # Set up data source to get full training set for classifier training
  full_ds = ds_train.repeat(1).shuffle(num_train_examples, seed=0)
  full_ds = full_ds.map(preprocess_data)
  train_datasets_for_classifier = full_ds.batch(
      test_batch_size, drop_remainder=True)
  train_iter_for_classifier = (
      train_datasets_for_classifier.make_initializable_iterator())
  train_data_for_classifier = train_iter_for_classifier.get_next()

  # Load validation dataset.
  try:
    valid_dataset = tfds.load(
        name=dataset, split=tfds.Split.VALIDATION, **dataset_kwargs)
    num_valid_examples = ds_info.splits[tfds.Split.VALIDATION].num_examples
    assert (num_valid_examples %
            test_batch_size == 0), ('test_batch_size must be a divisor of %d' %
                                    num_valid_examples)
    valid_dataset = valid_dataset.repeat(1).batch(
        test_batch_size, drop_remainder=True)
    valid_dataset = valid_dataset.map(preprocess_data)
    valid_iter = valid_dataset.make_initializable_iterator()
    valid_data = valid_iter.get_next()
  except (KeyError, ValueError):
    logging.warning('No validation set!!')
    valid_iter = None
    valid_data = None

  # Load test dataset.
  test_dataset = tfds.load(
      name=dataset, split=tfds.Split.TEST, **dataset_kwargs)
  num_test_examples = ds_info.splits['test'].num_examples
  assert (num_test_examples %
          test_batch_size == 0), ('test_batch_size must be a divisor of %d' %
                                  num_test_examples)
  test_dataset = test_dataset.repeat(1).batch(
      test_batch_size, drop_remainder=True)
  test_dataset = test_dataset.map(preprocess_data)
  test_iter = test_dataset.make_initializable_iterator()
  test_data = test_iter.get_next()
  logging.info('Loaded %s data', dataset)

  return DatasetTuple(train_data, train_iter_for_classifier,
                      train_data_for_classifier, valid_iter, valid_data,
                      test_iter, test_data, ds_info)


def setup_training_and_eval_graphs(x, label, y, n_y, curl_model,
                                   classify_with_samples, is_training, name):
  """Set up the graph and return ops for training or evaluation.

  Args:
    x: tf placeholder for image.
    label: tf placeholder for ground truth label.
    y: tf placeholder for some self-supervised label/prediction.
    n_y: int, dimensionality of discrete latent variable y.
    curl_model: snt.AbstractModule representing the CURL model.
    classify_with_samples: bool, whether to *sample* latents for classification.
    is_training: bool, whether this graph is the training graph.
    name: str, graph name.

  Returns:
    A namedtuple with the required graph ops to perform training or evaluation.

  """
  # kl_y_supervised is -log q(y=y_true | x)
  (log_p_x, kl_y, kl_z, log_p_x_supervised, kl_y_supervised,
   kl_z_supervised) = curl_model.log_prob_elbo_components(x, y)

  ll = log_p_x - kl_y - kl_z
  elbo = -tf.reduce_mean(ll)

  # Supervised loss, either for SMGR, or adaptation to supervised benchmark.
  ll_supervised = log_p_x_supervised - kl_y_supervised - kl_z_supervised
  elbo_supervised = -tf.reduce_mean(ll_supervised)

  # Summaries
  kl_y = tf.reduce_mean(kl_y)
  kl_z = tf.reduce_mean(kl_z)
  log_p_x_supervised = tf.reduce_mean(log_p_x_supervised)
  kl_y_supervised = tf.reduce_mean(kl_y_supervised)
  kl_z_supervised = tf.reduce_mean(kl_z_supervised)

  # Evaluation.
  hiddens = curl_model.get_shared_rep(x, is_training=is_training)
  cat = curl_model.infer_cluster(hiddens)
  cat_probs = cat.probs

  confusion = tf.confusion_matrix(label, tf.argmax(cat_probs, axis=1),
                                  num_classes=n_y, name=name + '_confusion')
  purity = (tf.reduce_sum(tf.reduce_max(confusion, axis=0))
            / tf.reduce_sum(confusion))

  if classify_with_samples:
    latents = curl_model.infer_latent(
        hiddens=hiddens, y=tf.to_float(cat.sample())).sample()
  else:
    latents = curl_model.infer_latent(
        hiddens=hiddens, y=tf.to_float(cat.mode())).mean()

  return MainOps(elbo, ll, log_p_x, kl_y, kl_z, elbo_supervised, ll_supervised,
                 log_p_x_supervised, kl_y_supervised, kl_z_supervised,
                 cat_probs, confusion, purity, latents)


def get_generated_data(sess, gen_op, y_input, gen_buffer_size,
                       component_counts):
  """Get generated model data (in place of saving a model snapshot).

  Args:
    sess: tf.Session.
    gen_op: tf op representing a batch of generated data.
    y_input: tf placeholder for which mixture components to generate from.
    gen_buffer_size: int, number of data points to generate.
    component_counts: np.array, prior probabilities over components.

  Returns:
    A tuple of two numpy arrays
      The generated data
      The corresponding labels
  """

  batch_size, n_y = y_input.shape.as_list()

  # Sample based on the history of all components used.
  cluster_sample_probs = component_counts.astype(float)
  cluster_sample_probs = np.maximum(1e-12, cluster_sample_probs)
  cluster_sample_probs = cluster_sample_probs / np.sum(cluster_sample_probs)

  # Now generate the data based on the specified cluster prior.
  gen_buffer_images = []
  gen_buffer_labels = []
  for _ in range(gen_buffer_size):
    gen_label = np.random.choice(
        np.arange(n_y),
        size=(batch_size,),
        replace=True,
        p=cluster_sample_probs)
    y_gen_posterior_vals = np.zeros((batch_size, n_y))
    y_gen_posterior_vals[np.arange(batch_size), gen_label] = 1
    gen_image = sess.run(gen_op, feed_dict={y_input: y_gen_posterior_vals})
    gen_buffer_images.append(gen_image)
    gen_buffer_labels.append(gen_label)

  gen_buffer_images = np.vstack(gen_buffer_images)
  gen_buffer_labels = np.concatenate(gen_buffer_labels)

  return gen_buffer_images, gen_buffer_labels


def setup_dynamic_ops(n_y):
  """Set up ops to move / copy mixture component weights for dynamic expansion.

  Args:
    n_y: int, dimensionality of discrete latent variable y.

  Returns:
    A dict containing all of the ops required for dynamic updating.

  """
  # Set up graph ops to dynamically modify component params.
  graph = tf.get_default_graph()

  # 1) Ops to get and set latent encoder params (entire tensors)
  latent_enc_tensors = {}
  for k in range(n_y):
    latent_enc_tensors['latent_w_' + str(k)] = graph.get_tensor_by_name(
        'latent_encoder/mlp_latent_encoder_{}/w:0'.format(k))
    latent_enc_tensors['latent_b_' + str(k)] = graph.get_tensor_by_name(
        'latent_encoder/mlp_latent_encoder_{}/b:0'.format(k))

  latent_enc_assign_ops = {}
  latent_enc_phs = {}
  for key, tensor in latent_enc_tensors.items():
    latent_enc_phs[key] = tfc.placeholder(tensor.dtype, tensor.shape)
    latent_enc_assign_ops[key] = tf.assign(tensor, latent_enc_phs[key])

  # 2) Ops to get and set cluster encoder params (columns of a tensor)
  # We will be copying column ind_from to column ind_to.
  cluster_w = graph.get_tensor_by_name(
      'cluster_encoder/mlp_cluster_encoder_final/w:0')
  cluster_b = graph.get_tensor_by_name(
      'cluster_encoder/mlp_cluster_encoder_final/b:0')

  ind_from = tfc.placeholder(dtype=tf.int32)
  ind_to = tfc.placeholder(dtype=tf.int32)

  # Determine indices of cluster encoder weights and biases to be updated
  w_indices = tf.transpose(
      tf.stack([
          tf.range(cluster_w.shape[0], dtype=tf.int32),
          ind_to * tf.ones(shape=(cluster_w.shape[0],), dtype=tf.int32)
      ]))
  b_indices = ind_to
  # Determine updates themselves
  cluster_w_updates = tf.squeeze(
      tf.slice(cluster_w, begin=(0, ind_from), size=(cluster_w.shape[0], 1)))
  cluster_b_updates = cluster_b[ind_from]
  # Create update ops
  cluster_w_update_op = tf.scatter_nd_update(cluster_w, w_indices,
                                             cluster_w_updates)
  cluster_b_update_op = tf.scatter_update(cluster_b, b_indices,
                                          cluster_b_updates)

  # 3) Ops to get and set latent prior params (columns of a tensor)
  # We will be copying column ind_from to column ind_to.
  latent_prior_mu_w = graph.get_tensor_by_name(
      'latent_decoder/latent_prior_mu/w:0')
  latent_prior_sigma_w = graph.get_tensor_by_name(
      'latent_decoder/latent_prior_sigma/w:0')

  mu_indices = tf.transpose(
      tf.stack([
          ind_to * tf.ones(shape=(latent_prior_mu_w.shape[1],), dtype=tf.int32),
          tf.range(latent_prior_mu_w.shape[1], dtype=tf.int32)
      ]))
  mu_updates = tf.squeeze(
      tf.slice(
          latent_prior_mu_w,
          begin=(ind_from, 0),
          size=(1, latent_prior_mu_w.shape[1])))
  mu_update_op = tf.scatter_nd_update(latent_prior_mu_w, mu_indices, mu_updates)
  sigma_indices = tf.transpose(
      tf.stack([
          ind_to *
          tf.ones(shape=(latent_prior_sigma_w.shape[1],), dtype=tf.int32),
          tf.range(latent_prior_sigma_w.shape[1], dtype=tf.int32)
      ]))
  sigma_updates = tf.squeeze(
      tf.slice(
          latent_prior_sigma_w,
          begin=(ind_from, 0),
          size=(1, latent_prior_sigma_w.shape[1])))
  sigma_update_op = tf.scatter_nd_update(latent_prior_sigma_w, sigma_indices,
                                         sigma_updates)

  dynamic_ops = {
      'ind_from_ph': ind_from,
      'ind_to_ph': ind_to,
      'latent_enc_tensors': latent_enc_tensors,
      'latent_enc_assign_ops': latent_enc_assign_ops,
      'latent_enc_phs': latent_enc_phs,
      'cluster_w_update_op': cluster_w_update_op,
      'cluster_b_update_op': cluster_b_update_op,
      'mu_update_op': mu_update_op,
      'sigma_update_op': sigma_update_op
  }

  return dynamic_ops


def copy_component_params(ind_from, ind_to, sess, ind_from_ph, ind_to_ph,
                          latent_enc_tensors, latent_enc_assign_ops,
                          latent_enc_phs,
                          cluster_w_update_op, cluster_b_update_op,
                          mu_update_op, sigma_update_op):
  """Copy parameters from component i to component j.

  Args:
    ind_from: int, component index to copy from.
    ind_to: int, component index to copy to.
    sess: tf.Session.
    ind_from_ph: tf placeholder for component to copy from.
    ind_to_ph: tf placeholder for component to copy to.
    latent_enc_tensors: dict, tensors in the latent posterior encoder.
    latent_enc_assign_ops: dict, assignment ops for latent posterior encoder.
    latent_enc_phs: dict, placeholders for assignment ops.
    cluster_w_update_op: op for updating weights of cluster encoder.
    cluster_b_update_op: op for updating biased of cluster encoder.
    mu_update_op: op for updating mu weights of latent prior.
    sigma_update_op: op for updating sigma weights of latent prior.

  """
  update_ops = []
  feed_dict = {}
  # Copy for latent encoder.
  new_w_val, new_b_val = sess.run([
      latent_enc_tensors['latent_w_' + str(ind_from)],
      latent_enc_tensors['latent_b_' + str(ind_from)]
  ])
  update_ops.extend([
      latent_enc_assign_ops['latent_w_' + str(ind_to)],
      latent_enc_assign_ops['latent_b_' + str(ind_to)]
  ])
  feed_dict.update({
      latent_enc_phs['latent_w_' + str(ind_to)]: new_w_val,
      latent_enc_phs['latent_b_' + str(ind_to)]: new_b_val
  })

  # Copy for cluster encoder softmax.
  update_ops.extend([cluster_w_update_op, cluster_b_update_op])
  feed_dict.update({ind_from_ph: ind_from, ind_to_ph: ind_to})

  # Copy for latent prior.
  update_ops.extend([mu_update_op, sigma_update_op])
  feed_dict.update({ind_from_ph: ind_from, ind_to_ph: ind_to})
  sess.run(update_ops, feed_dict)


def run_training(
    dataset,
    training_data_type,
    n_concurrent_classes,
    blend_classes,
    train_supervised,
    n_steps,
    random_seed,
    lr_init,
    lr_factor,
    lr_schedule,
    output_type,
    n_y,
    n_y_active,
    n_z,
    encoder_kwargs,
    decoder_kwargs,
    dynamic_expansion,
    ll_thresh,
    classify_with_samples,
    report_interval,
    knn_values,
    gen_replay_type,
    use_supervised_replay):
  """Run training script.

  Args:
    dataset: str, name of the dataset.
    training_data_type: str, type of training run ('iid' or 'sequential').
    n_concurrent_classes: int, # of classes seen at a time (ignored for 'iid').
    blend_classes: bool, whether to blend in samples from the next class.
    train_supervised: bool, whether to use supervision during training.
    n_steps: int, number of total training steps.
    random_seed: int, seed for tf and numpy RNG.
    lr_init: float, initial learning rate.
    lr_factor: float, learning rate decay factor.
    lr_schedule: float, epochs at which the decay should be applied.
    output_type: str, output distribution (currently only 'bernoulli').
    n_y: int, maximum possible dimensionality of discrete latent variable y.
    n_y_active: int, starting dimensionality of discrete latent variable y.
    n_z: int, dimensionality of continuous latent variable z.
    encoder_kwargs: dict, parameters to specify encoder.
    decoder_kwargs: dict, parameters to specify decoder.
    dynamic_expansion: bool, whether to perform dynamic expansion.
    ll_thresh: float, log-likelihood threshold below which to keep poor samples.
    classify_with_samples: bool, whether to sample latents when classifying.
    report_interval: int, number of steps after which to evaluate and report.
    knn_values: list of ints, k values for different k-NN classifiers to run
    (values of 3, 5, and 10 were used in different parts of the paper).
    gen_replay_type: str, 'fixed', 'dynamic', or None.
    use_supervised_replay: str, whether to use supervised replay (aka 'SMGR').
  """

  # Set tf random seed.
  tfc.set_random_seed(random_seed)
  np.set_printoptions(precision=2, suppress=True)

  # First set up the data source(s) and get dataset info.
  if dataset == 'mnist':
    batch_size = 100
    test_batch_size = 1000
    dataset_kwargs = {}
    image_key = 'image'
    label_key = 'label'
  elif dataset == 'omniglot':
    batch_size = 15
    test_batch_size = 1318
    dataset_kwargs = {}
    image_key = 'image'
    label_key = 'alphabet'
  else:
    raise NotImplementedError

  dataset_ops = get_data_sources(dataset, dataset_kwargs, batch_size,
                                 test_batch_size, training_data_type,
                                 n_concurrent_classes, image_key, label_key)
  train_data = dataset_ops.train_data
  train_data_for_clf = dataset_ops.train_data_for_clf
  valid_data = dataset_ops.valid_data
  test_data = dataset_ops.test_data

  output_shape = dataset_ops.ds_info.features[image_key].shape
  n_x = np.prod(output_shape)
  n_classes = dataset_ops.ds_info.features[label_key].num_classes
  num_train_examples = dataset_ops.ds_info.splits['train'].num_examples

  # Check that the number of classes is compatible with the training scenario
  assert n_classes % n_concurrent_classes == 0
  assert n_steps % (n_classes / n_concurrent_classes) == 0

  # Set specific params depending on the type of gen replay
  if gen_replay_type == 'fixed':
    data_period = data_period = int(n_steps /
                                    (n_classes / n_concurrent_classes))
    gen_every_n = 2  # Blend in a gen replay batch every 2 steps
    gen_refresh_period = data_period  # How often to refresh the batches of
    # generated data (equivalent to snapshotting a generative model)
    gen_refresh_on_expansion = False  # Don't refresh on dyn expansion
  elif gen_replay_type == 'dynamic':
    gen_every_n = 2  # Blend in a gen replay batch every 2 steps
    gen_refresh_period = 1e8  # Never refresh generated data periodically
    gen_refresh_on_expansion = True  # Refresh on dyn expansion instead
  elif gen_replay_type is None:
    gen_every_n = 0  # Don't use any gen replay batches
    gen_refresh_period = 1e8  # Never refresh generated data periodically
    gen_refresh_on_expansion = False  # Don't refresh on dyn expansion
  else:
    raise NotImplementedError

  max_gen_batches = 5000  # Max num of gen batches (proxy for storing a model)

  # Set dynamic expansion parameters
  exp_wait_steps = 100  # Steps to wait after expansion before eligible again
  exp_burn_in = 100  # Steps to wait at start of learning before eligible
  exp_buffer_size = 100  # Size of the buffer of poorly explained data
  num_buffer_train_steps = 10  # Num steps to train component on buffer

  # Define a global tf variable for the number of active components.
  n_y_active_np = n_y_active
  n_y_active = tfc.get_variable(
      initializer=tf.constant(n_y_active_np, dtype=tf.int32),
      trainable=False,
      name='n_y_active',
      dtype=tf.int32)

  logging.info('Starting CURL script on %s data.', dataset)

  # Set up placeholders for training.

  x_train_raw = tfc.placeholder(
      dtype=tf.float32, shape=(batch_size,) + output_shape)
  label_train = tfc.placeholder(dtype=tf.int32, shape=(batch_size,))

  def binarize_fn(x):
    """Binarize a Bernoulli by rounding the probabilities.

    Args:
      x: tf tensor, input image.

    Returns:
      A tf tensor with the binarized image
    """
    return tf.cast(tf.greater(x, 0.5 * tf.ones_like(x)), tf.float32)

  if dataset == 'mnist':
    x_train = binarize_fn(x_train_raw)
    x_valid = binarize_fn(valid_data[image_key]) if valid_data else None
    x_test = binarize_fn(test_data[image_key])
    x_train_for_clf = binarize_fn(train_data_for_clf[image_key])
  elif 'cifar' in dataset or dataset == 'omniglot':
    x_train = x_train_raw
    x_valid = valid_data[image_key] if valid_data else None
    x_test = test_data[image_key]
    x_train_for_clf = train_data_for_clf[image_key]
  else:
    raise ValueError('Unknown dataset {}'.format(dataset))

  label_valid = valid_data[label_key] if valid_data else None
  label_test = test_data[label_key]

  # Set up CURL modules.
  shared_encoder = model.SharedEncoder(name='shared_encoder', **encoder_kwargs)
  latent_encoder = functools.partial(model.latent_encoder_fn, n_y=n_y, n_z=n_z)
  latent_encoder = snt.Module(latent_encoder, name='latent_encoder')
  latent_decoder = functools.partial(model.latent_decoder_fn, n_z=n_z)
  latent_decoder = snt.Module(latent_decoder, name='latent_decoder')
  cluster_encoder = functools.partial(
      model.cluster_encoder_fn, n_y_active=n_y_active, n_y=n_y)
  cluster_encoder = snt.Module(cluster_encoder, name='cluster_encoder')
  data_decoder = functools.partial(
      model.data_decoder_fn,
      output_type=output_type,
      output_shape=output_shape,
      n_x=n_x,
      n_y=n_y,
      **decoder_kwargs)
  data_decoder = snt.Module(data_decoder, name='data_decoder')

  # Uniform prior over y.
  prior_train_probs = utils.construct_prior_probs(batch_size, n_y, n_y_active)
  prior_train = snt.Module(
      lambda: tfp.distributions.OneHotCategorical(probs=prior_train_probs),
      name='prior_unconditional_train')
  prior_test_probs = utils.construct_prior_probs(test_batch_size, n_y,
                                                 n_y_active)
  prior_test = snt.Module(
      lambda: tfp.distributions.OneHotCategorical(probs=prior_test_probs),
      name='prior_unconditional_test')

  model_train = model.Curl(
      prior_train,
      latent_decoder,
      data_decoder,
      shared_encoder,
      cluster_encoder,
      latent_encoder,
      n_y_active,
      is_training=True,
      name='curl_train')
  model_eval = model.Curl(
      prior_test,
      latent_decoder,
      data_decoder,
      shared_encoder,
      cluster_encoder,
      latent_encoder,
      n_y_active,
      is_training=False,
      name='curl_test')

  # Set up training graph
  y_train = label_train if train_supervised else None
  y_valid = label_valid if train_supervised else None
  y_test = label_test if train_supervised else None

  train_ops = setup_training_and_eval_graphs(
      x_train,
      label_train,
      y_train,
      n_y,
      model_train,
      classify_with_samples,
      is_training=True,
      name='train')

  hiddens_for_clf = model_eval.get_shared_rep(x_train_for_clf,
                                              is_training=False)
  cat_for_clf = model_eval.infer_cluster(hiddens_for_clf)

  if classify_with_samples:
    latents_for_clf = model_eval.infer_latent(
        hiddens=hiddens_for_clf, y=tf.to_float(cat_for_clf.sample())).sample()
  else:
    latents_for_clf = model_eval.infer_latent(
        hiddens=hiddens_for_clf, y=tf.to_float(cat_for_clf.mode())).mean()

  # Set up validation graph
  if valid_data is not None:
    valid_ops = setup_training_and_eval_graphs(
        x_valid,
        label_valid,
        y_valid,
        n_y,
        model_eval,
        classify_with_samples,
        is_training=False,
        name='valid')

  # Set up test graph
  test_ops = setup_training_and_eval_graphs(
      x_test,
      label_test,
      y_test,
      n_y,
      model_eval,
      classify_with_samples,
      is_training=False,
      name='test')

  # Set up optimizer (with scheduler).
  global_step = tf.train.get_or_create_global_step()
  lr_schedule = [
      tf.cast(el * num_train_examples / batch_size, tf.int64)
      for el in lr_schedule
  ]
  num_schedule_steps = tf.reduce_sum(
      tf.cast(global_step >= lr_schedule, tf.float32))
  lr = float(lr_init) * float(lr_factor)**num_schedule_steps
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step = optimizer.minimize(train_ops.elbo)
    train_step_supervised = optimizer.minimize(train_ops.elbo_supervised)

    # For dynamic expansion, we want to train only new-component-related params
    cat_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        'cluster_encoder/mlp_cluster_encoder_final')
    component_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        'latent_encoder/mlp_latent_encoder_*')
    prior_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        'latent_decoder/latent_prior*')

    train_step_expansion = optimizer.minimize(
        train_ops.elbo_supervised,
        var_list=cat_params+component_params+prior_params)

  # Set up ops for generative replay
  if gen_every_n > 0:
    # How many generative batches will we use each period?
    gen_buffer_size = min(
        int(gen_refresh_period / gen_every_n), max_gen_batches)

    # Class each sample should be drawn from (default to uniform prior)
    y_gen = tfp.distributions.OneHotCategorical(
        probs=np.ones((batch_size, n_y)) / n_y,
        dtype=tf.float32,
        name='extra_train_classes').sample()

    gen_samples = model_train.sample(y=y_gen, mean=True)
    if dataset == 'mnist' or dataset == 'omniglot':
      gen_samples = binarize_fn(gen_samples)

  # Set up ops to dynamically modify parameters (for dynamic expansion)
  dynamic_ops = setup_dynamic_ops(n_y)

  logging.info('Created computation graph.')

  n_steps_per_class = n_steps / n_classes  # pylint: disable=invalid-name

  cumulative_component_counts = np.array([0] * n_y).astype(float)
  recent_component_counts = np.array([0] * n_y).astype(float)

  gen_buffer_ind = 0

  # Buffer of poorly explained data (if we're doing dynamic expansion).
  poor_data_buffer = []
  poor_data_labels = []
  all_full_poor_data_buffers = []
  all_full_poor_data_labels = []
  has_expanded = False
  steps_since_expansion = 0
  gen_buffer_ind = 0
  eligible_for_expansion = False  # Flag to ensure we wait a bit after expansion

  # Set up basic ops to run and quantities to log.
  ops_to_run = {
      'train_ELBO': train_ops.elbo,
      'train_log_p_x': train_ops.log_p_x,
      'train_kl_y': train_ops.kl_y,
      'train_kl_z': train_ops.kl_z,
      'train_ll': train_ops.ll,
      'train_batch_purity': train_ops.purity,
      'train_probs': train_ops.cat_probs,
      'n_y_active': n_y_active
  }
  if valid_data is not None:
    valid_ops_to_run = {
        'valid_ELBO': valid_ops.elbo,
        'valid_kl_y': valid_ops.kl_y,
        'valid_kl_z': valid_ops.kl_z,
        'valid_confusion': valid_ops.confusion
    }
  else:
    valid_ops_to_run = {}
  test_ops_to_run = {
      'test_ELBO': test_ops.elbo,
      'test_kl_y': test_ops.kl_y,
      'test_kl_z': test_ops.kl_z,
      'test_confusion': test_ops.confusion
  }
  to_log = ['train_batch_purity']
  to_log_eval = ['test_purity', 'test_ELBO', 'test_kl_y', 'test_kl_z']
  if valid_data is not None:
    to_log_eval += ['valid_ELBO', 'valid_purity']

  if train_supervised:
    # Track supervised losses, train on supervised loss.
    ops_to_run.update({
        'train_ELBO_supervised': train_ops.elbo_supervised,
        'train_log_p_x_supervised': train_ops.log_p_x_supervised,
        'train_kl_y_supervised': train_ops.kl_y_supervised,
        'train_kl_z_supervised': train_ops.kl_z_supervised,
        'train_ll_supervised': train_ops.ll_supervised
    })
    default_train_step = train_step_supervised
    to_log += [
        'train_ELBO_supervised', 'train_log_p_x_supervised',
        'train_kl_y_supervised', 'train_kl_z_supervised'
    ]
  else:
    # Track unsupervised losses, train on unsupervised loss.
    ops_to_run.update({
        'train_ELBO': train_ops.elbo,
        'train_kl_y': train_ops.kl_y,
        'train_kl_z': train_ops.kl_z,
        'train_ll': train_ops.ll
    })
    default_train_step = train_step
    to_log += ['train_ELBO', 'train_kl_y', 'train_kl_z']

  with tf.train.SingularMonitoredSession() as sess:

    for step in range(n_steps):
      feed_dict = {}

      # Use the default training loss, but vary it each step depending on the
      # training scenario (eg. for supervised gen replay, we alternate losses)
      ops_to_run['train_step'] = default_train_step

      ### 1) PERIODICALLY TAKE SNAPSHOTS FOR GENERATIVE REPLAY ###
      if (gen_refresh_period and step % gen_refresh_period == 0 and
          gen_every_n > 0):

        # First, increment cumulative count and reset recent probs count.
        cumulative_component_counts += recent_component_counts
        recent_component_counts = np.zeros(n_y)

        # Generate enough samples for the rest of the next period
        # (Functionally equivalent to storing and sampling from the model).
        gen_buffer_images, gen_buffer_labels = get_generated_data(
            sess=sess,
            gen_op=gen_samples,
            y_input=y_gen,
            gen_buffer_size=gen_buffer_size,
            component_counts=cumulative_component_counts)

      ### 2) DECIDE WHICH DATA SOURCE TO USE (GENERATIVE OR REAL DATA) ###
      periodic_refresh_started = (
          gen_refresh_period and step >= gen_refresh_period)
      refresh_on_expansion_started = (gen_refresh_on_expansion and has_expanded)
      if ((periodic_refresh_started or refresh_on_expansion_started) and
          gen_every_n > 0 and step % gen_every_n == 1):
        # Use generated data for the training batch
        used_real_data = False

        s = gen_buffer_ind * batch_size
        e = (gen_buffer_ind + 1) * batch_size

        gen_data_array = {
            'image': gen_buffer_images[s:e],
            'label': gen_buffer_labels[s:e]
        }
        gen_buffer_ind = (gen_buffer_ind + 1) % gen_buffer_size

        # Feed it as x_train because it's already reshaped and binarized.
        feed_dict.update({
            x_train: gen_data_array['image'],
            label_train: gen_data_array['label']
        })

        if use_supervised_replay:
          # Convert label to one-hot before feeding in.
          gen_label_onehot = np.eye(n_y)[gen_data_array['label']]
          feed_dict.update({model_train.y_label: gen_label_onehot})
          ops_to_run['train_step'] = train_step_supervised

      else:
        # Else use the standard training data sources.
        used_real_data = True

        # Select appropriate data source for iid or sequential setup.
        if training_data_type == 'sequential':
          current_data_period = int(
              min(step / n_steps_per_class, len(train_data) - 1))

          # If training supervised, set n_y_active directly based on how many
          # classes have been seen
          if train_supervised:
            assert not dynamic_expansion
            n_y_active_np = n_concurrent_classes * (
                current_data_period // n_concurrent_classes +1)
            n_y_active.load(n_y_active_np, sess)

          train_data_array = sess.run(train_data[current_data_period])

          # If we are blending classes, figure out where we are in the data
          # period and add some fraction of other samples.
          if blend_classes:
            # If in the first quarter, blend in examples from the previous class
            if (step % n_steps_per_class < n_steps_per_class / 4 and
                current_data_period > 0):
              other_train_data_array = sess.run(
                  train_data[current_data_period - 1])

              num_other = int(
                  (n_steps_per_class / 2 - 2 *
                   (step % n_steps_per_class)) * batch_size / n_steps_per_class)
              other_inds = np.random.permutation(batch_size)[:num_other]

              train_data_array[image_key][:num_other] = other_train_data_array[
                  image_key][other_inds]
              train_data_array[label_key][:num_other] = other_train_data_array[
                  label_key][other_inds]

            # If in the last quarter, blend in examples from the next class
            elif (step % n_steps_per_class > 3 * n_steps_per_class / 4 and
                  current_data_period < n_classes - 1):
              other_train_data_array = sess.run(train_data[current_data_period +
                                                           1])

              num_other = int(
                  (2 * (step % n_steps_per_class) - 3 * n_steps_per_class / 2) *
                  batch_size / n_steps_per_class)
              other_inds = np.random.permutation(batch_size)[:num_other]

              train_data_array[image_key][:num_other] = other_train_data_array[
                  image_key][other_inds]
              train_data_array['label'][:num_other] = other_train_data_array[
                  label_key][other_inds]

            # Otherwise, just use the current class

        else:
          train_data_array = sess.run(train_data)

        feed_dict.update({
            x_train_raw: train_data_array[image_key],
            label_train: train_data_array[label_key]
        })

      ### 3) PERFORM A GRADIENT STEP ###
      results = sess.run(ops_to_run, feed_dict=feed_dict)
      del results['train_step']

      ### 4) COMPUTE ADDITIONAL DIAGNOSTIC OPS ON VALIDATION/TEST SETS. ###
      if (step+1) % report_interval == 0:
        if valid_data is not None:
          logging.info('Evaluating on validation and test set!')
          proc_ops = {
              k: (np.sum if 'confusion' in k
                  else np.mean) for k in valid_ops_to_run
          }
          results.update(
              process_dataset(
                  dataset_ops.valid_iter,
                  valid_ops_to_run,
                  sess,
                  feed_dict=feed_dict,
                  processing_ops=proc_ops))
          results['valid_purity'] = compute_purity(results['valid_confusion'])
        else:
          logging.info('Evaluating on test set!')
          proc_ops = {
              k: (np.sum if 'confusion' in k
                  else np.mean) for k in test_ops_to_run
          }
        results.update(process_dataset(dataset_ops.test_iter,
                                       test_ops_to_run,
                                       sess,
                                       feed_dict=feed_dict,
                                       processing_ops=proc_ops))
        results['test_purity'] = compute_purity(results['test_confusion'])
        curr_to_log = to_log + to_log_eval
      else:
        curr_to_log = list(to_log)  # copy to prevent in-place modifications

      ### 5) DYNAMIC EXPANSION ###
      if dynamic_expansion and used_real_data:
        # If we're doing dynamic expansion and below max capacity then add
        # poorly defined data points to a buffer.

        # First check whether the model is eligible for expansion (the model
        # becomes ineligible for a fixed time after each expansion, and when
        # it has hit max capacity).
        if (steps_since_expansion >= exp_wait_steps and step >= exp_burn_in and
            n_y_active_np < n_y):
          eligible_for_expansion = True

        steps_since_expansion += 1

        if eligible_for_expansion:
          # Add poorly explained data samples to a buffer.
          poor_inds = results['train_ll'] < ll_thresh
          poor_data_buffer.extend(feed_dict[x_train_raw][poor_inds])
          poor_data_labels.extend(feed_dict[label_train][poor_inds])

          n_poor_data = len(poor_data_buffer)

          # If buffer is big enough, then add a new component and train just the
          # new component with several steps of gradient descent.
          # (We just feed in a onehot cluster vector to indicate which
          # component).
          if n_poor_data >= exp_buffer_size:
            # Dump the buffers so we can log them.
            all_full_poor_data_buffers.append(poor_data_buffer)
            all_full_poor_data_labels.append(poor_data_labels)

            # Take a new generative snapshot if specified.
            if gen_refresh_on_expansion and gen_every_n > 0:
              # Increment cumulative count and reset recent probs count.
              cumulative_component_counts += recent_component_counts
              recent_component_counts = np.zeros(n_y)

              gen_buffer_images, gen_buffer_labels = get_generated_data(
                  sess=sess,
                  gen_op=gen_samples,
                  y_input=y_gen,
                  gen_buffer_size=gen_buffer_size,
                  component_counts=cumulative_component_counts)

            # Cull to a multiple of batch_size (keep the later data samples).
            n_poor_batches = int(n_poor_data / batch_size)
            poor_data_buffer = poor_data_buffer[-(n_poor_batches * batch_size):]
            poor_data_labels = poor_data_labels[-(n_poor_batches * batch_size):]

            # Find most probable component (on poor batch).
            poor_cprobs = []
            for bs in range(n_poor_batches):
              poor_cprobs.append(
                  sess.run(
                      train_ops.cat_probs,
                      feed_dict={
                          x_train_raw:
                              poor_data_buffer[bs * batch_size:(bs + 1) *
                                               batch_size]
                      }))
            best_cluster = np.argmax(np.sum(np.vstack(poor_cprobs), axis=0))

            # Initialize parameters of the new component from most prob
            # existing.
            new_cluster = n_y_active_np

            copy_component_params(best_cluster, new_cluster, sess,
                                  **dynamic_ops)

            # Increment mixture component count n_y_active.
            n_y_active_np += 1
            n_y_active.load(n_y_active_np, sess)

            # Perform a number of steps of gradient descent on the data buffer,
            # training only the new component (supervised loss).
            for _ in range(num_buffer_train_steps):
              for bs in range(n_poor_batches):
                x_batch = poor_data_buffer[bs * batch_size:(bs + 1) *
                                           batch_size]
                label_batch = [new_cluster] * batch_size
                label_onehot_batch = np.eye(n_y)[label_batch]
                _ = sess.run(
                    train_step_expansion,
                    feed_dict={
                        x_train_raw: x_batch,
                        model_train.y_label: label_onehot_batch
                    })

            # Empty the buffer.
            poor_data_buffer = []
            poor_data_labels = []

            # Reset the threshold flag so we have a burn in before the next
            # component.
            eligible_for_expansion = False
            has_expanded = True
            steps_since_expansion = 0

      # Accumulate counts.
      if used_real_data:
        train_cat_probs_vals = results['train_probs']
        recent_component_counts += np.sum(
            train_cat_probs_vals, axis=0).astype(float)

      ### 6) LOGGING AND EVALUATION ###
      cleanup_for_print = lambda x: ', {}: %.{}f'.format(
          x.capitalize().replace('_', ' '), 3)
      log_str = 'Iteration %d'
      log_str += ''.join([cleanup_for_print(el) for el in curr_to_log])
      log_str += ' n_active: %d'
      logging.info(
          log_str,
          *([step] + [results[el] for el in curr_to_log] + [n_y_active_np]))

      # Periodically perform evaluation
      if (step + 1) % report_interval == 0:

        # Report test purity and related measures
        logging.info(
            'Iteration %d, Test purity: %.3f, Test ELBO: %.3f, Test '
            'KLy: %.3f, Test KLz: %.3f', step, results['test_purity'],
            results['test_ELBO'], results['test_kl_y'], results['test_kl_z'])
        # Flush data only once in a while to allow buffering of data for more
        # efficient writes.
        results['all_full_poor_data_buffers'] = all_full_poor_data_buffers
        results['all_full_poor_data_labels'] = all_full_poor_data_labels
        logging.info('Also training a classifier in latent space')

        # Perform knn classification from latents, to evaluate discriminability.

        # Get and encode training and test datasets.
        clf_train_vals = process_dataset(
            dataset_ops.train_iter_for_clf, {
                'latents': latents_for_clf,
                'labels': train_data_for_clf[label_key]
            },
            sess,
            feed_dict,
            aggregation_ops=np.concatenate)
        clf_test_vals = process_dataset(
            dataset_ops.test_iter, {
                'latents': test_ops.latents,
                'labels': test_data[label_key]
            },
            sess,
            aggregation_ops=np.concatenate)

        # Perform knn classification.
        knn_models = []
        for nval in knn_values:
          # Fit training dataset.
          clf = neighbors.KNeighborsClassifier(n_neighbors=nval)
          clf.fit(clf_train_vals['latents'], clf_train_vals['labels'])
          knn_models.append(clf)

          results['train_' + str(nval) + 'nn_acc'] = clf.score(
              clf_train_vals['latents'], clf_train_vals['labels'])

          # Get test performance.
          results['test_' + str(nval) + 'nn_acc'] = clf.score(
              clf_test_vals['latents'], clf_test_vals['labels'])

          logging.info(
              'Iteration %d %d-NN classifier accuracies, Training: '
              '%.3f, Test: %.3f', step, nval,
              results['train_' + str(nval) + 'nn_acc'],
              results['test_' + str(nval) + 'nn_acc'])
