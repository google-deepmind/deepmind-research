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

"""Training pipeline for the prediction of particle mobilities in glasses."""


import collections
import enum
import pickle
from typing import Any, Dict, List, Optional, Text, Tuple, Sequence

from absl import logging
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


from glassy_dynamics import graph_model

tf.enable_resource_variables()

LossCollection = collections.namedtuple('LossCollection',
                                        'l1_loss, l2_loss, correlation')
GlassSimulationData = collections.namedtuple('GlassSimulationData',
                                             'positions, targets, types, box')


class ParticleType(enum.IntEnum):
  """The simulation contains two particle types, identified as type A and B.

  The dataset encodes the particle type in an integer.
    - 0 corresponds to particle type A.
    - 1 corresponds to particle type B.
  """
  A = 0
  B = 1


def get_targets(
    initial_positions: np.ndarray,
    trajectory_target_positions: Sequence[np.ndarray]) -> np.ndarray:
  """Returns the averaged particle mobilities from the sampled trajectories.

  Args:
    initial_positions: the initial positions of the particles with shape
      [n_particles, 3].
    trajectory_target_positions: the absolute positions of the particles at the
      target time for all sampled trajectories, each with shape
      [n_particles, 3].
  """
  targets = np.mean([np.linalg.norm(t - initial_positions, axis=-1)
                     for t in trajectory_target_positions], axis=0)
  return targets.astype(np.float32)


def load_data(
    file_pattern: Text,
    time_index: int,
    max_files_to_load: Optional[int] = None) -> List[GlassSimulationData]:
  """Returns a dictionary containing the training or test dataset.

  The dictionary contains:
    `positions`: `np.ndarray` containing the particle positions with shape
      [n_particles, 3].
    `targets`: `np.ndarray` containing particle mobilities with shape
      [n_particles].
    `types`: `np.ndarray` containing the particle types with shape with shape
      [n_particles].
    `box`: `np.ndarray` containing the dimensions of the periodic box with shape
      [3].

  Args:
    file_pattern: pattern matching the files with the simulation data.
    time_index: the time index of the targets.
    max_files_to_load: the maximum number of files to load.
  """
  filenames = tf.io.gfile.glob(file_pattern)
  if max_files_to_load:
    filenames = filenames[:max_files_to_load]

  static_structures = []
  for filename in filenames:
    with tf.io.gfile.GFile(filename, 'rb') as f:
      data = pickle.load(f)
    static_structures.append(GlassSimulationData(
        positions=data['positions'].astype(np.float32),
        targets=get_targets(
            data['positions'], data['trajectory_target_positions'][time_index]),
        types=data['types'].astype(np.int32),
        box=data['box'].astype(np.float32)))
  return static_structures


def get_loss_ops(
    prediction: tf.Tensor,
    target: tf.Tensor,
    types: tf.Tensor) -> LossCollection:
  """Returns L1/L2 loss and correlation for type A particles.

  Args:
    prediction: tensor with shape [n_particles] containing the predicted
      particle mobilities.
    target: tensor with shape [n_particles] containing the true particle
      mobilities.
    types: tensor with shape [n_particles] containing the particle types.
  """
  # Considers only type A particles.
  mask = tf.equal(types, ParticleType.A)
  prediction = tf.boolean_mask(prediction, mask)
  target = tf.boolean_mask(target, mask)
  return LossCollection(
      l1_loss=tf.reduce_mean(tf.abs(prediction - target)),
      l2_loss=tf.reduce_mean((prediction - target)**2),
      correlation=tf.squeeze(tfp.stats.correlation(
          prediction[:, tf.newaxis], target[:, tf.newaxis])))


def get_minimize_op(
    loss: tf.Tensor,
    learning_rate: float,
    grad_clip: Optional[float] = None) -> tf.Tensor:
  """Returns minimization operation.

  Args:
    loss: the loss tensor which is minimized.
    learning_rate: the learning rate used by the optimizer.
    grad_clip: all gradients are clipped to the given value if not None or 0.
  """
  optimizer = tf.train.AdamOptimizer(learning_rate)
  grads_and_vars = optimizer.compute_gradients(loss)
  if grad_clip:
    grads, _ = tf.clip_by_global_norm([g for g, _ in grads_and_vars], grad_clip)
    grads_and_vars = [(g, pair[1]) for g, pair in zip(grads, grads_and_vars)]
  minimize = optimizer.apply_gradients(grads_and_vars)
  return minimize


def _log_stats_and_return_mean_correlation(
    label: Text,
    stats: Sequence[LossCollection]) -> float:
  """Logs performance statistics and returns mean correlation.

  Args:
    label: label printed before the combined statistics e.g. train or test.
    stats: statistics calculated for each batch in a dataset.

  Returns:
    mean correlation
  """
  for key in LossCollection._fields:
    values = [getattr(s, key) for s in stats]
    mean = np.mean(values)
    std = np.std(values)
    logging.info('%s: %s: %.4f +/- %.4f', label, key, mean, std)
  return np.mean([s.correlation for s in stats])


def train_model(train_file_pattern: Text,
                test_file_pattern: Text,
                max_files_to_load: Optional[int] = None,
                n_epochs: int = 1000,
                time_index: int = 9,
                augment_data_using_rotations: bool = True,
                learning_rate: float = 1e-4,
                grad_clip: Optional[float] = 1.0,
                n_recurrences: int = 7,
                mlp_sizes: Tuple[int] = (64, 64),
                mlp_kwargs: Optional[Dict[Text, Any]] = None,
                edge_threshold: float = 2.0,
                measurement_store_interval: int = 1000,
                checkpoint_path: Optional[Text] = None) -> float:  # pytype: disable=annotation-type-mismatch
  """Trains GraphModel using tensorflow.

  Args:
    train_file_pattern: pattern matching the files with the training data.
    test_file_pattern: pattern matching the files with the test data.
    max_files_to_load: the maximum number of train and test files to load.
      If None, all files will be loaded.
    n_epochs: the number of passes through the training dataset (epochs).
    time_index: the time index (0-9) of the target mobilities.
    augment_data_using_rotations: data is augemented by using random rotations.
    learning_rate: the learning rate used by the optimizer.
    grad_clip: all gradients are clipped to the given value.
    n_recurrences: the number of message passing steps in the graphnet.
    mlp_sizes: the number of neurons in each layer of the MLP.
    mlp_kwargs: additional keyword aguments passed to the MLP.
    edge_threshold: particles at distance less than threshold are connected by
      an edge.
    measurement_store_interval: number of steps between storing objective values
      (loss and correlation).
    checkpoint_path: path used to store the checkpoint with the highest
      correlation on the test set.

  Returns:
    Correlation on the test dataset of best model encountered during training.
  """
  if mlp_kwargs is None:
    mlp_kwargs = dict(initializers=dict(w=tf.variance_scaling_initializer(1.0),
                                        b=tf.variance_scaling_initializer(0.1)))
  # Loads train and test dataset.
  dataset_kwargs = dict(
      time_index=time_index,
      max_files_to_load=max_files_to_load)
  training_data = load_data(train_file_pattern, **dataset_kwargs)
  test_data = load_data(test_file_pattern, **dataset_kwargs)

  # Defines wrapper functions, which can directly be passed to the
  # tf.data.Dataset.map function.
  def _make_graph_from_static_structure(static_structure):
    """Converts static structure to graph, targets and types."""
    return (graph_model.make_graph_from_static_structure(
        static_structure.positions,
        static_structure.types,
        static_structure.box,
        edge_threshold),
            static_structure.targets,
            static_structure.types)

  def _apply_random_rotation(graph, targets, types):
    """Applies random rotations to the graph and forwards targets and types."""
    return graph_model.apply_random_rotation(graph), targets, types

  # Defines data-pipeline based on tf.data.Dataset following the official
  # guideline: https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays.
  # We use initializable iterators to avoid embedding the training and test data
  # directly into the graph.
  # Instead we feed the data to the iterators during the initalization of the
  # iterators before the main training loop.
  placeholders = GlassSimulationData._make(
      tf.placeholder(s.dtype, (None,) + s.shape) for s in training_data[0])
  dataset = tf.data.Dataset.from_tensor_slices(placeholders)
  dataset = dataset.map(_make_graph_from_static_structure)
  dataset = dataset.cache()
  dataset = dataset.shuffle(400)
  # Augments data. This has to be done after calling dataset.cache!
  if augment_data_using_rotations:
    dataset = dataset.map(_apply_random_rotation)
  dataset = dataset.repeat()
  train_iterator = dataset.make_initializable_iterator()

  dataset = tf.data.Dataset.from_tensor_slices(placeholders)
  dataset = dataset.map(_make_graph_from_static_structure)
  dataset = dataset.cache()
  dataset = dataset.repeat()
  test_iterator = dataset.make_initializable_iterator()

  # Creates tensorflow graph.
  # Note: We decouple the training and test datasets from the input pipeline
  # by creating a new iterator from a string-handle placeholder with the same
  # output types and shapes as the training dataset.
  dataset_handle = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(
      dataset_handle, train_iterator.output_types, train_iterator.output_shapes)
  graph, targets, types = iterator.get_next()

  model = graph_model.GraphBasedModel(
      n_recurrences, mlp_sizes, mlp_kwargs)
  prediction = model(graph)

  # Defines loss and minimization operations.
  loss_ops = get_loss_ops(prediction, targets, types)
  minimize_op = get_minimize_op(loss_ops.l2_loss, learning_rate, grad_clip)

  best_so_far = -1
  train_stats = []
  test_stats = []

  saver = tf.train.Saver()

  with tf.train.SingularMonitoredSession() as session:
    # Initializes train and test iterators with the training and test datasets.
    # The obtained training and test string-handles can be passed to the
    # dataset_handle placeholder to select the dataset.
    train_handle = session.run(train_iterator.string_handle())
    test_handle = session.run(test_iterator.string_handle())
    feed_dict = {p: [x[i] for x in training_data]
                 for i, p in enumerate(placeholders)}
    session.run(train_iterator.initializer, feed_dict=feed_dict)
    feed_dict = {p: [x[i] for x in test_data]
                 for i, p in enumerate(placeholders)}
    session.run(test_iterator.initializer, feed_dict=feed_dict)

    # Trains model using stochatic gradient descent on the training dataset.
    n_training_steps = len(training_data) * n_epochs
    for i in range(n_training_steps):
      feed_dict = {dataset_handle: train_handle}
      train_loss, _ = session.run((loss_ops, minimize_op), feed_dict=feed_dict)
      train_stats.append(train_loss)

      if (i+1) % measurement_store_interval == 0:
        # Evaluates model on test dataset.
        for _ in range(len(test_data)):
          feed_dict = {dataset_handle: test_handle}
          test_stats.append(session.run(loss_ops, feed_dict=feed_dict))

        # Outputs performance statistics on training and test dataset.
        _log_stats_and_return_mean_correlation('Train', train_stats)
        correlation = _log_stats_and_return_mean_correlation('Test', test_stats)
        train_stats = []
        test_stats = []

        # Updates best model based on the observed correlation on the test
        # dataset.
        if correlation > best_so_far:
          best_so_far = correlation
          if checkpoint_path:
            saver.save(session.raw_session(), checkpoint_path)

  return best_so_far


def apply_model(checkpoint_path: Text,
                file_pattern: Text,
                max_files_to_load: Optional[int] = None,
                time_index: int = 9) -> List[np.ndarray]:
  """Applies trained GraphModel using tensorflow.

  Args:
    checkpoint_path: path from which the model is loaded.
    file_pattern: pattern matching the files with the data.
    max_files_to_load: the maximum number of files to load.
      If None, all files will be loaded.
    time_index: the time index (0-9) of the target mobilities.

  Returns:
    Predictions of the model for all files.
  """
  dataset_kwargs = dict(
      time_index=time_index,
      max_files_to_load=max_files_to_load)
  data = load_data(file_pattern, **dataset_kwargs)

  tf.reset_default_graph()
  saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
  graph = tf.get_default_graph()

  placeholders = GlassSimulationData(
      positions=graph.get_tensor_by_name('Placeholder:0'),
      targets=graph.get_tensor_by_name('Placeholder_1:0'),
      types=graph.get_tensor_by_name('Placeholder_2:0'),
      box=graph.get_tensor_by_name('Placeholder_3:0'))
  prediction_tensor = graph.get_tensor_by_name('Graph_1/Squeeze:0')
  correlation_tensor = graph.get_tensor_by_name('Squeeze:0')

  dataset_handle = graph.get_tensor_by_name('Placeholder_4:0')
  test_initalizer = graph.get_operation_by_name('MakeIterator_1')
  test_string_handle = graph.get_tensor_by_name('IteratorToStringHandle_1:0')

  with tf.Session() as session:
    saver.restore(session, checkpoint_path)
    handle = session.run(test_string_handle)
    feed_dict = {p: [x[i] for x in data] for i, p in enumerate(placeholders)}
    session.run(test_initalizer, feed_dict=feed_dict)
    predictions = []
    correlations = []
    for _ in range(len(data)):
      p, c = session.run((prediction_tensor, correlation_tensor),
                         feed_dict={dataset_handle: handle})
      predictions.append(p)
      correlations.append(c)

  logging.info('Correlation: %.4f +/- %.4f',
               np.mean(correlations),
               np.std(correlations))
  return predictions
