# Copyright 2021 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Evaluation runner."""

import collections
from absl import logging
import tensorflow.compat.v2 as tf

from galaxy_mergers import config as tp_config
from galaxy_mergers import helpers
from galaxy_mergers import losses
from galaxy_mergers import model
from galaxy_mergers import preprocessing


class GalaxyMergeClassifierEvaluator():
  """Galaxy Merge Rate Prediction Evaluation Runner."""

  def __init__(self, strategy, optimizer_config, total_train_batch_size,
               train_net_args, eval_batch_size, eval_net_args,
               l2_regularization, data_config, resnet_kwargs, n_train_epochs):
    """Initializes evaluator/experiment."""
    logging.info('Initializing evaluator...')
    self._strategy = strategy
    self._data_config = data_config
    self._use_additional_features = bool(data_config['additional_features'])
    self._eval_batch_size = eval_batch_size
    self._eval_net_args = eval_net_args
    self._num_buckets = data_config['num_eval_buckets']
    self._n_repeats = data_config['n_crop_repeat']
    self._image_size = data_config['image_size']
    self._task_type = data_config['task']
    self._loss_config = data_config['loss_config']
    self._model_uncertainty = data_config['model_uncertainty']
    del l2_regularization, optimizer_config, train_net_args
    del total_train_batch_size, n_train_epochs

    logging.info('Creating model...')
    num_classes = 2 if self._model_uncertainty else 1
    if self._task_type == losses.TASK_CLASSIFICATION:
      num_classes = len(self._data_config['class_boundaries'])
    self.model = model.ResNet(
        n_repeats=self._data_config['n_crop_repeat'], num_classes=num_classes,
        use_additional_features=self._use_additional_features, **resnet_kwargs)

    self._eval_input = None

  def build_eval_input(self, additional_lambdas=None):
    """Create the galaxy merger evaluation dataset."""

    def decode_fn(record_bytes):
      parsed_example = tf.io.parse_single_example(
          record_bytes,
          {
              'image':
                  tf.io.VarLenFeature(tf.float32),
              'image_shape':
                  tf.io.FixedLenFeature([3], dtype=tf.int64),
              'axis':
                  tf.io.FixedLenFeature([], dtype=tf.int64),
              'proposed_crop':
                  tf.io.FixedLenFeature([2, 2], dtype=tf.int64),
              'normalized_time':
                  tf.io.FixedLenFeature([], dtype=tf.float32),
              'unnormalized_time':
                  tf.io.FixedLenFeature([], dtype=tf.float32),
              'grounded_normalized_time':
                  tf.io.FixedLenFeature([], dtype=tf.float32),
              'redshift':
                  tf.io.FixedLenFeature([], dtype=tf.float32),
              'sequence_average_redshift':
                  tf.io.FixedLenFeature([], dtype=tf.float32),
              'mass':
                  tf.io.FixedLenFeature([], dtype=tf.float32),
              'time_index':
                  tf.io.FixedLenFeature([], dtype=tf.int64),
              'sequence_id':
                  tf.io.FixedLenFeature([], dtype=tf.string),
          })
      parsed_example['image'] = tf.sparse.to_dense(
          parsed_example['image'], default_value=0)
      dataset_row = parsed_example
      return dataset_row

    def build_eval_pipeline(_):
      """Generate the processed input evaluation data."""

      logging.info('Building evaluation input pipeline...')
      ds_path = self._data_config['dataset_path']
      ds = tf.data.TFRecordDataset([ds_path]).map(decode_fn)

      augmentations = dict(
          rotation_and_flip=False,
          rescaling=False,
          translation=False
          )
      ds = preprocessing.prepare_dataset(
          ds=ds, target_size=self._image_size,
          crop_type=self._data_config['test_crop_type'],
          n_repeats=self._n_repeats,
          augmentations=augmentations,
          task_type=self._task_type,
          additional_features=self._data_config['additional_features'],
          class_boundaries=self._data_config['class_boundaries'],
          time_intervals=self._data_config['time_filter_intervals'],
          frequencies_to_use=self._data_config['frequencies_to_use'],
          additional_lambdas=additional_lambdas)

      batched_ds = ds.cache().batch(self._eval_batch_size).prefetch(128)
      logging.info('Finished building input pipeline...')
      return batched_ds

    return self._strategy.experimental_distribute_datasets_from_function(
        build_eval_pipeline)

  def run_test_model_ensemble(self, images, physical_features, augmentations):
    """Run evaluation on input images."""
    image_variations = [images]
    image_shape = images.shape.as_list()

    if augmentations['rotation_and_flip']:
      image_variations = preprocessing.get_all_rotations_and_flips(
          image_variations)

    if augmentations['rescaling']:
      image_variations = preprocessing.get_all_rescalings(
          image_variations, image_shape[1], augmentations['translation'])

    # Put all augmented images into the batch: batch * num_augmented
    augmented_images = tf.stack(image_variations, axis=0)
    augmented_images = tf.reshape(augmented_images, [-1] + image_shape[1:])
    if self._use_additional_features:
      physical_features = tf.concat(
          [physical_features] * len(image_variations), axis=0)

    n_reps = self._data_config['n_crop_repeat']
    augmented_images = preprocessing.move_repeats_to_batch(augmented_images,
                                                           n_reps)

    logits_or_times = self.model(augmented_images, physical_features,
                                 **self._eval_net_args)
    if self._task_type == losses.TASK_CLASSIFICATION:
      mu, log_sigma_sq = helpers.aggregate_classification_ensemble(
          logits_or_times, len(image_variations),
          self._data_config['test_time_ensembling'])
    else:
      assert self._task_type in losses.REGRESSION_TASKS
      mu, log_sigma_sq = helpers.aggregate_regression_ensemble(
          logits_or_times, len(image_variations),
          self._model_uncertainty,
          self._data_config['test_time_ensembling'])

    return mu, log_sigma_sq

  @property
  def checkpoint_items(self):
    return {'model': self.model}


def run_model_on_dataset(evaluator, dataset, config, n_batches=16):
  """Runs the model against a dataset, aggregates model output."""

  scalar_metrics_to_log = collections.defaultdict(list)
  model_outputs_to_log = collections.defaultdict(list)
  dataset_features_to_log = collections.defaultdict(list)

  batch_count = 1
  for all_inputs in dataset:
    if config.experiment_kwargs.data_config['additional_features']:
      images = all_inputs[0]
      physical_features = all_inputs[1]
      labels, regression_targets, _ = all_inputs[2:5]
      other_dataset_features = all_inputs[5:]
    else:
      images, physical_features = all_inputs[0], None
      labels, regression_targets, _ = all_inputs[1:4]
      other_dataset_features = all_inputs[4:]

    mu, log_sigma_sq = evaluator.run_test_model_ensemble(
        images, physical_features,
        config.experiment_kwargs.data_config['test_augmentations'])

    loss_config = config.experiment_kwargs.data_config['loss_config']
    task_type = config.experiment_kwargs.data_config['task']
    uncertainty = config.experiment_kwargs.data_config['model_uncertainty']
    conf = config.experiment_kwargs.data_config['eval_confidence_interval']
    scalar_metrics, vector_metrics = losses.compute_loss_and_metrics(
        mu, log_sigma_sq, regression_targets, labels,
        task_type, uncertainty, loss_config, 0, conf, mode='eval')

    for i, dataset_feature in enumerate(other_dataset_features):
      dataset_features_to_log[i].append(dataset_feature.numpy())

    for scalar_metric in scalar_metrics:
      v = scalar_metrics[scalar_metric]
      val = v if isinstance(v, int) or isinstance(v, float) else v.numpy()
      scalar_metrics_to_log[scalar_metric].append(val)

    for vector_metric in vector_metrics:
      val = vector_metrics[vector_metric].numpy()
      model_outputs_to_log[vector_metric].append(val)

    regression_targets_np = regression_targets.numpy()
    labels_np = labels.numpy()
    model_outputs_to_log['regression_targets'].append(regression_targets_np)
    model_outputs_to_log['labels'].append(labels_np)
    model_outputs_to_log['model_input_images'].append(images.numpy())

    if n_batches and batch_count >= n_batches:
      break
    batch_count += 1

  return scalar_metrics_to_log, model_outputs_to_log, dataset_features_to_log


def get_config_dataset_evaluator(filter_time_intervals,
                                 ckpt_path,
                                 config_override=None,
                                 setup_dataset=True):
  """Set-up a default config, evaluation dataset, and evaluator."""
  config = tp_config.get_config(filter_time_intervals=filter_time_intervals)

  if config_override:
    with config.ignore_type():
      config.update_from_flattened_dict(config_override)

  strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
  experiment = GalaxyMergeClassifierEvaluator(
      strategy=strategy, **config.experiment_kwargs)

  helpers.restore_checkpoint(ckpt_path, experiment)

  if setup_dataset:
    additional_lambdas = [
        lambda ds: ds['sequence_id'],
        lambda ds: ds['time_index'],
        lambda ds: ds['axis'],
        lambda ds: ds['normalized_time'],
        lambda ds: ds['grounded_normalized_time'],
        lambda ds: ds['unnormalized_time'],
        lambda ds: ds['redshift'],
        lambda ds: ds['mass']
    ]

    ds = experiment.build_eval_input(additional_lambdas=additional_lambdas)
  else:
    ds = None
  return config, ds, experiment
