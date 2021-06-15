# Copyright 2021 DeepMind Technologies Limited.
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

"""Ensemble k-fold predictions and generate final submission file."""

import collections
import os

from absl import app
from absl import flags
from absl import logging
import dill
import jax
import numpy as np
from ogb import lsc

# pylint: disable=g-bad-import-order
import data_utils
import losses


_NUM_KFOLD_SPLITS = 10

FLAGS = flags.FLAGS


_DATA_ROOT = flags.DEFINE_string('data_root', None, 'Path to the data root')
_SPLIT = flags.DEFINE_enum('split', None, ['valid', 'test'], 'Data split')
_PREDICTIONS_PATH = flags.DEFINE_string(
    'predictions_path', None, 'Path with the output of the k-fold models.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', None, 'Output path.')


def _np_one_hot(targets: np.ndarray, nb_classes: int):
  res = np.zeros(targets.shape + (nb_classes,), dtype=np.float32)
  np.put_along_axis(res, targets.astype(np.int32)[..., None], 1.0, axis=-1)
  return res


def ensemble_predictions(
    node_idx_to_logits_list,
    all_labels,
    node_indices,
    use_mode_break_tie_by_mean: bool = True,
):
  """Ensemble together predictions for each node and generate final predictions."""
  # First, assert that each node has the same number of predictions to ensemble.
  num_predictions_per_node = [
      len(x) for x in node_idx_to_logits_list.values()
  ]
  num_models = np.unique(num_predictions_per_node)
  assert num_models.shape[0] == 1
  num_models = num_models[0]
  # Gather all logits, shape should be [num_nodes, num_models, num_classes].
  all_logits = np.stack(
      [np.stack(node_idx_to_logits_list[idx]) for idx in node_indices])
  assert all_logits.shape == (node_indices.shape[0], num_models,
                              data_utils.NUM_CLASSES)
  # Softmax on the final axis.
  all_probs = jax.nn.softmax(all_logits, axis=-1)
  # Take average across models axis to get probabilities.
  mean_probs = np.mean(all_probs, axis=1)

  # Assert there are no 2 equal logits for different classes.
  max_logit_value = np.max(all_logits, axis=-1)
  num_classes_with_max_value = (
      all_logits == max_logit_value[..., None]).sum(axis=-1)

  num_logit_ties = (num_classes_with_max_value > 1).sum()
  if num_logit_ties:
    logging.warn(
        'Found %d models with the exact same logits for two of the classes. '
        '`argmax` will choose the first.', num_logit_ties)

  # Each model votes on one class per type.
  all_votes = np.argmax(all_logits, axis=-1)
  assert all_votes.shape == (node_indices.shape[0], num_models)

  all_votes_one_hot = _np_one_hot(all_votes, data_utils.NUM_CLASSES)
  assert all_votes_one_hot.shape == (node_indices.shape[0], num_models,
                                     data_utils.NUM_CLASSES)

  num_votes_per_class = np.sum(all_votes_one_hot, axis=1)
  assert num_votes_per_class.shape == (
      node_indices.shape[0], data_utils.NUM_CLASSES)

  if use_mode_break_tie_by_mean:
    # Slight hack, give high weight to votes (any number > 1 works really)
    # and add probabilities between [0, 1] per class to tie-break only within
    # classes with equal votes.
    total_score = 10 * num_votes_per_class + mean_probs
  else:
    # Just take mean.
    total_score = mean_probs

  ensembled_logits = np.log(total_score)
  return losses.Predictions(
      node_indices=node_indices,
      labels=all_labels,
      logits=ensembled_logits,
      predictions=np.argmax(ensembled_logits, axis=-1),
  )


def load_predictions(predictions_path, split):
  """Loads set of predictions made by given XID."""

  # Generate list of predictions per node.
  # Note for validation each validation index is only present in exactly 1
  # model of the k-fold, however for test it is present in all of them.
  node_idx_to_logits_list = collections.defaultdict(list)

  # For the 10 models in the ensemble.
  for i in range(_NUM_KFOLD_SPLITS):
    path = os.path.join(predictions_path, str(i))

    # Find subdirectories.
    # Directories will be something like:
    # os.path.join(path, "step_104899_2021-06-14T18:20:05", "(test|valid).dill")
    # So we make sure there is only one.
    candidates = []
    for date_str in os.listdir(path):
      candidate_path = os.path.join(path, date_str, f'{split}.dill')
      if os.path.exists(candidate_path):
        candidates.append(candidate_path)
    if not candidates:
      raise ValueError(f'No {split} predictions found at {path}')
    elif len(candidates) > 1:
      raise ValueError(f'Found more than one {split} predictions: {candidates}')

    path_for_kth_model_predictions = candidates[0]
    with open(path_for_kth_model_predictions, 'rb') as f:
      results = dill.load(f)
    logging.info('Loaded %s', path_for_kth_model_predictions)
    for (node_idx, logits) in zip(results.node_indices,
                                  results.logits):
      node_idx_to_logits_list[node_idx].append(logits)

  return node_idx_to_logits_list


def generate_ensembled_predictions(
    data_root: str, predictions_path: str, split: str) -> losses.Predictions:
  """Ensemble checkpoints from all WIDs in XID and generates submission file."""

  array_dict = data_utils.get_arrays(
      data_root=data_root,
      return_pca_embeddings=False,
      return_adjacencies=False)

  # Load all valid and test predictions.
  node_idx_to_logits_list = load_predictions(predictions_path, split)

  # Assert that the indices loaded are as expected.
  expected_idx = array_dict[f'{split}_indices']
  idx_found = np.array(list(node_idx_to_logits_list.keys()))
  assert np.all(np.sort(idx_found) == expected_idx)

  if split == 'valid':
    true_labels = array_dict['paper_label'][expected_idx.astype(np.int32)]
  else:
    # Don't know the test labels.
    true_labels = np.full(expected_idx.shape, np.nan)

  # Ensemble together all predictions.
  return ensemble_predictions(
      node_idx_to_logits_list, true_labels, expected_idx)


def evaluate_validation(valid_predictions):

  evaluator = lsc.MAG240MEvaluator()

  evaluator_ouput = evaluator.eval(
      dict(y_pred=valid_predictions.predictions.astype(np.float64),
           y_true=valid_predictions.labels))
  logging.info(
      'Validation accuracy as reported by MAG240MEvaluator: %s',
      evaluator_ouput)


def save_test_submission_file(test_predictions, output_dir):
  evaluator = lsc.MAG240MEvaluator()
  evaluator.save_test_submission(
      dict(y_pred=test_predictions.predictions.astype(np.float64)), output_dir)
  logging.info('Test submission file generated at %s', output_dir)


def main(argv):
  del argv

  split = _SPLIT.value

  ensembled_predictions = generate_ensembled_predictions(
      data_root=_DATA_ROOT.value,
      predictions_path=_PREDICTIONS_PATH.value,
      split=split)

  output_dir = _OUTPUT_PATH.value
  os.makedirs(output_dir, exist_ok=True)

  if split == 'valid':
    evaluate_validation(ensembled_predictions)
  elif split == 'test':
    save_test_submission_file(ensembled_predictions, output_dir)

  ensembled_predictions_path = os.path.join(output_dir, f'{split}.dill')
  assert not os.path.exists(ensembled_predictions_path)
  with open(ensembled_predictions_path, 'wb') as f:
    dill.dump(ensembled_predictions, f)
  logging.info(
      '%s predictions stored at %s', split, ensembled_predictions_path)


if __name__ == '__main__':
  app.run(main)
