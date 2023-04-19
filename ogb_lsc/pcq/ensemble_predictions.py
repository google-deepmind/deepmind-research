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

"""Script to generate ensembled PCQ test predictions."""

import collections
import os
import pathlib
from typing import List, NamedTuple

from absl import app
from absl import flags
from absl import logging
import dill
import numpy as np
from ogb import lsc

# pylint: disable=g-bad-import-order
# pytype: disable=import-error
import datasets

_NUM_SEEDS = 2

_CLIP_VALUE = 20.

_NUM_KFOLD_SPLITS = 10

_SEED_START = flags.DEFINE_integer(
    'seed_start', 42, 'Initial seed for the list of ensemble models.')

_CONFORMER_PATH = flags.DEFINE_string(
    'conformer_path', None, 'Path to conformer predictions.', required=True)

_NON_CONFORMER_PATH = flags.DEFINE_string(
    'non_conformer_path',
    None,
    'Path to non-conformer predictions.',
    required=True)

_OUTPUT_PATH = flags.DEFINE_string('output_path', None, 'Output path.')

_SPLIT = flags.DEFINE_enum('split', 'test', ['test', 'valid'],
                           'Split: valid or test.')


class _Predictions(NamedTuple):
  predictions: np.ndarray
  indices: np.ndarray


def _load_dill(fname) -> bytes:
  with open(fname, 'rb') as f:
    return dill.load(f)


def _sort_by_indices(predictions: _Predictions) -> _Predictions:
  order = np.argsort(predictions.indices)
  return _Predictions(
      predictions=predictions.predictions[order],
      indices=predictions.indices[order])


def load_predictions(path: str, split: str) -> _Predictions:
  """Load written prediction file."""
  if len(os.listdir(path)) != 1:
    raise ValueError('Prediction directory must have exactly '
                     'one prediction sub-directory: %s' % path)

  prediction_subdir = os.listdir(path)[0]
  return _Predictions(*_load_dill(f'{path}/{prediction_subdir}/{split}.dill'))


def mean_mae_distance(x, y):
  return np.abs(x - y).mean()


def _load_valid_labels() -> np.ndarray:
  labels = [label for _, label in datasets.load_smile_strings(with_labels=True)]
  return np.array([labels[i] for i in datasets.load_splits()['valid']])


def evaluate_valid_predictions(ensembled_predictions: _Predictions):
  """Evaluates the predictions on the validation set."""
  ensembled_predictions = _sort_by_indices(ensembled_predictions)
  evaluator = lsc.PCQM4MEvaluator()
  results = evaluator.eval(
      dict(
          y_pred=ensembled_predictions.predictions,
          y_true=_load_valid_labels()))
  logging.info('MAE on validation dataset: %f', results['mae'])


def clip_predictions(predictions: _Predictions) -> _Predictions:
  return predictions._replace(
      predictions=np.clip(predictions.predictions, 0., _CLIP_VALUE))


def _generate_test_prediction_file(test_predictions: np.ndarray,
                                   output_path: pathlib.Path) -> pathlib.Path:
  """Generates the final file for submission."""

  # Check that predictions are not nuts.
  assert test_predictions.dtype in [np.float64, np.float32]
  assert not np.any(np.isnan(test_predictions))
  assert np.all(np.isfinite(test_predictions))
  assert test_predictions.min() >= 0.
  assert test_predictions.max() <= 40.

  # Too risky to overwrite.
  if output_path.exists():
    raise ValueError(f'{output_path} already exists')

  # Write to a local directory, and copy to final path (possibly cns).
  # It is not possible to write directlt on CNS.
  evaluator = lsc.PCQM4MEvaluator()

  evaluator.save_test_submission(
      dict(y_pred=test_predictions), str(output_path))
  return output_path


def merge_complementary_results(split: str, results_a: _Predictions,
                                results_b: _Predictions) -> _Predictions:
  """Merges two prediction results with no overlap."""

  indices_a = set(results_a.indices)
  indices_b = set(results_b.indices)
  assert not indices_a.intersection(indices_b)

  if split == 'test':
    merged_indices = list(sorted(indices_a | indices_b))
    expected_indices = datasets.load_splits()[split]
    assert np.all(expected_indices == merged_indices)

  predictions = np.concatenate([results_a.predictions, results_b.predictions])
  indices = np.concatenate([results_a.indices, results_b.indices])
  predictions = _sort_by_indices(
      _Predictions(indices=indices, predictions=predictions))
  return predictions


def ensemble_valid_predictions(
    predictions_list: List[_Predictions]) -> _Predictions:
  """Ensembles a list of predictions."""
  index_to_predictions = collections.defaultdict(list)
  for predictions in predictions_list:
    for idx, pred in zip(predictions.indices, predictions.predictions):
      index_to_predictions[idx].append(pred)

  for idx, ensemble_list in index_to_predictions.items():
    if len(ensemble_list) != _NUM_SEEDS:
      raise RuntimeError(
          'Graph index in the validation set received wrong number of '
          'predictions to ensemble.')

  index_to_predictions = {
      k: np.median(pred_list, axis=0)
      for k, pred_list in index_to_predictions.items()
  }
  return _sort_by_indices(
      _Predictions(
          indices=np.array(list(index_to_predictions.keys())),
          predictions=np.array(list(index_to_predictions.values()))))


def ensemble_test_predictions(
    predictions_list: List[_Predictions]) -> _Predictions:
  """Ensembles a list of predictions."""
  predictions = np.median([pred.predictions for pred in predictions_list],
                          axis=0)
  common_indices = predictions_list[0].indices
  for preds in predictions_list[1:]:
    assert np.all(preds.indices == common_indices)

  return _Predictions(predictions=predictions, indices=common_indices)


def create_submission_from_predictions(
    output_path: pathlib.Path, test_predictions: _Predictions) -> pathlib.Path:
  """Creates a submission for predictions on a path."""
  assert _SPLIT.value == 'test'

  output_path = _generate_test_prediction_file(
      test_predictions.predictions,
      output_path=output_path / 'submission_files')

  return output_path / 'y_pred_pcqm4m.npz'


def merge_predictions(split: str) -> List[_Predictions]:
  """Generates features merged from conformer and non-conformer predictions."""
  merged_predictions: List[_Predictions] = []
  seed = _SEED_START.value

  # Load conformer and non-conformer predictions.
  for unused_seed_group in (0, 1):
    for k in range(_NUM_KFOLD_SPLITS):
      conformer_predictions: _Predictions = load_predictions(
          f'{_CONFORMER_PATH.value}/k{k}_seed{seed}', split)

      non_conformer_predictions: _Predictions = load_predictions(
          f'{_NON_CONFORMER_PATH.value}/k{k}_seed{seed}', split)

      merged_predictions.append(
          merge_complementary_results(_SPLIT.value, conformer_predictions,
                                      non_conformer_predictions))

      seed += 1
  return merged_predictions


def main(_):
  split: str = _SPLIT.value

  # Merge conformer and non-conformer predictions.
  merged_predictions = merge_predictions(split)

  # Clip before ensembling.
  clipped_predictions = list(map(clip_predictions, merged_predictions))

  # Ensemble predictions.
  if split == 'valid':
    ensembled_predictions = ensemble_valid_predictions(clipped_predictions)
  else:
    assert split == 'test'
    ensembled_predictions = ensemble_test_predictions(clipped_predictions)

  # Clip after ensembling.
  ensembled_predictions = clip_predictions(ensembled_predictions)

  ensembled_predictions_path = pathlib.Path(_OUTPUT_PATH.value)
  ensembled_predictions_path.mkdir(parents=True, exist_ok=True)

  with open(ensembled_predictions_path / f'{split}_predictions.dill',
            'wb') as f:
    dill.dump(ensembled_predictions, f)

  if split == 'valid':
    evaluate_valid_predictions(ensembled_predictions)
  else:
    assert split == 'test'
    output_path = create_submission_from_predictions(ensembled_predictions_path,
                                                     ensembled_predictions)
    logging.info('Submission files written to %s', output_path)


if __name__ == '__main__':
  app.run(main)
