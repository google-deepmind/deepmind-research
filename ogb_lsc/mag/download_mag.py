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

"""Download data required for training and evaluating models."""

import pathlib

from absl import app
from absl import flags
from absl import logging
from google.cloud import storage

# pylint: disable=g-bad-import-order
import data_utils

Path = pathlib.Path


_BUCKET_NAME = 'deepmind-ogb-lsc'
_MAX_DOWNLOAD_ATTEMPTS = 5

FLAGS = flags.FLAGS

flags.DEFINE_enum('payload', None, ['data', 'models'],
                  'Download "data" or "models"?')
flags.DEFINE_string('task_root', None, 'Local task root directory')

DATA_RELATIVE_PATHS = (
    data_utils.RAW_NODE_YEAR_FILENAME,
    data_utils.TRAIN_INDEX_FILENAME,
    data_utils.VALID_INDEX_FILENAME,
    data_utils.TEST_INDEX_FILENAME,
    data_utils.K_FOLD_SPLITS_DIR,
    data_utils.FUSED_NODE_LABELS_FILENAME,
    data_utils.FUSED_PAPER_EDGES_FILENAME,
    data_utils.FUSED_PAPER_EDGES_T_FILENAME,
    data_utils.EDGES_AUTHOR_INSTITUTION,
    data_utils.EDGES_INSTITUTION_AUTHOR,
    data_utils.EDGES_AUTHOR_PAPER,
    data_utils.EDGES_PAPER_AUTHOR,
    data_utils.PCA_MERGED_FEATURES_FILENAME,
    )


class DataCorruptionError(Exception):
  pass


def _get_gcs_root():
  return Path('mag') / FLAGS.payload


def _get_gcs_bucket():
  storage_client = storage.Client.create_anonymous_client()
  return storage_client.bucket(_BUCKET_NAME)


def _write_blob_to_destination(blob, task_root, ignore_existing=True):
  """Write the blob."""
  logging.info("Copying blob: '%s'", blob.name)
  destination_path = Path(task_root) / Path(*Path(blob.name).parts[1:])
  logging.info("  ... to: '%s'", str(destination_path))
  if ignore_existing and destination_path.exists():
    return
  destination_path.parent.mkdir(parents=True, exist_ok=True)
  checksum = 'crc32c'
  for attempt in range(_MAX_DOWNLOAD_ATTEMPTS):
    try:
      blob.download_to_filename(destination_path.as_posix(), checksum=checksum)
    except storage.client.resumable_media.common.DataCorruption:
      pass
    else:
      break
  else:
    raise DataCorruptionError(f"Checksum ('{checksum}') for {blob.name} failed "
                              f'after {attempt + 1} attempts')


def main(unused_argv):
  bucket = _get_gcs_bucket()
  if FLAGS.payload == 'data':
    relative_paths = DATA_RELATIVE_PATHS
  else:
    relative_paths = (None,)
  for relative_path in relative_paths:
    if relative_path is None:
      relative_path = str(_get_gcs_root())
    else:
      relative_path = str(_get_gcs_root() / relative_path)
    logging.info("Copying relative path: '%s'", relative_path)
    blobs = bucket.list_blobs(prefix=relative_path)
    for blob in blobs:
      _write_blob_to_destination(blob, FLAGS.task_root)


if __name__ == '__main__':
  flags.mark_flag_as_required('payload')
  flags.mark_flag_as_required('task_root')
  app.run(main)
