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
"""Adult dataset.

See https://archive.ics.uci.edu/ml/datasets/adult.
"""

import os

from absl import logging
import pandas as pd




_COLUMNS = ('age', 'workclass', 'final-weight', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'income')
_CATEGORICAL_COLUMNS = ('workclass', 'education', 'marital-status',
                        'occupation', 'race', 'relationship', 'sex',
                        'native-country', 'income')


def _read_data(
    name,
    data_path=''):
  with os.path.join(data_path, name) as data_file:
    data = pd.read_csv(data_file, header=None, index_col=False,
                       names=_COLUMNS, skipinitialspace=True, na_values='?')
    for categorical in _CATEGORICAL_COLUMNS:
      data[categorical] = data[categorical].astype('category')
  return data


def _combine_category_coding(df_1, df_2):
  """Combines the categories between dataframes df_1 and df_2.

  This is used to ensure that training and test data use the same category
  coding, so that the one-hot vectors representing the values are compatible
  between training and test data.

  Args:
    df_1: Pandas DataFrame.
    df_2: Pandas DataFrame. Must have the same columns as df_1.
  """
  for column in df_1.columns:
    if df_1[column].dtype.name == 'category':
      categories_1 = set(df_1[column].cat.categories)
      categories_2 = set(df_2[column].cat.categories)
      categories = sorted(categories_1 | categories_2)
      df_1[column].cat.set_categories(categories, inplace=True)
      df_2[column].cat.set_categories(categories, inplace=True)


def read_all_data(root_dir, remove_missing=True):
  """Return (train, test) dataframes, optionally removing incomplete rows."""
  train_data = _read_data('adult.data', root_dir)
  test_data = _read_data('adult.test', root_dir)
  _combine_category_coding(train_data, test_data)
  if remove_missing:
    train_data = train_data.dropna()
    test_data = test_data.dropna()

  logging.info('Training data dtypes: %s', train_data.dtypes)
  return train_data, test_data
