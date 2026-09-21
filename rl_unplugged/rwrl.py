# Copyright 2020 DeepMind Technologies Limited.
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
"""Real World RL for RL Unplugged datasets.

Examples in the dataset represent SARS transitions stored when running a
partially online trained agent as described in https://arxiv.org/abs/1904.12901.

We release 8 datasets in total -- with no combined challenge and easy combined
challenge on the cartpole, walker, quadruped, and humanoid tasks. For details
on how the dataset was generated, please refer to the paper.

Every transition in the dataset is a tuple containing the following features:

* o_t: Observation at time t. Observations have been processed using the
    canonical
* a_t: Action taken at time t.
* r_t: Reward at time t.
* d_t: Discount at time t.
* o_tp1: Observation at time t+1.
* a_tp1: Action taken at time t+1. This is set to equal to the last action
    for the last timestep.

Note that this serves as an example. For optimal data loading speed, consider
separating out data preprocessing from the data loading loop during training,
e.g. saving the preprocessed data.
"""

import collections
import functools
import os
from typing import Any, Dict, Optional, Sequence

from acme import wrappers
import dm_env
import realworldrl_suite.environments as rwrl_envs
import reverb
import tensorflow as tf
import tree

DELIMITER = ':'
# Control suite tasks have 1000 timesteps per episode. One additional timestep
# accounts for the very first observation where no action has been taken yet.
DEFAULT_NUM_TIMESTEPS = 1001


def _decombine_key(k: str, delimiter: str = DELIMITER) -> Sequence[str]:
  return k.split(delimiter)


def tf_example_to_feature_description(example,
                                      num_timesteps=DEFAULT_NUM_TIMESTEPS):
  """Takes a string tensor encoding an tf example and returns its features."""
  if not tf.executing_eagerly():
    raise AssertionError(
        'tf_example_to_reverb_sample() only works under eager mode.')
  example = tf.train.Example.FromString(example.numpy())

  ret = {}
  for k, v in example.features.feature.items():
    l = len(v.float_list.value)
    if l % num_timesteps:
      raise ValueError('Unexpected feature length %d. It should be divisible '
                       'by num_timesteps: %d' % (l, num_timesteps))
    size = l // num_timesteps
    ret[k] = tf.io.FixedLenFeature([num_timesteps, size], tf.float32)
  return ret


def tree_deflatten_with_delimiter(
    flat_dict: Dict[str, Any], delimiter: str = DELIMITER) -> Dict[str, Any]:
  """De-flattens a dict to its originally nested structure.

  Does the opposite of {combine_nested_keys(k) :v
                        for k, v in tree.flatten_with_path(nested_dicts)}
  Example: {'a:b': 1} -> {'a': {'b': 1}}
  Args:
    flat_dict: the keys of which equals the `path` separated by `delimiter`.
    delimiter: the delimiter that separates the keys of the nested dict.

  Returns:
    An un-flattened dict.
  """
  root = collections.defaultdict(dict)
  for delimited_key, v in flat_dict.items():
    keys = _decombine_key(delimited_key, delimiter=delimiter)
    node = root
    for k in keys[:-1]:
      node = node[k]
    node[keys[-1]] = v
  return dict(root)


def get_slice_of_nested(nested: Dict[str, Any], start: int,
                        end: int) -> Dict[str, Any]:
  return tree.map_structure(lambda item: item[start:end], nested)


def repeat_last_and_append_to_nested(nested: Dict[str, Any]) -> Dict[str, Any]:
  return tree.map_structure(
      lambda item: tf.concat((item, item[-1:]), axis=0), nested)


def tf_example_to_reverb_sample(example,
                                feature_description,
                                num_timesteps=DEFAULT_NUM_TIMESTEPS):
  """Converts the episode encoded as a tf example into SARSA reverb samples."""
  example = tf.io.parse_single_example(example, feature_description)
  kv = tree_deflatten_with_delimiter(example)
  output = (
      get_slice_of_nested(kv['observation'], 0, num_timesteps - 1),
      get_slice_of_nested(kv['action'], 1, num_timesteps),
      kv['reward'][1:num_timesteps],
      # The two fields below aren't needed for learning,
      # but are kept here to be compatible with acme learner format.
      kv['discount'][1:num_timesteps],
      get_slice_of_nested(kv['observation'], 1, num_timesteps),
      repeat_last_and_append_to_nested(
          get_slice_of_nested(kv['action'], 2, num_timesteps)))
  ret = tf.data.Dataset.from_tensor_slices(output)
  ret = ret.map(lambda *x: reverb.ReplaySample(info=b'None', data=x))  # pytype: disable=wrong-arg-types
  return ret


def dataset(path: str,
            combined_challenge: str,
            domain: str,
            task: str,
            difficulty: str,
            num_shards: int = 100,
            shuffle_buffer_size: int = 100000) -> tf.data.Dataset:
  """TF dataset of RWRL SARSA tuples."""
  path = os.path.join(
      path,
      f'combined_challenge_{combined_challenge}/{domain}/{task}/'
      f'offline_rl_challenge_{difficulty}'
  )
  filenames = [
      f'{path}/episodes.tfrecord-{i:05d}-of-{num_shards:05d}'
      for i in range(num_shards)
  ]
  file_ds = tf.data.Dataset.from_tensor_slices(filenames)
  file_ds = file_ds.repeat().shuffle(num_shards)
  tf_example_ds = file_ds.interleave(
      tf.data.TFRecordDataset,
      cycle_length=tf.data.experimental.AUTOTUNE,
      block_length=5)

  # Take one item to get the output types and shapes.
  example_item = None
  for example_item in tf.data.TFRecordDataset(filenames[:1]).take(1):
    break
  if example_item is None:
    raise ValueError('Empty dataset')

  feature_description = tf_example_to_feature_description(example_item)

  reverb_ds = tf_example_ds.interleave(
      functools.partial(
          tf_example_to_reverb_sample, feature_description=feature_description),
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      deterministic=False)
  reverb_ds = reverb_ds.prefetch(100)
  reverb_ds = reverb_ds.shuffle(shuffle_buffer_size)

  return reverb_ds


def environment(
    combined_challenge: str,
    domain: str,
    task: str,
    log_output: Optional[str] = None,
    environment_kwargs: Optional[Dict[str, Any]] = None) -> dm_env.Environment:
  """RWRL environment."""
  env = rwrl_envs.load(
      domain_name=domain,
      task_name=task,
      log_output=log_output,
      environment_kwargs=environment_kwargs,
      combined_challenge=combined_challenge)
  return wrappers.SinglePrecisionWrapper(env)
