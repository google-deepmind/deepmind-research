# Lint as: python3.
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for storing configuration flags."""

import json


class ConfigDict(dict):
  """Configuration dictionary with convenient dot element access."""

  def __init__(self, *args, **kwargs):
    super(ConfigDict, self).__init__(*args, **kwargs)
    for arg in args:
      if isinstance(arg, dict):
        for key, value in arg.items():
          self._add(key, value)
    for key, value in kwargs.items():
      self._add(key, value)

  def _add(self, key, value):
    if isinstance(value, dict):
      self[key] = ConfigDict(value)
    else:
      self[key] = value

  def __getattr__(self, attr):
    try:
      return self[attr]
    except KeyError as e:
      raise AttributeError(e)

  def __setattr__(self, key, value):
    self.__setitem__(key, value)

  def __setitem__(self, key, value):
    super(ConfigDict, self).__setitem__(key, value)
    self.__dict__.update({key: value})

  def __delattr__(self, item):
    self.__delitem__(item)

  def __delitem__(self, key):
    super(ConfigDict, self).__delitem__(key)
    del self.__dict__[key]

  def to_json(self):
    return json.dumps(self)

  @classmethod
  def from_json(cls, json_string):
    return cls(json.loads(json_string))
