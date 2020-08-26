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
# ============================================================================

"""Level generator for Mujoban based on levels from follwing dataset.

https://github.com/deepmind/boxoban-levels/
"""

import glob
import os
import zipfile

import numpy as np
import requests

BOXOBAN_URL = "https://github.com/deepmind/boxoban-levels/archive/master.zip"


def boxoban_level_generator(levels_set="unfiltered", data_split="valid"):
  env = Boxoban(levels_set=levels_set, data_split=data_split)
  while True:
    index = np.random.randint(0, env.num_levels-1)
    yield env.levels[index]


class Boxoban(object):
  """Class for loading and generatting Boxoban levels."""

  def __init__(self,
               levels_set="unfiltered",
               data_split="valid"):
    self._levels_set = levels_set
    self._data_split = data_split
    self._levels = []

    data_file_path_local = os.path.join(os.path.dirname(__file__),
                                        "boxoban_cache",
                                        "{}_{}.npz".format(self._levels_set,
                                                           self._data_split))

    data_file_path_global = os.path.join("/tmp/boxoban_cache",
                                         "{}_{}.npz".format(self._levels_set,
                                                            self._data_split))

    if os.path.exists(data_file_path_local):
      self.levels = np.load(data_file_path_local)["levels"]
    elif os.path.exists(data_file_path_global):
      self.levels = np.load(data_file_path_global)["levels"]
    else:
      self.levels = self.get_data()
    self.num_levels = len(self.levels)

  def get_data(self):
    """Downloads and cache the data."""
    try:
      cache_path = os.path.join(
          os.path.dirname(__file__), "boxoban_cache")
      os.makedirs(cache_path, exist_ok=True)
    except PermissionError:
      cache_path = os.path.join("/tmp/boxoban_cache")
      if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)

    # Get the zip file
    zip_file_path = os.path.join(cache_path, "master.zip")
    if not os.path.exists(zip_file_path):
      response = requests.get(BOXOBAN_URL, stream=True)
      handle = open(zip_file_path, "wb")
      for chunk in response.iter_content(chunk_size=512):
        if chunk:
          handle.write(chunk)
      handle.close()

      with zipfile.ZipFile(zip_file_path, "r") as zipref:
        zipref.extractall(cache_path)

    # convert to npz
    path = os.path.join(cache_path, "boxoban-levels-master",
                        self._levels_set,
                        self._data_split)
    files = glob.glob(path + "/*.txt")
    levels = "".join([open(f, "r").read() for f in files])
    levels = levels.split("\n;")
    levels = ["\n".join(item.split("\n")[1:]) for item in levels]
    levels = np.asarray(levels)
    data_file_path = os.path.join(
        cache_path, "{}_{}.npz".format(self._levels_set, self._data_split))
    np.savez(data_file_path, levels=levels)
    return levels
