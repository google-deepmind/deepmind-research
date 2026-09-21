# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Settings for adding noise to the action and measurements."""

import numpy as np
from numpy import random

from fusion_tcv import tcv_common


class Noise:
  """Class for adding noise to the action and measurements."""

  def __init__(self,
               action_mean=None,
               action_std=None,
               measurements_mean=None,
               measurements_std=None,
               seed=None):
    """Initializes the class.

    Args:
      action_mean: mean of the Gaussian noise (action bias).
      action_std: std of the Gaussian action noise.
      measurements_mean: mean of the Gaussian noise (measurement bias).
      measurements_std: Dictionary mapping the tcv measurement names to noise.
      seed: seed for the random number generator. If none seed is unset.
    """
    # Check all of the shapes are present and correct.
    assert action_std.shape == (tcv_common.NUM_ACTIONS,)
    assert action_mean.shape == (tcv_common.NUM_ACTIONS,)
    for name, num in tcv_common.TCV_MEASUREMENTS.items():
      assert name in measurements_std
      assert measurements_mean[name].shape == (num,)
      assert measurements_std[name].shape == (num,)
    self._action_mean = action_mean
    self._action_std = action_std
    self._meas_mean = measurements_mean
    self._meas_std = measurements_std
    self._meas_mean_vec = tcv_common.dict_to_measurement(self._meas_mean)
    self._meas_std_vec = tcv_common.dict_to_measurement(self._meas_std)
    self._gen = random.RandomState(seed)

  @classmethod
  def use_zero_noise(cls):
    no_noise_mean = dict()
    no_noise_std = dict()
    for name, num in tcv_common.TCV_MEASUREMENTS.items():
      no_noise_mean[name] = np.zeros((num,))
      no_noise_std[name] = np.zeros((num,))
    return cls(
        action_mean=np.zeros((tcv_common.NUM_ACTIONS)),
        action_std=np.zeros((tcv_common.NUM_ACTIONS)),
        measurements_mean=no_noise_mean,
        measurements_std=no_noise_std)

  @classmethod
  def use_default_noise(cls, scale=1):
    """Returns the default observation noise parameters."""

    # There is no noise added to the actions, because the noise should be added
    # to the action after/as part of the power supply model as opposed to the
    # input to the power supply model.
    action_noise_mean = np.zeros((tcv_common.NUM_ACTIONS))
    action_noise_std = np.zeros((tcv_common.NUM_ACTIONS))

    meas_noise_mean = dict()
    for key, l in tcv_common.TCV_MEASUREMENTS.items():
      meas_noise_mean[key] = np.zeros((l,))
    meas_noise_std = dict(
        clint_vloop=np.array([0]),
        clint_rvloop=np.array([scale * 1e-4] * 37),
        bm=np.array([scale * 1e-4] * 38),
        IE=np.array([scale * 20] * 8),
        IF=np.array([scale * 5] * 8),
        IOH=np.array([scale * 20] *2),
        Bdot=np.array([scale * 0.05] * 20),
        DIOH=np.array([scale * 30]),
        FIR_FRINGE=np.array([0]),
        IG=np.array([scale * 2.5]),
        ONEMM=np.array([0]),
        vloop=np.array([scale * 0.3]),
        IPHI=np.array([0]),
        )
    return cls(
        action_mean=action_noise_mean,
        action_std=action_noise_std,
        measurements_mean=meas_noise_mean,
        measurements_std=meas_noise_std)

  def add_action_noise(self, action):
    errs = self._gen.normal(size=action.shape,
                            loc=self._action_mean,
                            scale=self._action_std)
    return action + errs

  def add_measurement_noise(self, measurement_vec):
    errs = self._gen.normal(size=measurement_vec.shape,
                            loc=self._meas_mean_vec,
                            scale=self._meas_std_vec)
    # Make the IOH measurements consistent. The "real" measurements are IOH
    # and DIOH, so use those.
    errs = tcv_common.measurements_to_dict(errs)
    errs["IOH"][1] = errs["IOH"][0] + errs["DIOH"][0]
    errs = tcv_common.dict_to_measurement(errs)
    return measurement_vec + errs
