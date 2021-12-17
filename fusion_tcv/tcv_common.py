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
"""Constants and general tooling for TCV plant."""

import collections
from typing import Sequence, Text
from dm_env import specs
import numpy as np

from fusion_tcv import named_array


DT = 1e-4  # ie 10kHz


# Below are general input/output specifications used for controllers that are
# run on hardware. This interface corresponds to the so called "KH hybrid"
# controller specification that is used in various experiments by EPFL. Hence,
# this interface definition contains measurements and actions not used in our
# tasks.

# Number of actions the environment is exposing. Includes dummy (FAST) action.
NUM_ACTIONS = 20

# Number of actuated coils in sim (without the dummy action coil).
NUM_COILS_ACTUATED = 19

# Current and voltage limits by coil type
# Note this are the limits used in the environments and are different
# from the 'machine engineering limits' (as used/exposed by FGE).
# We apply a safety factor (=< 1.0) to the engineering limits.
CURRENT_SAFETY_FACTOR = 0.8
ENV_COIL_MAX_CURRENTS = collections.OrderedDict(
    E=7500*CURRENT_SAFETY_FACTOR,
    F=7500*CURRENT_SAFETY_FACTOR,
    OH=26000*CURRENT_SAFETY_FACTOR,
    DUMMY=2000*CURRENT_SAFETY_FACTOR,
    G=2000*CURRENT_SAFETY_FACTOR)


# The g-coil has a saturation voltage that is tunable on a shot-by-shot basis.
# There is a deadband, where an action with absolute value of less than 8% of
# the saturation voltage is treated as zero.
ENV_G_COIL_SATURATION_VOLTAGE = 300
ENV_G_COIL_DEADBAND = ENV_G_COIL_SATURATION_VOLTAGE * 0.08

VOLTAGE_SAFETY_FACTOR = 1.0
ENV_COIL_MAX_VOLTAGE = collections.OrderedDict(
    E=1400*VOLTAGE_SAFETY_FACTOR,
    F=2200*VOLTAGE_SAFETY_FACTOR,
    OH=1400*VOLTAGE_SAFETY_FACTOR,
    DUMMY=400*VOLTAGE_SAFETY_FACTOR,
    # This value is also used to clip values for the internal controller,
    # and also to set the deadband voltage.
    G=ENV_G_COIL_SATURATION_VOLTAGE)


# Ordered actions send by a controller to the TCV.
TCV_ACTIONS = (
    'E_001', 'E_002', 'E_003', 'E_004', 'E_005', 'E_006', 'E_007', 'E_008',
    'F_001', 'F_002', 'F_003', 'F_004', 'F_005', 'F_006', 'F_007', 'F_008',
    'OH_001', 'OH_002',
    'DUMMY_001',  # GAS, ignored by TCV.
    'G_001'  # FAST
)
TCV_ACTION_INDICES = {n: i for i, n in enumerate(TCV_ACTIONS)}

TCV_ACTION_TYPES = collections.OrderedDict(
    E=8,
    F=8,
    OH=2,
    DUMMY=1,
    G=1,
)

# Map the TCV actions to ranges of indices in the array.
TCV_ACTION_RANGES = named_array.NamedRanges(TCV_ACTION_TYPES)

# The voltages seem not to be centered at 0, but instead near these values:
TCV_ACTION_OFFSETS = {
    'E_001': 6.79,
    'E_002': -10.40,
    'E_003': -1.45,
    'E_004': 0.18,
    'E_005': 11.36,
    'E_006': -0.95,
    'E_007': -4.28,
    'E_008': 44.22,
    'F_001': 38.49,
    'F_002': -2.94,
    'F_003': 5.58,
    'F_004': 1.09,
    'F_005': -36.63,
    'F_006': -9.18,
    'F_007': 5.34,
    'F_008': 10.53,
    'OH_001': -53.63,
    'OH_002': -14.76,
}

TCV_ACTION_DELAYS = {
    'E': [0.0005] * 8,
    'F': [0.0005] * 8,
    'OH': [0.0005] * 2,
    'G': [0.0001],
}


# Ordered measurements and their dimensions from to the TCV controller specs.
TCV_MEASUREMENTS = collections.OrderedDict(
    clint_vloop=1,  # Flux loop 1
    clint_rvloop=37,  # Difference of flux between loops 2-38 and flux loop 1
    bm=38,  # Magnetic field probes
    IE=8,  # E-coil currents
    IF=8,  # F-coil currents
    IOH=2,  # OH-coil currents
    Bdot=20,  # Selection of 20 time-derivatives of magnetic field probes (bm).
    DIOH=1,  # OH-coil currents difference: OH(0) - OH(1).
    FIR_FRINGE=1,  # Not used, ignore.
    IG=1,  # G-coil current
    ONEMM=1,  # Not used, ignore
    vloop=1,  # Flux loop 1 derivative
    IPHI=1,  # Current through the Toroidal Field coils. Constant. Ignore.
)

NUM_MEASUREMENTS = sum(TCV_MEASUREMENTS.values())
# map the TCV measurements to ranges of indices in the array
TCV_MEASUREMENT_RANGES = named_array.NamedRanges(TCV_MEASUREMENTS)

# Several of the measurement probes for the rvloops are broken. Add an extra key
# that allows us to only grab the usable ones
BROKEN_RVLOOP_IDXS = [9, 10, 11]

TCV_MEASUREMENT_RANGES.set_range('clint_rvloop_usable', [
    idx for i, idx in enumerate(TCV_MEASUREMENT_RANGES['clint_rvloop'])
    if i not in BROKEN_RVLOOP_IDXS])

TCV_COIL_CURRENTS_INDEX = [
    *TCV_MEASUREMENT_RANGES['IE'],
    *TCV_MEASUREMENT_RANGES['IF'],
    *TCV_MEASUREMENT_RANGES['IOH'],
    *TCV_MEASUREMENT_RANGES['IPHI'],  # In place of DUMMY.
    *TCV_MEASUREMENT_RANGES['IG'],
]


# References for what we want the agent to accomplish.
REF_RANGES = named_array.NamedRanges({
    'R': 2,
    'Z': 2,
    'Ip': 2,
    'kappa': 2,
    'delta': 2,
    'radius': 2,
    'lambda': 2,
    'diverted': 2,  # bool, must be diverted
    'limited': 2,  # bool, must be limited
    'shape_r': 32,
    'shape_z': 32,
    'x_points_r': 8,
    'x_points_z': 8,
    'legs_r': 16,  # Use for diverted/snowflake
    'legs_z': 16,
    'limit_point_r': 2,
    'limit_point_z': 2,
})

# Environments should use a consistent datatype for interacting with agents.
ENVIRONMENT_DATA_TYPE = np.float64


def observation_spec():
  """Observation spec for all TCV environments."""
  return {
      'references':
          specs.Array(
              shape=(REF_RANGES.size,),
              dtype=ENVIRONMENT_DATA_TYPE,
              name='references'),
      'measurements':
          specs.Array(
              shape=(TCV_MEASUREMENT_RANGES.size,),
              dtype=ENVIRONMENT_DATA_TYPE,
              name='measurements'),
      'last_action':
          specs.Array(
              shape=(TCV_ACTION_RANGES.size,),
              dtype=ENVIRONMENT_DATA_TYPE,
              name='last_action'),
  }


def measurements_to_dict(measurements):
  """Converts a single measurement vector or a time series to a dict.

  Args:
    measurements: A single measurement of size `NUM_MEASUREMENTS` or a time
      series, where the batch dimension is last, shape: (NUM_MEASUREMENTS, t).

  Returns:
    A dict mapping keys `TCV_MEASUREMENTS` to the corresponding measurements.

  """
  assert measurements.shape[0] == NUM_MEASUREMENTS
  measurements_dict = collections.OrderedDict()
  index = 0
  for key, dim in TCV_MEASUREMENTS.items():
    measurements_dict[key] = measurements[index:index + dim, ...]
    index += dim
  return measurements_dict


def dict_to_measurement(measurement_dict):
  """Converts a single measurement dict to a vector or time series.

  Args:
    measurement_dict: A dict with the measurement keys containing np arrays of
        size (meas_size, ...). The inner sizes all have to be the same.

  Returns:
    An array of size (num_measurements, ...)

  """
  assert len(measurement_dict) == len(TCV_MEASUREMENTS)
  # Grab the shape of the first array.
  shape = measurement_dict['clint_vloop'].shape

  out_shape = list(shape)
  out_shape[0] = NUM_MEASUREMENTS
  out_shape = tuple(out_shape)
  measurements = np.zeros((out_shape))
  index = 0
  for key, dim in TCV_MEASUREMENTS.items():
    dim = TCV_MEASUREMENTS[key]
    measurements[index:index + dim, ...] = measurement_dict[key]
    index += dim
  return measurements


def action_spec():
  return get_coil_spec(TCV_ACTIONS, ENV_COIL_MAX_VOLTAGE, ENVIRONMENT_DATA_TYPE)


def get_coil_spec(coil_names: Sequence[Text],
                  spec_mapping,
                  dtype=ENVIRONMENT_DATA_TYPE) -> specs.BoundedArray:
  """Maps specs indexed by coil type to coils given their type."""
  coil_max, coil_min = [], []
  for name in coil_names:
    # Coils names are <coil_type>_<coil_number>
    coil_type, _ = name.split('_')
    coil_max.append(spec_mapping[coil_type])
    coil_min.append(-spec_mapping[coil_type])
  return specs.BoundedArray(
      shape=(len(coil_names),), dtype=dtype, minimum=coil_min, maximum=coil_max)


INNER_LIMITER_R = 0.62400001
OUTER_LIMITER_R = 1.14179182
LIMITER_WIDTH = OUTER_LIMITER_R - INNER_LIMITER_R
LIMITER_RADIUS = LIMITER_WIDTH / 2
VESSEL_CENTER_R = INNER_LIMITER_R + LIMITER_RADIUS
