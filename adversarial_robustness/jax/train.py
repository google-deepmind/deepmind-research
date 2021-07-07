# Copyright 2021 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a JAXline experiment to perform robust adversarial training."""

import functools

from absl import app
from absl import flags
from jaxline import platform
import tensorflow.compat.v2 as tf

from adversarial_robustness.jax import experiment

if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  try:
    tf.config.set_visible_devices([], 'GPU')  # Prevent TF from using the GPU.
  except tf.errors.NotFoundError:
    pass
  app.run(functools.partial(platform.main, experiment.Experiment))
