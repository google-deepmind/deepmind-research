# Copyright 2020 DeepMind Technologies Limited.
#
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
"""Setup for pip package."""
from setuptools import setup

REQUIRED_PACKAGES = (
    "dm_hamiltonian_dynamics_suite@git+https://github.com/deepmind/dm_hamiltonian_dynamics_suite",  # pylint: disable=line-too-long.
    "absl-py>=0.12.0",
    "numpy>=1.16.4",
    "scikit-learn>=1.0",
    "typing>=3.7.4.3",
    "jax==0.2.20",
    "jaxline==0.0.3",
    "distrax==0.0.2",
    "optax==0.0.6",
    "dm-haiku==0.0.3",
)

LONG_DESCRIPTION = "\n".join([
    "A codebase containing the implementation of the following models:",
    "Hamiltonian Generative Network (HGN)",
    "Lagrangian Generative Network (LGN)",
    "Neural ODE",
    "Recurrent Generative Network (RGN)",
    "and RNN, LSTM and GRU.",
    "This is code accompanying the publication of:"
])


setup(
    name="physics_inspired_models",
    version="0.0.1",
    description="Implementation of multiple physically inspired models.",
    long_description=LONG_DESCRIPTION,
    url="https://github.com/deepmind/deepmind-research/physics_inspired_models",
    author="DeepMind",
    package_dir={"physics_inspired_models": "."},
    packages=["physics_inspired_models", "physics_inspired_models.models"],
    install_requires=REQUIRED_PACKAGES,
    platforms=["any"],
    license="Apache License, Version 2.0",
)
