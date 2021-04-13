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
    "absl-py",
    "dataclasses",
    "jax",
    "networkx",
    "numpy",
    "ordered-set",
    "typing",
)

LONG_DESCRIPTION = "\n".join([
    "Kronecker-Factored Approximate Curvature (K-FAC) optimizer implemented in "
    "JAX.",
    "",
    "Accompanying code for 'Better, Faster Fermionic Neural Networks'",
    "James S. Spencer, David Pfau, Aleksandar Botev, and W. M. C. Foulkes.",
    "https://arxiv.org/abs/2011.07125.",
])


setup(
    name="kfac_ferminet_alpha",
    version="0.0.1",
    description="A K-FAC optimizer implemented in JAX",
    long_description=LONG_DESCRIPTION,
    url="https://github.com/deepmind/deepmind-research/kfac_ferminet_alpha",
    author="DeepMind",
    package_dir={"kfac_ferminet_alpha": "."},
    packages=["kfac_ferminet_alpha"],
    install_requires=REQUIRED_PACKAGES,
    platforms=["any"],
    license="Apache License, Version 2.0",
)
