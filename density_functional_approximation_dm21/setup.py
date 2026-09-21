# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for DM21 functionals interface to PySCF."""

from setuptools import setup

REQUIRED_PACKAGES = [
    'absl-py',
    'attrs',
    # Note PySCF 1.7.6 and older do not support h5py 3.3.0:
    # https://github.com/pyscf/pyscf/issues/1016
    'h5py',
    'numpy',
    # Note DM21 functionals are compatible with PySCF 1.7.6 if an older version
    # of h5py is used.
    'pyscf>=2.0',
    'tensorflow',
    'tensorflow_hub',
]
CHECKPOINT_DATA = ['checkpoints/DM21*/*.pb', 'checkpoints/DM21*/variables/*']

setup(
    name='density_functional_approximation_dm21',
    version='0.1',
    description='An interface to PySCF for the DM21 functionals.',
    url='https://github.com/deepmind/deepmind-research/density_functional_approximation_dm21',
    author='DeepMind',
    author_email='no-reply@google.com',
    # Contained modules and scripts.
    packages=['density_functional_approximation_dm21'],
    package_data={
        'density_functional_approximation_dm21': CHECKPOINT_DATA,
    },
    scripts=['density_functional_approximation_dm21/export_saved_model.py'],
    install_requires=REQUIRED_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
    extras_require={'testing': ['pytest', 'scipy']},
)
