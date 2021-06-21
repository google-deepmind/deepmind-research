# Copyright 2020 DeepMind Technologies Limited
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

"""Setup for pip package."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['absl-py', 'dm_control', 'numpy', 'dm_env']

setup(
    name='box_arrangement',
    version='0.1',
    description=('Sparse reward tasks involving moving and pushing boxes to'
                 ' targets.'),
    url='https://github.com/deepmind/deepmind-research/box_arrangement',
    author='DeepMind',
    author_email='dhruvat@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
)
