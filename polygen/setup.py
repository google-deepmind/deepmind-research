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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['numpy', 'dm-sonnet==1.36', 'tensorflow==1.14',
                     'tensor2tensor==1.15', 'networkx', 'matplotlib', 'six']

setup(
    name='polygen',
    version='0.1',
    description='A library for PolyGen: An Autoregressive Generative Model of 3D Meshes.',
    url='https://github.com/deepmind/deepmind-research/polygen',
    author='DeepMind',
    author_email='no-reply@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
)
