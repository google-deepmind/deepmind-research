# Copyright 2019 DeepMind Technologies Limited
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


REQUIRED_PACKAGES = ['numpy', 'dm-sonnet==1.35', 'tensorflow==1.15.2',
                     'tensorflow-probability==0.7.0']

setup(
    name='hpu_net',
    version='0.1',
    description='A library for the Hierarchical Probabilistic U-Net model.',
    url='https://github.com/deepmind/deepmind-research/hierarchical_probabilistic_unet',
    author='DeepMind',
    author_email='no-reply@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
)
