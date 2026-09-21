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

"""Setup BYOL for pip package."""

import setuptools

setuptools.setup(
    name='byol',
    description='Bootstrap Your Own Latents',
    long_description=open('README.md').read(),
    author='DeepMind',
    author_email='no-reply@google.com',
    url='https://github.com/deepmind/deepmind-research/byol',
    install_requires=[
        'chex',
        'dm-acme',
        'dm-haiku',
        'dm-tree',
        'jax',
        'jaxlib',
        'numpy>=1.16',
        'optax',
        'tensorflow',
        'tensorflow-datasets',
    ],
    package_dir={'byol': ''},
    py_modules=[
        'byol.byol_experiment', 'byol.eval_experiment', 'byol.main_loop'
    ],
    packages=['byol.configs', 'byol.utils'])
