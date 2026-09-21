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

"""Main file for training VAE/AVAE."""

import enum
from absl import app
from absl import flags
import haiku as hk

from avae import data_iterators
from avae import decoders
from avae import encoders
from avae import train
from avae import vae


class Model(enum.Enum):
  vae = enum.auto()
  avae = enum.auto()


class EncoderArch(enum.Enum):
  color_mnist_mlp_encoder = 'ColorMnistMLPEncoder'


class DecoderArch(enum.Enum):
  color_mnist_mlp_decoder = 'ColorMnistMLPDecoder'


_DATASET = flags.DEFINE_enum_class(
    'dataset', data_iterators.Dataset.color_mnist, data_iterators.Dataset,
    'Dataset to train on')
_LATENT_DIM = flags.DEFINE_integer('latent_dim', 32,
                                   'Number of latent dimensions.')
_TRAIN_BATCH_SIZE = flags.DEFINE_integer('train_batch_size', 64,
                                         'Train batch size.')
_TEST_BATCH_SIZE = flags.DEFINE_integer('test_batch_size', 64,
                                        'Testing batch size.')
_TEST_EVERY = flags.DEFINE_integer('test_every', 1000,
                                   'Test every N iterations.')
_ITERATIONS = flags.DEFINE_integer('iterations', 102000,
                                   'Number of training iterations.')
_OBS_VAR = flags.DEFINE_float('obs_var', 0.5,
                              'Observation variance of the data. (Default 0.5)')


_MODEL = flags.DEFINE_enum_class('model', Model.avae, Model,
                                 'Model used for training.')
_RHO = flags.DEFINE_float('rho', 0.8, 'Rho parameter used with AVAE or SE.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
_RNG_SEED = flags.DEFINE_integer('rng_seed', 0,
                                 'Seed for random number generator.')
_CHECKPOINT_DIR = flags.DEFINE_string('checkpoint_dir', '/tmp/',
                                      'Directory for checkpointing.')
_CHECKPOINT_FILENAME = flags.DEFINE_string(
    'checkpoint_filename', 'color_mnist_avae_mlp', 'Checkpoint filename.')
_CHECKPOINT_EVERY = flags.DEFINE_integer(
    'checkpoint_every', 1000, 'Checkpoint every N steps.')
_ENCODER = flags.DEFINE_enum_class(
    'encoder', EncoderArch.color_mnist_mlp_encoder, EncoderArch,
    'Encoder class name.')
_DECODER = flags.DEFINE_enum_class(
    'decoder', DecoderArch.color_mnist_mlp_decoder, DecoderArch,
    'Decoder class name.')


def main(_):
  if _DATASET.value is data_iterators.Dataset.color_mnist:
    train_data_iterator = iter(
        data_iterators.ColorMnistDataIterator('train', _TRAIN_BATCH_SIZE.value))
    test_data_iterator = iter(
        data_iterators.ColorMnistDataIterator('test', _TEST_BATCH_SIZE.value))

  def _elbo_fun(input_data):
    if _ENCODER.value is EncoderArch.color_mnist_mlp_encoder:
      encoder = encoders.ColorMnistMLPEncoder(_LATENT_DIM.value)

    if _DECODER.value is DecoderArch.color_mnist_mlp_decoder:
      decoder = decoders.ColorMnistMLPDecoder(_OBS_VAR.value)

    vae_obj = vae.VAE(encoder, decoder, _RHO.value)

    if _MODEL.value is Model.vae:
      return vae_obj.vae_elbo(input_data, hk.next_rng_key())
    else:
      return vae_obj.avae_elbo(input_data, hk.next_rng_key())

  elbo_fun = hk.transform(_elbo_fun)

  extra_checkpoint_info = {
      'dataset': _DATASET.value.name,
      'encoder': _ENCODER.value.name,
      'decoder': _DECODER.value.name,
      'obs_var': _OBS_VAR.value,
      'rho': _RHO.value,
      'latent_dim': _LATENT_DIM.value,
  }

  train.train(
      train_data_iterator=train_data_iterator,
      test_data_iterator=test_data_iterator,
      elbo_fun=elbo_fun,
      learning_rate=_LEARNING_RATE.value,
      checkpoint_dir=_CHECKPOINT_DIR.value,
      checkpoint_filename=_CHECKPOINT_FILENAME.value,
      checkpoint_every=_CHECKPOINT_EVERY.value,
      test_every=_TEST_EVERY.value,
      iterations=_ITERATIONS.value,
      rng_seed=_RNG_SEED.value,
      extra_checkpoint_info=extra_checkpoint_info)


if __name__ == '__main__':
  app.run(main)
