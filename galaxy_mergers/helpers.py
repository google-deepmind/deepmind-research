# Copyright 2021 DeepMind Technologies Limited.
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

"""Helpers for a galaxy merger model evaluation."""

import glob
import os
from astropy import cosmology
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow.compat.v2 as tf


def restore_checkpoint(checkpoint_dir, experiment):
  checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name='global_step')
  checkpoint = tf.train.Checkpoint(
      _global_step_=global_step, **experiment.checkpoint_items)
  checkpoint.restore(checkpoint_path)


def sum_average_transformed_mu_and_sigma(mu, log_sigma_sq):
  """Computes <mu>, var(mu) + <var> in transformed representation.

  This corresponds to assuming that the output distribution is a sum of
  Gaussian and computing the mean and variance of the resulting (non-Gaussian)
  distribution.

  Args:
    mu: Tensor of shape [B, ...] representing the means of the input
      distributions.
    log_sigma_sq: Tensor of shape [B, ...] representing log(sigma**2) of the
      input distributions. Can be None, in which case the variance is assumed
      to be zero.

  Returns:
    mu: Tensor of shape [...] representing the means of the output
      distributions.
    log_sigma_sq: Tensor of shape [...] representing log(sigma**2) of the
      output distributions.
  """
  av_mu = tf.reduce_mean(mu, axis=0)
  var_mu = tf.math.reduce_std(mu, axis=0)**2
  if log_sigma_sq is None:
    return av_mu, tf.math.log(var_mu)
  max_log_sigma_sq = tf.reduce_max(log_sigma_sq, axis=0)
  log_sigma_sq -= max_log_sigma_sq
  # (sigma/sigma_0)**2
  sigma_sq = tf.math.exp(log_sigma_sq)
  # (<sigma**2>)/sigma_0**2 (<1)
  av_sigma_sq = tf.reduce_mean(sigma_sq, axis=0)
  # (<sigma**2> + var(mu))/sigma_0**2
  av_sigma_sq += var_mu * tf.math.exp(-max_log_sigma_sq)
  # log(<sigma**2> + var(mu))
  log_av_sigma_sq = tf.math.log(av_sigma_sq) + max_log_sigma_sq
  return av_mu, log_av_sigma_sq


def aggregate_regression_ensemble(logits_or_times, ensemble_size,
                                  use_uncertainty, test_time_ensembling):
  """Aggregate output of model ensemble."""
  out_shape = logits_or_times.shape.as_list()[1:]
  logits_or_times = tf.reshape(logits_or_times, [ensemble_size, -1] + out_shape)
  mus = logits_or_times[..., 0]
  log_sigma_sqs = logits_or_times[..., -1] if use_uncertainty else None

  if test_time_ensembling == 'sum':
    mu, log_sigma_sq = sum_average_transformed_mu_and_sigma(mus, log_sigma_sqs)
  elif test_time_ensembling == 'none':
    mu = mus[0]
    log_sigma_sq = log_sigma_sqs[0] if use_uncertainty else None
  else:
    raise ValueError('Unexpected test_time_ensembling')
  return mu, log_sigma_sq


def aggregate_classification_ensemble(logits_or_times, ensemble_size,
                                      test_time_ensembling):
  """Averages the output logits across models in the ensemble."""
  out_shape = logits_or_times.shape.as_list()[1:]
  logits = tf.reshape(logits_or_times, [ensemble_size, -1] + out_shape)

  if test_time_ensembling == 'sum':
    logits = tf.reduce_mean(logits, axis=0)
    return logits, None
  elif test_time_ensembling == 'none':
    return logits, None
  else:
    raise ValueError('Unexpected test_time_ensembling')


def unpack_evaluator_output(data, return_seq_info=False, return_redshift=False):
  """Unpack evaluator.run_model_on_dataset output."""
  mus = np.array(data[1]['mu']).flatten()
  sigmas = np.array(data[1]['sigma']).flatten()
  regression_targets = np.array(data[1]['regression_targets']).flatten()
  outputs = [mus, sigmas, regression_targets]

  if return_seq_info:
    seq_ids = np.array(data[2][0]).flatten()
    seq_ids = np.array([seq_id.decode('UTF-8') for seq_id in seq_ids])
    time_idxs = np.array(data[2][1]).flatten()
    axes = np.array(data[2][2]).flatten()
    outputs += [seq_ids, axes, time_idxs]

  if return_redshift:
    redshifts = np.array(data[2][6]).flatten()
    outputs += [redshifts]

  return outputs


def process_data_into_myrs(redshifts, *data_lists):
  """Converts normalized time to virial time using Planck cosmology."""
  # small hack to avoid build tools not recognizing non-standard trickery
  #   done in the astropy library:
  #   https://github.com/astropy/astropy/blob/master/astropy/cosmology/core.py#L3290
  #   that dynamically generates and imports new classes.
  planck13 = getattr(cosmology, 'Plank13')
  hubble_constants = planck13.H(redshifts)  # (km/s)/megaparsec
  inv_hubble_constants = 1/hubble_constants  # (megaparsec*s) / km
  megaparsec_to_km = 1e19*3.1
  seconds_to_gigayears = 1e-15/31.556
  conversion_factor = megaparsec_to_km * seconds_to_gigayears
  hubble_time_gigayears = conversion_factor * inv_hubble_constants

  hubble_to_virial_time = 0.14  # approximate simulation-based conversion factor
  virial_dyn_time = hubble_to_virial_time*hubble_time_gigayears.value
  return [data_list*virial_dyn_time for data_list in data_lists]


def print_rmse_and_class_accuracy(mus, regression_targets, redshifts):
  """Convert to virial dynamical time and print stats."""
  time_pred, time_gt = process_data_into_myrs(
      redshifts, mus, regression_targets)
  time_sq_errors = (time_pred-time_gt)**2
  rmse = np.sqrt(np.mean(time_sq_errors))
  labels = regression_targets > 0
  class_preds = mus > 0
  accuracy = sum((labels == class_preds).astype(np.int8)) / len(class_preds)

  print(f'95% Error: {np.percentile(np.sqrt(time_sq_errors), 95)}')
  print(f'RMSE: {rmse}')
  print(f'Classification Accuracy: {accuracy}')


def print_stats(vec, do_print=True):
  fvec = vec.flatten()
  if do_print:
    print(len(fvec), min(fvec), np.mean(fvec), np.median(fvec), max(fvec))
  return (len(fvec), min(fvec), np.mean(fvec), np.median(fvec), max(fvec))


def get_image_from_fits(base_dir, seq='475_31271', time='497', axis=2):
  """Read *.fits galaxy image from directory."""
  axis_map = {0: 'x', 1: 'y', 2: 'z'}
  fits_glob = f'{base_dir}/{seq}/fits_of_flux_psf/{time}/*_{axis_map[axis]}_*.fits'

  def get_freq_from_path(p):
    return int(p.split('/')[-1].split('_')[2][1:])

  fits_image_paths = sorted(glob.glob(fits_glob), key=get_freq_from_path)
  assert len(fits_image_paths) == 7
  combined_frequencies = []
  for fit_path in fits_image_paths:
    with open(fit_path, 'rb') as f:
      fits_data = np.array(fits.open(f)[0].data.astype(np.float32))
      combined_frequencies.append(fits_data)
  fits_image = np.transpose(np.array(combined_frequencies), (1, 2, 0))
  return fits_image


def stack_desired_galaxy_images(base_dir, seq, n_time_slices):
  """Searth through galaxy image directory gathering images."""
  fits_sequence_dir = os.path.join(base_dir, seq, 'fits_of_flux_psf')
  all_times_for_seq = os.listdir(fits_sequence_dir)
  hop = (len(all_times_for_seq)-1)//(n_time_slices-1)
  desired_time_idxs = [k*hop for k in range(n_time_slices)]

  all_imgs = []
  for j in desired_time_idxs:
    time = all_times_for_seq[j]
    img = get_image_from_fits(base_dir=base_dir, seq=seq, time=time, axis=2)
    all_imgs.append(img)

  min_img_size = min([img.shape[0] for img in all_imgs])
  return all_imgs, min_img_size


def draw_galaxy_image(image, target_size=None, color_map='viridis'):
  normalized_image = image / max(image.flatten())
  color_map = plt.get_cmap(color_map)
  colored_image = color_map(normalized_image)[:, :, :3]
  colored_image = (colored_image * 255).astype(np.uint8)
  colored_image = Image.fromarray(colored_image, mode='RGB')
  if target_size:
    colored_image = colored_image.resize(target_size, Image.ANTIALIAS)
  return colored_image


def collect_merger_sequence(ds, seq=b'370_11071', n_examples_to_sift=5000):
  images, targets, redshifts = [], [], []
  for i, all_inputs in enumerate(ds):
    if all_inputs[4][0].numpy() == seq:
      images.append(all_inputs[0][0].numpy())
      targets.append(all_inputs[2][0].numpy())
      redshifts.append(all_inputs[10][0].numpy())
    if i > n_examples_to_sift: break
  return np.squeeze(images), np.squeeze(targets), np.squeeze(redshifts)


def take_samples(sample_idxs, *data_lists):
  return [np.take(l, sample_idxs, axis=0) for l in data_lists]
