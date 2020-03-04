# Lint as: python3.
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code to run distogram inference."""

import collections
import os
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import six
import sonnet as snt
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from alphafold_casp13 import config_dict
from alphafold_casp13 import contacts_experiment
from alphafold_casp13 import distogram_io
from alphafold_casp13 import secstruct

flags.DEFINE_string('config_path', None, 'Path of the JSON config file.')
flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path for evaluation.')
flags.DEFINE_boolean('cpu', False, 'Force onto CPU.')
flags.DEFINE_string('output_path', None,
                    'Base path where all output files will be saved to.')
flags.DEFINE_string('eval_sstable', None,
                    'Path of the SSTable to read the input tf.Examples from.')
flags.DEFINE_string('stats_file', None,
                    'Path of the statistics file to use for normalization.')

FLAGS = flags.FLAGS


# A named tuple to store the outputs of a single prediction run.
Prediction = collections.namedtuple(
    'Prediction', [
        'single_message',  # A debugging message.
        'num_crops_local',  # The number of crops used to make this prediction.
        'sequence',  # The amino acid sequence.
        'filebase',  # The chain name. All output files will use this name.
        'softmax_probs',  # Softmax of the distogram.
        'ss',  # Secondary structure prediction.
        'asa',  # ASA prediction.
        'torsions',  # Torsion prediction.
    ])


def evaluate(crop_size_x, crop_size_y, feature_normalization, checkpoint_path,
             normalization_exclusion, eval_config, network_config):
  """Main evaluation loop."""
  experiment = contacts_experiment.Contacts(
      tfrecord=eval_config.eval_sstable,
      stats_file=eval_config.stats_file,
      network_config=network_config,
      crop_size_x=crop_size_x,
      crop_size_y=crop_size_y,
      feature_normalization=feature_normalization,
      normalization_exclusion=normalization_exclusion)

  checkpoint = snt.get_saver(experiment.model, collections=[
      tf.GraphKeys.GLOBAL_VARIABLES,
      tf.GraphKeys.MOVING_AVERAGE_VARIABLES])

  with tf.train.SingularMonitoredSession(hooks=[]) as sess:
    logging.info('Restoring from checkpoint %s', checkpoint_path)
    checkpoint.restore(sess, checkpoint_path)

    logging.info('Writing output to %s', eval_config.output_path)
    eval_begin_time = time.time()
    _run_evaluation(sess=sess,
                    experiment=experiment,
                    eval_config=eval_config,
                    output_dir=eval_config.output_path,
                    min_range=network_config.min_range,
                    max_range=network_config.max_range,
                    num_bins=network_config.num_bins,
                    torsion_bins=network_config.torsion_bins)
    logging.info('Finished eval %.1fs', (time.time() - eval_begin_time))


def _run_evaluation(
    sess, experiment, eval_config, output_dir, min_range, max_range, num_bins,
    torsion_bins):
  """Evaluate a contact map by aggregating crops.

  Args:
    sess: A tf.train.Session.
    experiment: An experiment class.
    eval_config: A config dict of eval parameters.
    output_dir: Directory to save the predictions to.
    min_range: The minimum range in Angstroms to consider in distograms.
    max_range: The maximum range in Angstroms to consider in distograms, see
      num_bins below for clarification.
    num_bins: The number of bins in the distance histogram being predicted.
      We divide the min_range--(min_range + max_range) Angstrom range into this
      many bins.
    torsion_bins: The number of bins the torsion angles are discretised into.
  """
  tf.io.gfile.makedirs(os.path.join(output_dir, 'pickle_files'))

  logging.info('Eval config is %s\nnum_bins: %d', eval_config, num_bins)
  num_examples = 0
  num_crops = 0
  start_all_time = time.time()

  # Either do the whole test set, or up to a specified limit.
  max_examples = experiment.num_eval_examples
  if eval_config.max_num_examples > 0:
    max_examples = min(max_examples, eval_config.max_num_examples)

  while num_examples < max_examples:
    one_prediction = compute_one_prediction(
        num_examples, experiment, sess, eval_config, num_bins, torsion_bins)

    single_message = one_prediction.single_message
    num_crops_local = one_prediction.num_crops_local
    sequence = one_prediction.sequence
    filebase = one_prediction.filebase
    softmax_probs = one_prediction.softmax_probs
    ss = one_prediction.ss
    asa = one_prediction.asa
    torsions = one_prediction.torsions

    num_examples += 1
    num_crops += num_crops_local

    # Save the output files.
    filename = os.path.join(output_dir,
                            'pickle_files', '%s.pickle' % filebase)
    distogram_io.save_distance_histogram(
        filename, softmax_probs, filebase, sequence,
        min_range=min_range, max_range=max_range, num_bins=num_bins)

    if experiment.model.torsion_multiplier > 0:
      torsions_dir = os.path.join(output_dir, 'torsions')
      tf.io.gfile.makedirs(torsions_dir)
      distogram_io.save_torsions(torsions_dir, filebase, sequence, torsions)

    if experiment.model.secstruct_multiplier > 0:
      ss_dir = os.path.join(output_dir, 'secstruct')
      tf.io.gfile.makedirs(ss_dir)
      secstruct.save_secstructs(ss_dir, filebase, None, sequence, ss)

    if experiment.model.asa_multiplier > 0:
      asa_dir = os.path.join(output_dir, 'asa')
      tf.io.gfile.makedirs(asa_dir)
      secstruct.save_secstructs(asa_dir, filebase, None, sequence,
                                np.expand_dims(asa, 1), label='Deepmind 2D ASA')

    time_spent = time.time() - start_all_time
    logging.info(
        'Evaluate %d examples, %d crops %.1f crops/ex. '
        'Took %.1fs, %.3f s/example %.3f crops/s\n%s',
        num_examples, num_crops, num_crops / float(num_examples), time_spent,
        time_spent / num_examples, num_crops / time_spent, single_message)

  logging.info('Tested on %d', num_examples)


def compute_one_prediction(
    num_examples, experiment, sess, eval_config, num_bins, torsion_bins):
  """Find the contact map for a single domain."""
  num_crops_local = 0
  debug_steps = 0
  start = time.time()
  output_fetches = {'probs': experiment.eval_probs}
  output_fetches['softmax_probs'] = experiment.eval_probs_softmax
  # Add the auxiliary outputs if present.
  experiment.model.update_crop_fetches(output_fetches)
  # Get data.
  batch = experiment.get_one_example(sess)
  length = batch['sequence_lengths'][0]
  batch_size = batch['sequence_lengths'].shape[0]
  domain = batch['domain_name'][0][0].decode('utf-8')
  chain = batch['chain_name'][0][0].decode('utf-8')
  filebase = domain or chain
  sequence = six.ensure_str(batch['sequences'][0][0])
  logging.info('SepWorking on %d %s %s %d', num_examples, domain, chain, length)
  inputs_1d = batch['inputs_1d']
  if 'residue_index' in batch:
    logging.info('Getting residue_index from features')
    residue_index = np.squeeze(
        batch['residue_index'], axis=2).astype(np.int32)
  else:
    logging.info('Generating residue_index')
    residue_index = np.tile(np.expand_dims(
        np.arange(length, dtype=np.int32), 0), [batch_size, 1])
  assert batch_size == 1
  num_examples += batch_size
  # Crops.
  prob_accum = np.zeros((length, length, 2))
  ss_accum = np.zeros((length, 8))
  torsions_accum = np.zeros((length, torsion_bins**2))
  asa_accum = np.zeros((length,))
  weights_1d_accum = np.zeros((length,))
  softmax_prob_accum = np.zeros((length, length, num_bins), dtype=np.float32)

  crop_size_x = experiment.crop_size_x
  crop_step_x = crop_size_x // eval_config.crop_shingle_x
  crop_size_y = experiment.crop_size_y
  crop_step_y = crop_size_y // eval_config.crop_shingle_y

  prob_weights = 1
  if eval_config.pyramid_weights > 0:
    sx = np.expand_dims(np.linspace(1.0 / crop_size_x, 1, crop_size_x), 1)
    sy = np.expand_dims(np.linspace(1.0 / crop_size_y, 1, crop_size_y), 0)
    prob_weights = np.minimum(np.minimum(sx, np.flipud(sx)),
                              np.minimum(sy, np.fliplr(sy)))
    prob_weights /= np.max(prob_weights)
    prob_weights = np.minimum(prob_weights, eval_config.pyramid_weights)
  logging.log_first_n(logging.INFO, 'Crop: %dx%d step %d,%d pyr %.2f',
                      debug_steps,
                      crop_size_x, crop_size_y,
                      crop_step_x, crop_step_y, eval_config.pyramid_weights)
  # Accumulate all crops, starting and ending half off the square.
  for i in range(-crop_size_x // 2, length - crop_size_x // 2, crop_step_x):
    for j in range(-crop_size_y // 2, length - crop_size_y // 2, crop_step_y):
      # The ideal crop.
      patch = compute_one_patch(
          sess, experiment, output_fetches, inputs_1d, residue_index,
          prob_weights, batch, length, i, j, crop_size_x, crop_size_y)
      # Assemble the crops into a final complete prediction.
      ic = max(0, i)
      jc = max(0, j)
      ic_to = ic + patch['prob'].shape[1]
      jc_to = jc + patch['prob'].shape[0]
      prob_accum[jc:jc_to, ic:ic_to, 0] += patch['prob'] * patch['weight']
      prob_accum[jc:jc_to, ic:ic_to, 1] += patch['weight']
      softmax_prob_accum[jc:jc_to, ic:ic_to, :] += (
          patch['softmax'] * np.expand_dims(patch['weight'], 2))
      weights_1d_accum[jc:jc_to] += 1
      weights_1d_accum[ic:ic_to] += 1
      if 'asa_x' in patch:
        asa_accum[ic:ic + patch['asa_x'].shape[0]] += np.squeeze(
            patch['asa_x'], axis=1)
        asa_accum[jc:jc + patch['asa_y'].shape[0]] += np.squeeze(
            patch['asa_y'], axis=1)
      if 'ss_x' in patch:
        ss_accum[ic:ic + patch['ss_x'].shape[0]] += patch['ss_x']
        ss_accum[jc:jc + patch['ss_y'].shape[0]] += patch['ss_y']
      if 'torsions_x' in patch:
        torsions_accum[
            ic:ic + patch['torsions_x'].shape[0]] += patch['torsions_x']
        torsions_accum[
            jc:jc + patch['torsions_y'].shape[0]] += patch['torsions_y']
      num_crops_local += 1
  single_message = (
      'Constructed %s len %d from %d chunks [%d, %d x %d, %d] '
      'in %5.1fs' % (
          filebase, length, num_crops_local,
          crop_size_x, crop_step_x, crop_size_y, crop_step_y,
          time.time() - start))
  logging.info(single_message)
  logging.info('prob_accum[:, :, 1]: %s', prob_accum[:, :, 1])
  assert (prob_accum[:, :, 1] > 0.0).all()
  probs = prob_accum[:, :, 0] / prob_accum[:, :, 1]
  softmax_probs = softmax_prob_accum[:, :, :] / prob_accum[:, :, 1:2]

  asa_accum /= weights_1d_accum
  ss_accum /= np.expand_dims(weights_1d_accum, 1)
  torsions_accum /= np.expand_dims(weights_1d_accum, 1)

  # The probs are symmetrical.
  probs = (probs + probs.transpose()) / 2
  if num_bins > 1:
    softmax_probs = (softmax_probs + np.transpose(
        softmax_probs, axes=[1, 0, 2])) / 2
  return Prediction(
      single_message=single_message,
      num_crops_local=num_crops_local,
      sequence=sequence,
      filebase=filebase,
      softmax_probs=softmax_probs,
      ss=ss_accum,
      asa=asa_accum,
      torsions=torsions_accum)


def compute_one_patch(sess, experiment, output_fetches, inputs_1d,
                      residue_index, prob_weights, batch, length, i, j,
                      crop_size_x, crop_size_y):
  """Compute the output predictions for a single crop."""
  # Note that these are allowed to go off the end of the protein.
  end_x = i + crop_size_x
  end_y = j + crop_size_y
  crop_limits = np.array([[i, end_x, j, end_y]], dtype=np.int32)
  ic = max(0, i)
  jc = max(0, j)
  end_x_cropped = min(length, end_x)
  end_y_cropped = min(length, end_y)
  prepad_x = max(0, -i)
  prepad_y = max(0, -j)
  postpad_x = end_x - end_x_cropped
  postpad_y = end_y - end_y_cropped

  # Precrop the 2D features:
  inputs_2d = np.pad(batch['inputs_2d'][
      :, jc:end_y, ic:end_x, :],
                     [[0, 0],
                      [prepad_y, postpad_y],
                      [prepad_x, postpad_x],
                      [0, 0]], mode='constant')
  assert inputs_2d.shape[1] == crop_size_y
  assert inputs_2d.shape[2] == crop_size_x

  # Generate the corresponding crop, but it might be truncated.
  cxx = batch['inputs_2d'][:, ic:end_x, ic:end_x, :]
  cyy = batch['inputs_2d'][:, jc:end_y, jc:end_y, :]
  if cxx.shape[1] < inputs_2d.shape[1]:
    cxx = np.pad(cxx, [[0, 0],
                       [prepad_x, max(0, i + crop_size_y - length)],
                       [prepad_x, postpad_x],
                       [0, 0]], mode='constant')
  assert cxx.shape[1] == crop_size_y
  assert cxx.shape[2] == crop_size_x
  if cyy.shape[2] < inputs_2d.shape[2]:
    cyy = np.pad(cyy, [[0, 0],
                       [prepad_y, postpad_y],
                       [prepad_y, max(0, j + crop_size_x - length)],
                       [0, 0]], mode='constant')
  assert cyy.shape[1] == crop_size_y
  assert cyy.shape[2] == crop_size_x
  inputs_2d = np.concatenate([inputs_2d, cxx, cyy], 3)

  output_results = sess.run(output_fetches, feed_dict={
      experiment.inputs_1d_placeholder: inputs_1d,
      experiment.residue_index_placeholder: residue_index,
      experiment.inputs_2d_placeholder: inputs_2d,
      experiment.crop_placeholder: crop_limits,
  })
  # Crop out the "live" region of the probs.
  prob_patch = output_results['probs'][
      0, prepad_y:crop_size_y - postpad_y,
      prepad_x:crop_size_x - postpad_x]
  weight_patch = prob_weights[prepad_y:crop_size_y - postpad_y,
                              prepad_x:crop_size_x - postpad_x]
  patch = {'prob': prob_patch, 'weight': weight_patch}

  if 'softmax_probs' in output_results:
    patch['softmax'] = output_results['softmax_probs'][
        0, prepad_y:crop_size_y - postpad_y,
        prepad_x:crop_size_x - postpad_x]
  if 'secstruct_probs' in output_results:
    patch['ss_x'] = output_results['secstruct_probs'][
        0, prepad_x:crop_size_x - postpad_x]
    patch['ss_y'] = output_results['secstruct_probs'][
        0, crop_size_x + prepad_y:crop_size_x + crop_size_y - postpad_y]
  if 'torsion_probs' in output_results:
    patch['torsions_x'] = output_results['torsion_probs'][
        0, prepad_x:crop_size_x - postpad_x]
    patch['torsions_y'] = output_results['torsion_probs'][
        0, crop_size_x + prepad_y:crop_size_x + crop_size_y - postpad_y]
  if 'asa_output' in output_results:
    patch['asa_x'] = output_results['asa_output'][
        0, prepad_x:crop_size_x - postpad_x]
    patch['asa_y'] = output_results['asa_output'][
        0, crop_size_x + prepad_y:crop_size_x + crop_size_y - postpad_y]
  return patch


def main(argv):
  del argv  # Unused.

  logging.info('Loading a JSON config from: %s', FLAGS.config_path)
  with tf.io.gfile.GFile(FLAGS.config_path, 'r') as f:
    config = config_dict.ConfigDict.from_json(f.read())

  # Redefine the relevant output fields.
  if FLAGS.eval_sstable:
    config.eval_config.eval_sstable = FLAGS.eval_sstable
  if FLAGS.stats_file:
    config.eval_config.stats_file = FLAGS.stats_file
  if FLAGS.output_path:
    config.eval_config.output_path = FLAGS.output_path

  with tf.device('/cpu:0' if FLAGS.cpu else None):
    evaluate(checkpoint_path=FLAGS.checkpoint_path, **config)


if __name__ == '__main__':
  app.run(main)
