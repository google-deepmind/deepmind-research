# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# WikiGraphs is licensed under the terms of the Creative Commons
# Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.
#
# WikiText-103 data (unchanged) is licensed by Salesforce.com, Inc. under the
# terms of the Creative Commons Attribution-ShareAlike 4.0 International
# (CC BY-SA 4.0) license. You can find details about CC BY-SA 4.0 at:
#
#     https://creativecommons.org/licenses/by-sa/4.0/legalcode
#
# Freebase data is licensed by Google LLC under the terms of the Creative
# Commons CC BY 4.0 license. You may obtain a copy of the License at:
#
#     https://creativecommons.org/licenses/by/4.0/legalcode
#
# ==============================================================================
"""Compute the bleu score on generated text and the ground truth."""

import math
import os
import pickle

from absl import app
from absl import flags
from absl import logging

import numpy as np

import utils


flags.DEFINE_string('checkpoint_dir', '/tmp/transformerXL',
                    'Checkpoint directory to load saved samples.')
flags.DEFINE_string('dataset', 'freebase2wikitext', 'Which dataset to the model'
                    ' is trained on, one of "wikitext", "freebase2wikitext".')

FLAGS = flags.FLAGS


def group_samples(samples, tokenizer):
  """Groups generated and ground truth texts."""
  groups = {}
  for i, row in enumerate(samples):
    gt = tokenizer.decode(row['ground_truth_text'])
    sample = tokenizer.decode(row['sample_tokens'])
    if gt not in groups:
      groups[gt] = (gt.split(), [sample.split()])
    else:
      groups[gt][-1].append(sample.split())
    if (i + 1) % 100 == 0:
      logging.info('Processed %d samples', i + 1)
  return groups


def eval_samples(raw_samples, tokenizer):
  """Evaluates generated samples."""
  gt_refs = []
  samples = []

  groups = group_samples(raw_samples, tokenizer)
  groups = list(groups.values())
  avg_group_size = np.mean([len(g[-1]) for g in groups])
  logging.info('Average samples per example: %.2f', avg_group_size)
  avg_group_size = int(math.ceil(avg_group_size))
  for i, (gt, s) in enumerate(groups):
    gt_refs.append(gt)
    idx = i % len(groups)
    samples.append(groups[idx][-1])

  gt_bleu, gt_n_grams = utils.compute_bleu(samples, gt_refs)

  logging.info('Processed %d samples in total.', sum([len(s) for s in samples]))
  flat_samples = []
  for s in samples:
    flat_samples.extend(s)
  logging.info('Average sample len: %.2f',
               np.mean([len(s) for s in flat_samples]))
  logging.info('Average ground-truth len: %.2f',
               np.mean([len(gt) for gt in gt_refs]))

  logging.info('Ground-truth BLEU: %6.2f, n-gram precision: (%s)',
               gt_bleu * 100,
               ', '.join(['%6.2f%%' % (s * 100) for s in gt_n_grams]))


def main(_):
  tokenizer = utils.init_tokenizer(FLAGS.dataset)
  checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, 'samples.pkl')
  logging.info('Loading samples from %s', checkpoint_dir)
  with open(checkpoint_dir, 'rb') as f:
    samples = pickle.load(f)['samples']
  eval_samples(samples, tokenizer)


if __name__ == '__main__':
  app.run(main)
