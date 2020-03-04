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
"""Combines predictions by pasting."""

import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import six
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from alphafold_casp13 import distogram_io
from alphafold_casp13 import parsers

flags.DEFINE_string("pickle_input_dir", None,
                    "Directory to read pickle distance histogram files from.")
flags.DEFINE_string("output_dir", None, "Directory to write chain RR files to.")
flags.DEFINE_string("tfrecord_path", "",
                    "If provided, construct the average weighted by number of "
                    "alignments.")
flags.DEFINE_string("crop_sizes", "64,128,256", "The crop sizes to use.")
flags.DEFINE_integer("crop_step", 32, "The step size for cropping.")
FLAGS = flags.FLAGS


def generate_domains(target, sequence, crop_sizes, crop_step):
  """Take fasta files and generate a domain definition for data generation."""
  logging.info("Generating crop domains for target %s", target)

  windows = [int(x) for x in crop_sizes.split(",")]
  num_residues = len(sequence)
  domains = []
  domains.append({"name": target, "description": (1, num_residues)})

  for window in windows:
    starts = list(range(0, num_residues - window, crop_step))
    # Append a last crop to ensure we get all the way to the end of the
    # sequence, even when num_residues - window is not divisible by crop_step.
    if num_residues >= window:
      starts += [num_residues - window]
    for start in starts:
      name = "%s-l%i_s%i" % (target, window, start)
      domains.append({"name": name, "description": (start + 1, start + window)})
  return domains


def get_weights(path):
  """Fetch all the weights from a TFRecord."""
  if not path:
    return {}
  logging.info("Getting weights from %s", path)
  weights = {}
  record_iterator = tf.python_io.tf_record_iterator(path=path)
  for serialized_tfexample in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(serialized_tfexample)

    domain_name = six.ensure_str(
        example.features.feature["domain_name"].bytes_list.value[0])
    weights[domain_name] = float(
        example.features.feature["num_alignments"].int64_list.value[0])
    logging.info("Weight %s: %d", domain_name, weights[domain_name])
  logging.info("Loaded %d weights", len(weights))
  return weights


def paste_distance_histograms(
    input_dir, output_dir, weights, crop_sizes, crop_step):
  """Paste together distograms for given domains of given targets and write.

  Domains distance histograms are 'pasted', meaning they are substituted
  directly into the contact map. The order is determined by the order in the
  domain definition file.

  Args:
    input_dir: String, path to directory containing chain and domain-level
      distogram files.
    output_dir: String, path to directory to write out chain-level distrogram
      files.
    weights: A dictionary with weights.
    crop_sizes: The crop sizes.
    crop_step: The step size for cropping.

  Raises:
    ValueError: if histogram parameters don't match.
  """
  tf.io.gfile.makedirs(output_dir)

  targets = tf.io.gfile.glob(os.path.join(input_dir, "*.pickle"))
  targets = [os.path.splitext(os.path.basename(t))[0] for t in targets]
  targets = set([t.split("-")[0] for t in targets])
  logging.info("Pasting distance histograms for %d targets", len(targets))

  for target in sorted(targets):
    logging.info("%s as chain", target)

    chain_pickle_path = os.path.join(input_dir, "%s.pickle" % target)
    distance_histogram_dict = parsers.parse_distance_histogram_dict(
        chain_pickle_path)

    combined_cmap = np.array(distance_histogram_dict["probs"])
    # Make the counter map 1-deep but still rank 3.
    counter_map = np.ones_like(combined_cmap[:, :, 0:1])

    sequence = distance_histogram_dict["sequence"]

    target_domains = generate_domains(
        target=target, sequence=sequence, crop_sizes=crop_sizes,
        crop_step=crop_step)

    # Paste in each domain.
    for domain in sorted(target_domains, key=lambda x: x["name"]):
      if domain["name"] == target:
        logging.info("Skipping %s as domain", target)
        continue

      if "," in domain["description"]:
        logging.info("Skipping multisegment domain %s",
                     domain["name"])
        continue

      crop_start, crop_end = domain["description"]

      domain_pickle_path = os.path.join(input_dir, "%s.pickle" % domain["name"])

      weight = weights.get(domain["name"], 1e9)

      logging.info("Pasting %s: %d-%d. weight: %f", domain_pickle_path,
                   crop_start, crop_end, weight)

      domain_distance_histogram_dict = parsers.parse_distance_histogram_dict(
          domain_pickle_path)
      for field in ["num_bins", "min_range", "max_range"]:
        if domain_distance_histogram_dict[field] != distance_histogram_dict[
            field]:
          raise ValueError("Field {} does not match {} {}".format(
              field,
              domain_distance_histogram_dict[field],
              distance_histogram_dict[field]))
      weight_matrix_size = crop_end - crop_start + 1
      weight_matrix = np.ones(
          (weight_matrix_size, weight_matrix_size), dtype=np.float32) * weight
      combined_cmap[crop_start - 1:crop_end, crop_start - 1:crop_end, :] += (
          domain_distance_histogram_dict["probs"] *
          np.expand_dims(weight_matrix, 2))
      counter_map[crop_start - 1:crop_end,
                  crop_start - 1:crop_end, 0] += weight_matrix

    # Broadcast across the histogram bins.
    combined_cmap /= counter_map

    # Write out full-chain cmap for folding.
    output_chain_pickle_path = os.path.join(output_dir,
                                            "{}.pickle".format(target))

    logging.info("Writing to %s", output_chain_pickle_path)

    distance_histogram_dict["probs"] = combined_cmap
    distance_histogram_dict["target"] = target

    # Save the distogram pickle file.
    distogram_io.save_distance_histogram_from_dict(
        output_chain_pickle_path, distance_histogram_dict)

    # Compute the contact map and save it as an RR file.
    contact_probs = distogram_io.contact_map_from_distogram(
        distance_histogram_dict)
    rr_path = os.path.join(output_dir, "%s.rr" % target)
    distogram_io.save_rr_file(
        filename=rr_path,
        probs=contact_probs,
        domain=target,
        sequence=distance_histogram_dict["sequence"])


def main(argv):
  del argv  # Unused.
  flags.mark_flag_as_required("pickle_input_dir")

  weights = get_weights(FLAGS.tfrecord_path)

  paste_distance_histograms(
      FLAGS.pickle_input_dir, FLAGS.output_dir, weights, FLAGS.crop_sizes,
      FLAGS.crop_step)

if __name__ == "__main__":
  app.run(main)
