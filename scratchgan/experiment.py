# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited and Google LLC
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
"""Training script for ScratchGAN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.io import gfile
from scratchgan import discriminator_nets
from scratchgan import eval_metrics
from scratchgan import generators
from scratchgan import losses
from scratchgan import reader
from scratchgan import utils

flags.DEFINE_string("dataset", "emnlp2017", "Dataset.")
flags.DEFINE_integer("batch_size", 512, "Batch size")
flags.DEFINE_string("gen_type", "lstm", "Generator type.")
flags.DEFINE_string("disc_type", "lstm", "Discriminator type.")
flags.DEFINE_string("disc_loss_type", "ce", "Loss type.")
flags.DEFINE_integer("gen_feature_size", 512, "Generator feature size.")
flags.DEFINE_integer("disc_feature_size", 512, "Discriminator feature size.")
flags.DEFINE_integer("num_layers_gen", 2, "Number of generator layers.")
flags.DEFINE_integer("num_layers_disc", 1, "Number of discriminator layers.")
flags.DEFINE_bool("layer_norm_gen", False, "Layer norm generator.")
flags.DEFINE_bool("layer_norm_disc", True, "Layer norm discriminator.")
flags.DEFINE_float("gen_input_dropout", 0.0, "Input dropout generator.")
flags.DEFINE_float("gen_output_dropout", 0.0, "Input dropout discriminator.")
flags.DEFINE_float("l2_gen", 0.0, "L2 regularization generator.")
flags.DEFINE_float("l2_disc", 1e-6, "L2 regularization discriminator.")
flags.DEFINE_float("disc_dropout", 0.1, "Dropout discriminator")
flags.DEFINE_integer("trainable_embedding_size", 64,
                     "Size of trainable embedding.")
flags.DEFINE_bool("use_pretrained_embedding", True, "Use pretrained embedding.")
flags.DEFINE_integer("num_steps", int(200 * 1000), "Number of training steps.")
flags.DEFINE_integer("num_disc_updates", 1, "Number of discriminator updates.")
flags.DEFINE_integer("num_gen_updates", 1, "Number of generator updates.")
flags.DEFINE_string("data_dir", "/tmp/emnlp2017", "Directory where data is.")
flags.DEFINE_float("gen_lr", 9.59e-5, "Learning rate generator.")
flags.DEFINE_float("disc_lr", 9.38e-3, "Learning rate discriminator.")
flags.DEFINE_float("gen_beta1", 0.5, "Beta1 for generator.")
flags.DEFINE_float("disc_beta1", 0.5, "Beta1 for discriminator.")
flags.DEFINE_float("gamma", 0.23, "Discount factor.")
flags.DEFINE_float("baseline_decay", 0.08, "Baseline decay rate.")
flags.DEFINE_string("mode", "train", "train or evaluate_pair.")
flags.DEFINE_string("checkpoint_dir", "/tmp/emnlp2017/checkpoints/",
                    "Directory for checkpoints.")
flags.DEFINE_integer("export_every", 1000, "Frequency of checkpoint exports.")
flags.DEFINE_integer("num_examples_for_eval", int(1e4),
                     "Number of examples for evaluation")

EVALUATOR_SLEEP_PERIOD = 60  # Seconds evaluator sleeps if nothing to do.


def main(_):
  config = flags.FLAGS

  gfile.makedirs(config.checkpoint_dir)
  if config.mode == "train":
    train(config)
  elif config.mode == "evaluate_pair":
    while True:
      checkpoint_path = utils.maybe_pick_models_to_evaluate(
          checkpoint_dir=config.checkpoint_dir)
      if checkpoint_path:
        evaluate_pair(
            config=config,
            batch_size=config.batch_size,
            checkpoint_path=checkpoint_path,
            data_dir=config.data_dir,
            dataset=config.dataset,
            num_examples_for_eval=config.num_examples_for_eval)
      else:
        logging.info("No models to evaluate found, sleeping for %d seconds",
                     EVALUATOR_SLEEP_PERIOD)
        time.sleep(EVALUATOR_SLEEP_PERIOD)
  else:
    raise Exception(
        "Unexpected mode %s, supported modes are \"train\" or \"evaluate_pair\""
        % (config.mode))


def train(config):
  """Train."""
  logging.info("Training.")

  tf.reset_default_graph()
  np.set_printoptions(precision=4)

  # Get data.
  raw_data = reader.get_raw_data(
      data_path=config.data_dir, dataset=config.dataset)
  train_data, valid_data, word_to_id = raw_data
  id_to_word = {v: k for k, v in word_to_id.items()}
  vocab_size = len(word_to_id)
  max_length = reader.MAX_TOKENS_SEQUENCE[config.dataset]
  logging.info("Vocabulary size: %d", vocab_size)

  iterator = reader.iterator(raw_data=train_data, batch_size=config.batch_size)
  iterator_valid = reader.iterator(
      raw_data=valid_data, batch_size=config.batch_size)

  real_sequence = tf.placeholder(
      dtype=tf.int32,
      shape=[config.batch_size, max_length],
      name="real_sequence")
  real_sequence_length = tf.placeholder(
      dtype=tf.int32, shape=[config.batch_size], name="real_sequence_length")
  first_batch_np = next(iterator)
  valid_batch_np = next(iterator_valid)

  test_real_batch = {k: tf.constant(v) for k, v in first_batch_np.items()}
  test_fake_batch = {
      "sequence":
          tf.constant(
              np.random.choice(
                  vocab_size, size=[config.batch_size,
                                    max_length]).astype(np.int32)),
      "sequence_length":
          tf.constant(
              np.random.choice(max_length,
                               size=[config.batch_size]).astype(np.int32)),
  }
  valid_batch = {k: tf.constant(v) for k, v in valid_batch_np.items()}

  # Create generator.
  if config.use_pretrained_embedding:
    embedding_source = utils.get_embedding_path(config.data_dir, config.dataset)
    vocab_file = "/tmp/vocab.txt"
    with gfile.GFile(vocab_file, "w") as f:
      for i in range(len(id_to_word)):
        f.write(id_to_word[i] + "\n")
    logging.info("Temporary vocab file: %s", vocab_file)
  else:
    embedding_source = None
    vocab_file = None

  gen = generators.LSTMGen(
      vocab_size=vocab_size,
      feature_sizes=[config.gen_feature_size] * config.num_layers_gen,
      max_sequence_length=reader.MAX_TOKENS_SEQUENCE[config.dataset],
      batch_size=config.batch_size,
      use_layer_norm=config.layer_norm_gen,
      trainable_embedding_size=config.trainable_embedding_size,
      input_dropout=config.gen_input_dropout,
      output_dropout=config.gen_output_dropout,
      pad_token=reader.PAD_INT,
      embedding_source=embedding_source,
      vocab_file=vocab_file,
  )
  gen_outputs = gen()

  # Create discriminator.
  disc = discriminator_nets.LSTMEmbedDiscNet(
      vocab_size=vocab_size,
      feature_sizes=[config.disc_feature_size] * config.num_layers_disc,
      trainable_embedding_size=config.trainable_embedding_size,
      embedding_source=embedding_source,
      use_layer_norm=config.layer_norm_disc,
      pad_token=reader.PAD_INT,
      vocab_file=vocab_file,
      dropout=config.disc_dropout,
  )
  disc_logits_real = disc(
      sequence=real_sequence, sequence_length=real_sequence_length)
  disc_logits_fake = disc(
      sequence=gen_outputs["sequence"],
      sequence_length=gen_outputs["sequence_length"])

  # Loss of the discriminator.
  if config.disc_loss_type == "ce":
    targets_real = tf.ones(
        [config.batch_size, reader.MAX_TOKENS_SEQUENCE[config.dataset]])
    targets_fake = tf.zeros(
        [config.batch_size, reader.MAX_TOKENS_SEQUENCE[config.dataset]])
    loss_real = losses.sequential_cross_entropy_loss(disc_logits_real,
                                                     targets_real)
    loss_fake = losses.sequential_cross_entropy_loss(disc_logits_fake,
                                                     targets_fake)
    disc_loss = 0.5 * loss_real + 0.5 * loss_fake

  # Loss of the generator.
  gen_loss, cumulative_rewards, baseline = losses.reinforce_loss(
      disc_logits=disc_logits_fake,
      gen_logprobs=gen_outputs["logprobs"],
      gamma=config.gamma,
      decay=config.baseline_decay)

  # Optimizers
  disc_optimizer = tf.train.AdamOptimizer(
      learning_rate=config.disc_lr, beta1=config.disc_beta1)
  gen_optimizer = tf.train.AdamOptimizer(
      learning_rate=config.gen_lr, beta1=config.gen_beta1)

  # Get losses and variables.
  disc_vars = disc.get_all_variables()
  gen_vars = gen.get_all_variables()
  l2_disc = tf.reduce_sum(tf.add_n([tf.nn.l2_loss(v) for v in disc_vars]))
  l2_gen = tf.reduce_sum(tf.add_n([tf.nn.l2_loss(v) for v in gen_vars]))
  scalar_disc_loss = tf.reduce_mean(disc_loss) + config.l2_disc * l2_disc
  scalar_gen_loss = tf.reduce_mean(gen_loss) + config.l2_gen * l2_gen

  # Update ops.
  global_step = tf.train.get_or_create_global_step()
  disc_update = disc_optimizer.minimize(
      scalar_disc_loss, var_list=disc_vars, global_step=global_step)
  gen_update = gen_optimizer.minimize(
      scalar_gen_loss, var_list=gen_vars, global_step=global_step)

  # Saver.
  saver = tf.train.Saver()

  # Metrics
  test_disc_logits_real = disc(**test_real_batch)
  test_disc_logits_fake = disc(**test_fake_batch)
  valid_disc_logits = disc(**valid_batch)
  disc_predictions_real = tf.nn.sigmoid(disc_logits_real)
  disc_predictions_fake = tf.nn.sigmoid(disc_logits_fake)
  valid_disc_predictions = tf.reduce_mean(
      tf.nn.sigmoid(valid_disc_logits), axis=0)
  test_disc_predictions_real = tf.reduce_mean(
      tf.nn.sigmoid(test_disc_logits_real), axis=0)
  test_disc_predictions_fake = tf.reduce_mean(
      tf.nn.sigmoid(test_disc_logits_fake), axis=0)

  # Only log results for the first element of the batch.
  metrics = {
      "scalar_gen_loss": scalar_gen_loss,
      "scalar_disc_loss": scalar_disc_loss,
      "disc_predictions_real": tf.reduce_mean(disc_predictions_real),
      "disc_predictions_fake": tf.reduce_mean(disc_predictions_fake),
      "test_disc_predictions_real": tf.reduce_mean(test_disc_predictions_real),
      "test_disc_predictions_fake": tf.reduce_mean(test_disc_predictions_fake),
      "valid_disc_predictions": tf.reduce_mean(valid_disc_predictions),
      "cumulative_rewards": tf.reduce_mean(cumulative_rewards),
      "baseline": tf.reduce_mean(baseline),
  }

  # Training.
  logging.info("Starting training")
  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    latest_ckpt = tf.train.latest_checkpoint(config.checkpoint_dir)
    if latest_ckpt:
      saver.restore(sess, latest_ckpt)

    for step in range(config.num_steps):
      real_data_np = next(iterator)
      train_feed = {
          real_sequence: real_data_np["sequence"],
          real_sequence_length: real_data_np["sequence_length"],
      }

      # Update generator and discriminator.
      for _ in range(config.num_disc_updates):
        sess.run(disc_update, feed_dict=train_feed)
      for _ in range(config.num_gen_updates):
        sess.run(gen_update, feed_dict=train_feed)

      # Reporting
      if step % config.export_every == 0:
        gen_sequence_np, metrics_np = sess.run(
            [gen_outputs["sequence"], metrics], feed_dict=train_feed)
        metrics_np["gen_sentence"] = utils.sequence_to_sentence(
            gen_sequence_np[0, :], id_to_word)
        saver.save(
            sess,
            save_path=config.checkpoint_dir + "scratchgan",
            global_step=global_step)
        metrics_np["model_path"] = tf.train.latest_checkpoint(
            config.checkpoint_dir)
        logging.info(metrics_np)

    # After training, export models.
    saver.save(
        sess,
        save_path=config.checkpoint_dir + "scratchgan",
        global_step=global_step)
    logging.info("Saved final model at %s.",
                 tf.train.latest_checkpoint(config.checkpoint_dir))


def evaluate_pair(config, batch_size, checkpoint_path, data_dir, dataset,
                  num_examples_for_eval):
  """Evaluates a pair generator discriminator.

  This function loads a discriminator from disk, a generator, and evaluates the
  discriminator against the generator.

  It returns the mean probability of the discriminator against several batches,
  and the FID of the generator against the validation data.

  It also writes evaluation samples to disk.

  Args:
    config: dict, the config file.
    batch_size: int, size of the batch.
    checkpoint_path: string, full path to the TF checkpoint on disk.
    data_dir: string, path to a directory containing the dataset.
    dataset: string, "emnlp2017", to select the right dataset.
    num_examples_for_eval: int, number of examples for evaluation.
  """
  tf.reset_default_graph()
  logging.info("Evaluating checkpoint %s.", checkpoint_path)

  # Build graph.
  train_data, valid_data, word_to_id = reader.get_raw_data(
      data_dir, dataset=dataset)
  id_to_word = {v: k for k, v in word_to_id.items()}
  vocab_size = len(word_to_id)
  train_iterator = reader.iterator(raw_data=train_data, batch_size=batch_size)
  valid_iterator = reader.iterator(raw_data=valid_data, batch_size=batch_size)
  train_sequence = tf.placeholder(
      dtype=tf.int32,
      shape=[batch_size, reader.MAX_TOKENS_SEQUENCE[dataset]],
      name="train_sequence")
  train_sequence_length = tf.placeholder(
      dtype=tf.int32, shape=[batch_size], name="train_sequence_length")
  valid_sequence = tf.placeholder(
      dtype=tf.int32,
      shape=[batch_size, reader.MAX_TOKENS_SEQUENCE[dataset]],
      name="valid_sequence")
  valid_sequence_length = tf.placeholder(
      dtype=tf.int32, shape=[batch_size], name="valid_sequence_length")
  disc_inputs_train = {
      "sequence": train_sequence,
      "sequence_length": train_sequence_length,
  }
  disc_inputs_valid = {
      "sequence": valid_sequence,
      "sequence_length": valid_sequence_length,
  }
  if config.use_pretrained_embedding:
    embedding_source = utils.get_embedding_path(config.data_dir, config.dataset)
    vocab_file = "/tmp/vocab.txt"
    with gfile.GFile(vocab_file, "w") as f:
      for i in range(len(id_to_word)):
        f.write(id_to_word[i] + "\n")
    logging.info("Temporary vocab file: %s", vocab_file)
  else:
    embedding_source = None
    vocab_file = None
  gen = generators.LSTMGen(
      vocab_size=vocab_size,
      feature_sizes=[config.gen_feature_size] * config.num_layers_gen,
      max_sequence_length=reader.MAX_TOKENS_SEQUENCE[config.dataset],
      batch_size=config.batch_size,
      use_layer_norm=config.layer_norm_gen,
      trainable_embedding_size=config.trainable_embedding_size,
      input_dropout=config.gen_input_dropout,
      output_dropout=config.gen_output_dropout,
      pad_token=reader.PAD_INT,
      embedding_source=embedding_source,
      vocab_file=vocab_file,
  )
  gen_outputs = gen()

  disc = discriminator_nets.LSTMEmbedDiscNet(
      vocab_size=vocab_size,
      feature_sizes=[config.disc_feature_size] * config.num_layers_disc,
      trainable_embedding_size=config.trainable_embedding_size,
      embedding_source=embedding_source,
      use_layer_norm=config.layer_norm_disc,
      pad_token=reader.PAD_INT,
      vocab_file=vocab_file,
      dropout=config.disc_dropout,
  )

  disc_inputs = {
      "sequence": gen_outputs["sequence"],
      "sequence_length": gen_outputs["sequence_length"],
  }
  gen_logits = disc(**disc_inputs)
  train_logits = disc(**disc_inputs_train)
  valid_logits = disc(**disc_inputs_valid)

  # Saver.
  saver = tf.train.Saver()

  # Reduce over time and batch.
  train_probs = tf.reduce_mean(tf.nn.sigmoid(train_logits))
  valid_probs = tf.reduce_mean(tf.nn.sigmoid(valid_logits))
  gen_probs = tf.reduce_mean(tf.nn.sigmoid(gen_logits))

  outputs = {
      "train_probs": train_probs,
      "valid_probs": valid_probs,
      "gen_probs": gen_probs,
      "gen_sequences": gen_outputs["sequence"],
      "valid_sequences": valid_sequence
  }

  # Get average discriminator score and store generated sequences.
  all_valid_sentences = []
  all_gen_sentences = []
  all_gen_sequences = []
  mean_train_prob = 0.0
  mean_valid_prob = 0.0
  mean_gen_prob = 0.0

  logging.info("Graph constructed, generating batches.")
  num_batches = num_examples_for_eval // batch_size + 1

  # Restrict the thread pool size to prevent excessive GCU usage on Borg.
  tf_config = tf.ConfigProto()
  tf_config.intra_op_parallelism_threads = 16
  tf_config.inter_op_parallelism_threads = 16

  with tf.Session(config=tf_config) as sess:

    # Restore variables from checkpoints.
    logging.info("Restoring variables.")
    saver.restore(sess, checkpoint_path)

    for i in range(num_batches):
      logging.info("Batch %d / %d", i, num_batches)
      train_data_np = next(train_iterator)
      valid_data_np = next(valid_iterator)
      feed_dict = {
          train_sequence: train_data_np["sequence"],
          train_sequence_length: train_data_np["sequence_length"],
          valid_sequence: valid_data_np["sequence"],
          valid_sequence_length: valid_data_np["sequence_length"],
      }
      outputs_np = sess.run(outputs, feed_dict=feed_dict)
      all_gen_sequences.extend(outputs_np["gen_sequences"])
      gen_sentences = utils.batch_sequences_to_sentences(
          outputs_np["gen_sequences"], id_to_word)
      valid_sentences = utils.batch_sequences_to_sentences(
          outputs_np["valid_sequences"], id_to_word)
      all_valid_sentences.extend(valid_sentences)
      all_gen_sentences.extend(gen_sentences)
      mean_train_prob += outputs_np["train_probs"] / batch_size
      mean_valid_prob += outputs_np["valid_probs"] / batch_size
      mean_gen_prob += outputs_np["gen_probs"] / batch_size

  logging.info("Evaluating FID.")

  # Compute FID
  fid = eval_metrics.fid(
      generated_sentences=all_gen_sentences[:num_examples_for_eval],
      real_sentences=all_valid_sentences[:num_examples_for_eval])

  utils.write_eval_results(config.checkpoint_dir, all_gen_sentences,
                           os.path.basename(checkpoint_path), mean_train_prob,
                           mean_valid_prob, mean_gen_prob, fid)


if __name__ == "__main__":
  app.run(main)
