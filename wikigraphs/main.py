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
"""Train a transformer for language modeling on Wikitext-103."""

import concurrent
import functools
import os
import pickle
import time

from absl import app
from absl import flags
from absl import logging

import jax
import jraph
import numpy as np
import optax

from updaters import CheckpointingUpdater
from updaters import Updater
import utils


# Train
flags.DEFINE_integer('train_batch_size', 4, '(Per-Device) batch size for'
                     ' training.')
flags.DEFINE_integer('train_timesteps', 150, 'Sequence length to learn on')
flags.DEFINE_integer('train_memory_size', 150, 'Memory size for transformer XL')
flags.DEFINE_bool('debug', False, 'Whether to turn on debugging mode')
flags.DEFINE_string('job_mode', 'train',
                    'One of `train`, `eval`, `sample`, `retrieve`.')
flags.DEFINE_integer('random_seed', 42, 'Random seed id.')
flags.DEFINE_integer('num_gpus', 8, 'Number of GPUs for training.')

# Eval
flags.DEFINE_integer('eval_batch_size', 1, 'Evaluation batch size')
flags.DEFINE_string('eval_subset', 'valid', 'Which subset to evaluate on,'
                    ' one of `valid`, `test`.')
flags.DEFINE_integer('eval_every', 10, 'Evaluation frequency.')
flags.DEFINE_integer('eval_timesteps', 64, 'Sequence length to learn on')
flags.DEFINE_integer('eval_memory_size', 640, 'Memory size for transformer XL')
flags.DEFINE_integer('max_eval_samples', -1, 'Max number of eval samples. Set'
                     ' as -1 to use the entire eval set.')

# Model
flags.DEFINE_integer('emb_dim', 410, 'model width')
flags.DEFINE_integer('num_heads', 10, 'Number of attention heads')
flags.DEFINE_integer('num_layers', 16, 'Number of transformer layers')
flags.DEFINE_integer('dense_dim', 2100, 'Size of dense hidden layer.')
flags.DEFINE_integer('tail_shrink_factor', 4,
                     'Low-frequency vocabulary shrinkage factor in adaptive'
                     ' softmax.')
flags.DEFINE_string('emb_type', 'adaptive_softmax', 'Type of the word embedding'
                    ' layer.')
flags.DEFINE_integer('clamp_len', 400, 'Clamp length for transformer XL.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate for the transformer layers.')
flags.DEFINE_float('dropout_attn', 0.0, 'Dropout rate for the attention'
                   ' weights.')
flags.DEFINE_float('self_att_init_scale', 0.02,
                   'Self attention module initilization scale.')
flags.DEFINE_float('dense_init_scale', 0.02,
                   'Dense module initilization scale.')

# Graph neural net configs
flags.DEFINE_string('gnn_embed_type', 'adaptive', 'Token embedding type for the'
                    ' graph.')
flags.DEFINE_integer('gnn_embed_dim', 128, 'Graph node embedding size.')
flags.DEFINE_integer('gnn_num_layers', 1, 'Number of layers in the GNN.')
flags.DEFINE_bool('gnn_layer_norm', True, 'Whether to use layer norm in GNN.')

# Bag-of-words to text configs
flags.DEFINE_integer('bow_embedding_dim', 256, 'Size of the bow embeddings.')
flags.DEFINE_integer('bow_n_tokens', 1, 'Number of tokens to use for the'
                     ' bow2text model.')

# Sampling
flags.DEFINE_float('sampling_temperature', 0.8, 'Temperature used for'
                   ' sampling.  Sampling becomes more deterministic with a'
                   ' lower temperature.  Setting temperature to 1.0 samples'
                   ' from the model distribution.')
flags.DEFINE_bool('prompt_title', False, 'Whether to prompt title when sample')
flags.DEFINE_integer('sample_length', 512, 'Length of samples.')
flags.DEFINE_integer('sample_memory_size', 640, 'Memory size for sampling.')
flags.DEFINE_integer('num_samples', 1000, 'Maximum number of samples to'
                     ' generate.')

# Optimization
flags.DEFINE_float('init_lr', 0.00025, 'Initial learning rate.')
flags.DEFINE_float('min_lr_ratio', 0.0, 'Minimum learning rate as a ratio of'
                   ' `init_lr`.')
flags.DEFINE_string('lr_schedule', 'cosine', 'One of `default`, `cosine`.')
flags.DEFINE_float('grad_clip', 0.25, 'Maximum gradient norm allowed for'
                   ' clipping, set to a very large number to disable clipping.')
flags.DEFINE_integer('max_steps', 200_000, 'Number of training steps.')
flags.DEFINE_string('checkpoint_dir', '/tmp/graph2text',
                    'Directory to store checkpoints.')

# Data
flags.DEFINE_string('dataset', 'freebase2wikitext', 'Which dataset to train on,'
                    ' one of "wikitext", "freebase2wikitext".')
flags.DEFINE_string('model_type', 'graph2text', 'One of "text", "graph2text",'
                    ' "bow2text".')
flags.DEFINE_string('graph_data_version', 'max256', 'One of "max256", "max512",'
                    ' "max1024".')

flags.DEFINE_integer('log_every', 50, 'Log every this many steps.')
flags.DEFINE_integer('ckpt_every', 1000, 'Checkpoint every this many steps.')

FLAGS = flags.FLAGS


def _preprocess(batch, num_devices=1):
  return utils.preprocess(batch, FLAGS.model_type, num_devices)


def _train(updater, train_dataset, num_devices):
  """Train the transformer model."""
  # Initialize parameters.
  logging.info('Initializing parameters...')
  rng = jax.random.PRNGKey(FLAGS.random_seed)
  state = updater.init(
      rng, _preprocess(train_dataset.return_faux_batch(), num_devices))

  logging.info('Starting train loop...')
  prev_time = time.time()
  while True:
    data = next(train_dataset)
    state, metrics = updater.update(state, _preprocess(data, num_devices))
    # We use JAX runahead to mask data preprocessing and JAX dispatch overheads.
    # Using values from state/metrics too often will block the runahead and can
    # cause these overheads to become more prominent.
    step = np.array(metrics['step'])
    if step % FLAGS.log_every == 0:
      steps_per_sec = FLAGS.log_every / (time.time() - prev_time)
      prev_time = time.time()
      metrics.update({'steps_per_sec': steps_per_sec})
      logging.info({k: float(v) for k, v in metrics.items()})
    if step % FLAGS.ckpt_every == 0:
      updater.save_checkpoint(state)
    if step > FLAGS.max_steps:
      break


def _eval(updater, eval_dataset):
  """Evaluate the transformer model."""
  checkpoint_state = updater.load_checkpoint()
  rng = jax.random.PRNGKey(FLAGS.random_seed)
  state = updater.init_from_checkpoint(
      rng, _preprocess(eval_dataset.return_faux_batch()), checkpoint_state)
  eval_out, state = utils.evaluate(
      eval_dataset, state, updater, FLAGS.eval_batch_size, _preprocess,
      FLAGS.max_eval_samples, print_progress_every=20)
  logging.info('Eval output: %s', eval_out)


def _retrieve(updater, eval_dataset):
  """Graph and text retrieval using the transformer model."""
  checkpoint_state = updater.load_checkpoint()
  rng = jax.random.PRNGKey(FLAGS.random_seed)
  state = updater.init_from_checkpoint(
      rng, _preprocess(eval_dataset.return_faux_batch()), checkpoint_state)
  retrieval_out, _ = utils.compute_text_graph_relevance(
      eval_dataset, state, updater, preprocess_fn=_preprocess,
      print_progress_every=20)
  logging.info('Retrieval output: %s', retrieval_out)


def _sample(eval_dataset, tokenizer, devices, batch_size=1):
  """Evaluate the graph2text transformer."""
  checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, 'checkpoint.pkl')
  logging.info('Loading checkpoint from %s', checkpoint_dir)
  with open(checkpoint_dir, 'rb') as f:
    state = pickle.load(f)

  if FLAGS.model_type == 'graph2text':
    # process list of graphs into a batch
    eval_dataset = map(lambda x: dict(  # pylint: disable=g-long-lambda
        obs=x['obs'],
        target=x['target'],
        should_reset=x['should_reset'],
        mask=x['mask'],
        graphs=jraph.batch(x['graphs']),
        ), eval_dataset)
  eval_dataset = utils.take_unique_graphs(eval_dataset, FLAGS.model_type)

  samplers = []
  for device in devices:
    sampler = utils.build_sampler(tokenizer, device=device)
    samplers.append(sampler)

  step = state['step']
  params = state['params']
  sample_logger = []

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(samplers)) as executor:
    futures = dict()
    for sampler in samplers:
      batch = next(eval_dataset)
      prompts = utils.construct_prompts(
          batch['obs'], batch_size, FLAGS.sample_length, tokenizer,
          prompt_title=FLAGS.prompt_title)
      if FLAGS.model_type in ['graph2text', 'bow2text']:
        future = executor.submit(
            utils.generate_samples, params, tokenizer, sampler,
            model_type=FLAGS.model_type, prompts=prompts,
            graphs=batch['graphs'])
        futures[future] = (sampler, batch['graphs'], batch['obs'])
      else:
        future = executor.submit(
            utils.generate_samples, params, tokenizer, sampler,
            model_type=FLAGS.model_type, prompts=prompts, graphs=None)
        futures[future] = (sampler, batch['obs'])

    n_samples = 0

    while n_samples < FLAGS.num_samples:
      for future, future_items in list(futures.items()):
        if not future.done():
          continue
        samples, tokens = future.result()
        if FLAGS.model_type == 'graph2text':
          sampler, graphs, text = future_items
          graphs = jraph.unbatch(graphs)
        elif FLAGS.model_type == 'bow2text':
          sampler, graphs, text = future_items
        else:
          sampler, text = future_items

        if FLAGS.model_type in ['graph2text', 'bow2text']:
          for s, g, tk, txt in zip(samples, graphs, tokens, text):
            # Only log a small fraction of the generated samples, if we are
            # generating non-stop.  Otherwise log every sample.
            logging.info('[step %d]', step)
            logging.info('graph=\n%r', g)
            logging.info('sample=\n%s', s)
            if FLAGS.model_type == 'graph2text':
              sample_logger.append({
                  'step': step,
                  'sample': s,
                  'sample_tokens': tk,
                  'ground_truth_text': txt,
              })
            elif FLAGS.model_type == 'bow2text':
              sample_logger.append({
                  'step': step,
                  'bow': g,
                  'sample': s,
                  'sample_tokens': tk,
                  'ground_truth_text': txt,
              })
        else:
          for s, tk, txt in zip(samples, tokens, text):
            # Only log a small fraction of the generated samples, if we are
            # generating non-stop.  Otherwise log every sample.
            logging.info('[step %d]', step)
            logging.info('sample=\n%s', s)
            sample_logger.append({
                'step': step,
                'sample': s,
                'sample_tokens': tk,
                'ground_truth_text': txt,
            })
        n_samples += len(samples)
        logging.info('Finished generating %d samples', n_samples)

        del futures[future]

        if n_samples < FLAGS.num_samples:
          batch = next(eval_dataset)
          prompts = utils.construct_prompts(
              batch['obs'], batch_size, FLAGS.sample_length, tokenizer,
              prompt_title=FLAGS.prompt_title)
          if FLAGS.model_type in ['graph2text', 'bow2text']:
            future = executor.submit(
                utils.generate_samples, params, tokenizer, sampler,
                model_type=FLAGS.model_type, prompts=prompts,
                graphs=batch['graphs'])
            futures[future] = (sampler, batch['graphs'], batch['obs'])
          else:
            future = executor.submit(
                utils.generate_samples, params, tokenizer, sampler,
                model_type=FLAGS.model_type, prompts=prompts, graphs=None)
            futures[future] = (sampler, batch['obs'])

  logging.info('Finished')
  path = os.path.join(FLAGS.checkpoint_dir, 'samples.pkl')
  with open(path, 'wb') as f:
    pickle.dump(dict(samples=sample_logger), f)
  logging.info('Samples saved to %s', path)


def main(_):
  # Create the dataset.
  tokenizer = utils.init_tokenizer(FLAGS.dataset)
  graph_tokenizer = utils.init_graph_tokenizer()
  dataset_class = utils.get_dataset_class(FLAGS.dataset, FLAGS.model_type)
  has_graph = True if FLAGS.model_type == 'graph2text' else False
  local_devices = jax.local_devices()
  num_gpus = min(FLAGS.num_gpus, len(local_devices))

  if FLAGS.job_mode == 'train':
    train_dataset = dataset_class(
        tokenizer=tokenizer,
        graph_tokenizer=graph_tokenizer,
        batch_size=FLAGS.train_batch_size,
        subset='train',
        timesteps=FLAGS.train_timesteps,
        version=FLAGS.graph_data_version,
        shuffle_data=True,
        repeat=True,
        debug=FLAGS.debug)
    train_iter = iter(train_dataset)
    loss_fn = utils.build_loss_fn(vocab_size=tokenizer.vocab_size,
                                  cache_steps=FLAGS.train_memory_size)
    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.grad_clip),
        optax.scale_by_adam(),
        optax.scale_by_schedule(functools.partial(
            utils.schedule,
            lr_schedule=FLAGS.lr_schedule,
            init_lr=FLAGS.init_lr,
            min_lr_ratio=FLAGS.min_lr_ratio,
            max_steps=FLAGS.max_steps)),
        optax.scale(-1))
    optimizer = optax.apply_if_finite(optimizer, max_consecutive_errors=5)
    updater = Updater(loss_fn, optimizer,
                      devices=local_devices[:num_gpus],
                      has_graph=has_graph)
    updater = CheckpointingUpdater(updater, FLAGS.checkpoint_dir)
    _train(updater, train_iter, num_gpus)
  elif FLAGS.job_mode == 'eval':
    eval_dataset = dataset_class(
        tokenizer=tokenizer,
        graph_tokenizer=graph_tokenizer,
        batch_size=FLAGS.eval_batch_size,
        subset=FLAGS.eval_subset,
        timesteps=FLAGS.eval_timesteps,
        version=FLAGS.graph_data_version,
        shuffle_data=False,
        repeat=False,
        debug=FLAGS.debug)
    eval_iter = iter(eval_dataset)
    loss_fn = utils.build_loss_fn(vocab_size=tokenizer.vocab_size,
                                  cache_steps=FLAGS.eval_memory_size)
    # only use one device for evaluation
    devices = local_devices[:1]
    updater = Updater(loss_fn, optimizer=None, devices=devices,
                      has_graph=has_graph)
    updater = CheckpointingUpdater(updater, FLAGS.checkpoint_dir)
    _eval(updater, eval_iter)
  elif FLAGS.job_mode == 'sample':
    eval_dataset = dataset_class(
        tokenizer=tokenizer,
        graph_tokenizer=graph_tokenizer,
        batch_size=1,
        subset=FLAGS.eval_subset,
        timesteps=FLAGS.sample_length,
        version=FLAGS.graph_data_version,
        shuffle_data=False,
        repeat=True,
        debug=FLAGS.debug)
    eval_iter = iter(eval_dataset)
    _sample(eval_iter, tokenizer, local_devices[:num_gpus])
  elif FLAGS.job_mode == 'retrieve':
    eval_dataset = dataset_class(
        tokenizer=tokenizer,
        graph_tokenizer=graph_tokenizer,
        batch_size=1,
        subset=FLAGS.eval_subset,
        timesteps=FLAGS.eval_timesteps,
        version=FLAGS.graph_data_version,
        shuffle_data=False,
        repeat=False,
        graph_retrieval_dataset=True,
        debug=FLAGS.debug)
    eval_iter = iter(eval_dataset)
    loss_fn = utils.build_loss_fn(vocab_size=tokenizer.vocab_size,
                                  cache_steps=FLAGS.eval_memory_size)
    # only use one device for evaluation
    devices = local_devices[:1]
    updater = Updater(loss_fn, optimizer=None, devices=devices,
                      has_graph=has_graph)
    updater = CheckpointingUpdater(updater, FLAGS.checkpoint_dir)
    _retrieve(updater, eval_iter)

if __name__ == '__main__':
  app.run(main)
