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
"""Utility functions for the training script."""

import collections
import math
import random

from absl import flags
from absl import logging

import jax.numpy as jnp
import jraph
import numpy as np
import sklearn

from wikigraphs.data import paired_dataset as pd
from wikigraphs.data import tokenizers
from wikigraphs.data import wikitext as wt
from wikigraphs.model import graph_net as gn
from wikigraphs.model import sampler as transformer_sampler
from wikigraphs.model import transformer


FLAGS = flags.FLAGS

VOCAB_FILES_MAP = {
    'wikitext': '/tmp/data/wikitext-vocab.csv',
    'freebase2wikitext': '/tmp/data/text-vocab.csv',
}

GRAPH_VOCAB_FILE = '/tmp/data/graph-vocab.csv'


def init_tokenizer(dataset_name):
  """Initialie the tokenizer."""
  logging.info('Loading tokenizer...')
  tokenizer = tokenizers.WordTokenizer(VOCAB_FILES_MAP[dataset_name])
  logging.info('Vocab size: %d', tokenizer.vocab_size)
  return tokenizer


def init_graph_tokenizer():
  """Initialie the tokenizer."""
  logging.info('Loading graph tokenizer...')
  tokenizer = tokenizers.GraphTokenizer(GRAPH_VOCAB_FILE)
  logging.info('Vocab size: %d', tokenizer.vocab_size)
  return tokenizer


def get_dataset_class(dataset_name, model_type, job_mode='train'):
  """Get the dataset class used for all jobs."""
  if dataset_name == 'freebase2wikitext':
    if model_type == 'bow2text':
      return pd.Bow2TextDataset
    elif FLAGS.model_type == 'graph2text':
      return pd.Graph2TextDataset
    elif FLAGS.model_type == 'text':
      if job_mode in ['train', 'eval']:
        return pd.TextOnlyDataset
      else:
        # for sampling: taking the unique graphs for a fair comparison
        return pd.Bow2TextDataset
    else:
      # Add other graph2text data here.
      raise NotImplementedError()
  else:
    def dataset(graph_tokenizer, *args, **kwargs):
      del graph_tokenizer
      return wt.Dataset(*args, **kwargs)
    return dataset


def preprocess(batch, model_type, num_devices=1):
  """Preprocess the batch before sending to the model."""
  if model_type == 'text':
    if 'graphs' in batch:
      del batch['graphs']
  elif model_type == 'bow2text':
    # Do nothing, bow2text data is already in a good form.
    pass
  else:  # graph2text
    if num_devices == 1:
      graphs = gn.pad_graphs(jraph.batch(batch['graphs']))
    else:
      # We need to first batch graphs into num_devices batchs.
      graphs = gn.batch_graphs_by_device(batch['graphs'], num_devices)
      # Then we pad them to the maximum graph size in the batch and concat.
      # This way graphs can be distributed to each device through pmap.
      graphs = gn.pad_graphs_by_device(graphs)
    max_graph_size = gn.pad_size(graphs.n_node.max())
    batch.update({
        'graphs': graphs,
        'max_graph_size': max_graph_size})
  return batch


def text_model_fn(vocab_size):
  return transformer.TransformerXL(
      vocab_size=vocab_size,
      emb_dim=FLAGS.emb_dim,
      num_layers=FLAGS.num_layers,
      num_heads=FLAGS.num_heads,
      dropout_prob=FLAGS.dropout,
      dropout_attn_prob=FLAGS.dropout_attn,
      self_att_init_scale=FLAGS.self_att_init_scale,
      dense_init_scale=FLAGS.dense_init_scale,
      dense_dim=FLAGS.dense_dim,
      tail_shrink_factor=FLAGS.tail_shrink_factor,
      relative_pos_clamp_len=FLAGS.clamp_len or None)


def graph2text_model_fn(vocab_size):
  """Get graph2text transformer model."""
  return transformer.Graph2TextTransformer(
      vocab_size=vocab_size,
      emb_dim=FLAGS.emb_dim,
      num_layers=FLAGS.num_layers,
      num_heads=FLAGS.num_heads,
      dropout_prob=FLAGS.dropout,
      dropout_attn_prob=FLAGS.dropout_attn,
      self_att_init_scale=FLAGS.self_att_init_scale,
      dense_init_scale=FLAGS.dense_init_scale,
      dense_dim=FLAGS.dense_dim,
      tail_shrink_factor=FLAGS.tail_shrink_factor,
      relative_pos_clamp_len=FLAGS.clamp_len or None,
      gnn_embed_dim=FLAGS.gnn_embed_dim,
      gnn_num_layers=FLAGS.gnn_num_layers,
      gnn_layer_norm=FLAGS.gnn_layer_norm)


def bow2text_model_fn(vocab_size):
  """Get the bow2text model."""
  return transformer.Bow2TextTransformer(
      vocab_size=vocab_size,
      emb_dim=FLAGS.emb_dim,
      num_layers=FLAGS.num_layers,
      num_heads=FLAGS.num_heads,
      dropout_prob=FLAGS.dropout,
      dropout_attn_prob=FLAGS.dropout_attn,
      self_att_init_scale=FLAGS.self_att_init_scale,
      dense_init_scale=FLAGS.dense_init_scale,
      dense_dim=FLAGS.dense_dim,
      tail_shrink_factor=FLAGS.tail_shrink_factor,
      relative_pos_clamp_len=FLAGS.clamp_len or None,
      bow_embedding_dim=FLAGS.bow_embedding_dim,
      bow_n_tokens=FLAGS.bow_n_tokens)


def build_loss_fn(vocab_size, cache_steps):
  """Build the appropriate loss function according to the configs."""
  if FLAGS.model_type == 'text':
    def loss_fn(data, is_training=True):
      return text_model_fn(vocab_size=vocab_size).loss(
          data['obs'], data['target'], data['mask'],
          is_training=is_training,
          should_reset=data['should_reset'],
          cache_steps=cache_steps)
  elif FLAGS.model_type == 'graph2text':
    def loss_fn(data, max_graph_size, is_training=True):
      return graph2text_model_fn(vocab_size=vocab_size).loss(
          data['graphs'], max_graph_size, True,
          data['obs'], data['target'], data['mask'],
          is_training=is_training,
          should_reset=data['should_reset'],
          cache_steps=cache_steps)
  elif FLAGS.model_type == 'bow2text':
    def loss_fn(data, is_training=True):
      return bow2text_model_fn(vocab_size=vocab_size).loss(
          data['graphs'], data['obs'], data['target'], data['mask'],
          is_training=is_training,
          should_reset=data['should_reset'],
          cache_steps=cache_steps)
  else:
    raise ValueError(f'Unknown model type "{FLAGS.model_type}".')
  return loss_fn


def build_sampler(tokenizer, device=None):
  """Build the appropriate sampler according to the configs."""
  if FLAGS.model_type == 'text':
    model_fn = lambda prompts: text_model_fn(tokenizer.vocab_size)(  # pylint: disable=g-long-lambda
        prompts, is_training=False, cache_steps=FLAGS.sample_memory_size)
    sampler_class = transformer_sampler.TransformerXLSampler
  elif FLAGS.model_type == 'graph2text':
    def model_fn(graphs, max_graph_size, prompts):
      return graph2text_model_fn(tokenizer.vocab_size)(
          graphs, max_graph_size, True, prompts, is_training=False,
          cache_steps=FLAGS.sample_memory_size)
    sampler_class = transformer_sampler.Graph2TextTransformerSampler
  elif FLAGS.model_type == 'bow2text':
    def model_fn(graphs, prompts):
      return bow2text_model_fn(tokenizer.vocab_size)(
          graphs, prompts, is_training=False,
          cache_steps=FLAGS.sample_memory_size)
    sampler_class = transformer_sampler.Bow2TextTransformerSampler
  sampler = sampler_class(model_fn, FLAGS.sampling_temperature, device)
  return sampler


def schedule(i, lr_schedule, init_lr, min_lr_ratio, max_steps):
  if lr_schedule == 'cosine':
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * i / max_steps))
    decayed = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio
    return init_lr * decayed
  else:
    return jnp.where(
        i > 350000, init_lr / 3**3,
        jnp.where(i > 250000, init_lr / 3**2,
                  jnp.where(i > 150000, init_lr / 3, init_lr)))


def evaluate(eval_set, initial_state, updater, eval_batch_size=1,
             preprocess_fn=None, max_eval_samples=-1,
             print_progress_every=None):
  """Evaluate a model on given dataset."""
  total_losses = []
  total_counts = []
  token_accuracy = []
  seq_accuracy = []
  state = initial_state
  step = state['step']
  for i, batch in enumerate(eval_set):
    state, eval_out = updater.eval_return_state(state, preprocess_fn(batch))
    total_losses.append(eval_out['total_loss'])
    total_counts.append(eval_out['total_count'])
    token_accuracy.append(
        eval_out['token_accuracy'] * eval_out['total_count'])
    seq_accuracy.append(eval_out['seq_accuracy'])
    if print_progress_every and (i + 1) % print_progress_every == 0:
      total_loss = float(jnp.array(total_losses).sum())
      total_count = float(jnp.array(total_counts).sum())
      avg_loss = total_loss / total_count
      bpc = avg_loss * np.log2(np.e)
      perplexity = np.exp(avg_loss)
      logging.info(
          'Evaluated %d batches, total tokens %d, average loss %g,'
          ' bpc %g, perplexity %g.',
          i + 1, total_count, avg_loss, bpc, perplexity)
    if 0 < max_eval_samples <= (i + 1) * eval_batch_size:
      break

  total_loss = jnp.array(total_losses).sum()
  total_count = jnp.array(total_counts).sum()
  avg_loss = total_loss / total_count
  eval_out = dict(total_loss=float(total_loss),
                  total_count=float(total_count),
                  loss=float(avg_loss),
                  token_accuracy=float(
                      jnp.array(token_accuracy).sum() / total_count),
                  seq_accuracy=float(
                      jnp.array(seq_accuracy).sum() / len(seq_accuracy)),
                  step=float(step),
                  bits_per_token=float(avg_loss) * np.log2(np.e),
                  perplexity=np.exp(float(avg_loss)))
  return eval_out, state


def extract_title(text, tokenizer):
  r"""Extract the title in the text.

  The wikitext articles is in the format of `\n = TITLE = \n \n...`. We extract
  the title as the tokens from the start to when the `\n \n` first appears.

  Args:
    text: tokenized input text using `tokenizer`.
    tokenizer: text tokenizer.

  Returns:
    title_end_idx: a numpy.array of shape (batch_size,), it indicates the index
      in `text` that marks the end of the title.
  """
  batch_size, text_length = text.shape
  title_end_idx = np.ones(batch_size, dtype=np.int32)
  newline_token = tokenizer.encode('\n')[0]
  for b in range(batch_size):
    prev_token = 1  # start tokens
    for i in range(1, text_length):  # skip start token
      # when we first see '\n \n', that is the title
      if prev_token == newline_token and text[b, i] == newline_token:
        title_end_idx[b] = i
        break
      else:
        prev_token = text[b, i]
  return title_end_idx


def construct_prompts(text, batch_size, sample_length, tokenizer, prompt_title):
  """Construct prompts for text generation.

  Args:
    text: tokenized input text using `tokenizer`.
    batch_size: the size of the batch.
    sample_length: the length of the sample to be generated.
    tokenizer: text tokenizer.
    prompt_title: whether to return a prompt with the title of the `text`.

  Returns:
    prompts: a numpy.array of shape [batch_size, sample_length], in which -1
      indicates tokens that need to be generated using the sampler.

  """
  prompts = -np.ones((batch_size, sample_length), dtype=np.int32)
  prompts[:, 0] = tokenizer.bos_token()
  if prompt_title and text is not None:
    title_end_idx = extract_title(text, tokenizer)
    for i in range(batch_size):
      prompts[i, 1:title_end_idx[i]+1] = text[i, 1:title_end_idx[i]+1]
  return prompts


def generate_samples(params, tokenizer, sampler, model_type, prompts, graphs):
  """Generate a batch of samples using a sampler."""
  if model_type == 'text':
    samples = sampler.sample(params, prompts)
  elif model_type == 'graph2text':
    samples = sampler.sample(params, prompts, graphs, pad=True)
  elif model_type == 'bow2text':
    samples = sampler.sample(params, prompts, graphs)
  else:
    raise ValueError(f'Unknown model_type {model_type}')
  return [tokenizer.decode(s) for s in samples], samples


def take_unique_graphs(data_iter, model_type):
  """Filter data such that it only returns batches with unique graphs."""
  prev_graphs = None
  for batch in data_iter:
    graphs = batch.get('graphs', None)
    # If there's no graph in batch, don't do any filtering
    if graphs is None:
      yield batch
    else:
      if prev_graphs is None:
        prev_graphs = graphs
        yield batch
      else:
        if model_type == 'graph2text':
          not_same_graph = (prev_graphs.nodes.shape != graphs.nodes.shape or
                            not (prev_graphs.nodes == graphs.nodes).all())
        else:
          not_same_graph = (prev_graphs.shape != graphs.shape or
                            not (prev_graphs == graphs).all())
        if not_same_graph:
          prev_graphs = graphs
          yield batch


def compute_map_sklearn(pred, gt):
  """Computes mAP using scikit-learn."""
  assert len(gt.shape) == len(pred.shape) == 2, (
      'gt should be a one-hot encoding with the same shape as pred')
  ap = [
      sklearn.metrics.average_precision_score(
          gt[c, :], pred[c, :], average=None)
      for c in range(gt.shape[0])
  ]
  return sum(ap) / len(ap)


def compute_recall_at_k(pred, k=1):
  """Computes recall@1 score."""
  num_articles = pred.shape[1]
  return sklearn.metrics.top_k_accuracy_score(
      np.arange(num_articles), pred, k=k)


def compute_text_graph_relevance(
    eval_set, initial_state, updater, eval_batch_size=1, preprocess_fn=None,
    print_progress_every=None):
  """Compute the text and graph relevance a model on given dataset."""
  assert eval_batch_size == 1
  num_articles = eval_set.num_articles
  tokens_count = np.zeros((num_articles, num_articles))
  log_probs = np.zeros((num_articles, num_articles))  # [graphs, texts]
  state = initial_state
  for i, batch in enumerate(eval_set):
    state, eval_out = updater.eval_return_state(state, preprocess_fn(batch))
    graph_id = batch['graph_id'][0]
    seq_id = batch['seq_id'][0]
    tokens_count[graph_id, seq_id] += eval_out['total_count']
    log_probs[graph_id, seq_id] += eval_out['log_probs']
    if print_progress_every is not None and (i + 1) % print_progress_every == 0:
      logging.info('Evaluated %d samples', i + 1)

  log_probs_per_token = log_probs / tokens_count
  labels = np.eye(num_articles)
  eval_out = dict(
      log_probs=log_probs,
      tokens_count=tokens_count,
      log_probs_per_token=log_probs_per_token,
      text2graph_recall_at_1=compute_recall_at_k(log_probs_per_token.T, k=1),
      text2graph_recall_at_5=compute_recall_at_k(log_probs_per_token.T, k=5),
      text2graph_map=compute_map_sklearn(log_probs_per_token.T, labels),
      graph2text_recall_at_1=compute_recall_at_k(log_probs_per_token, k=1),
      graph2text_recall_at_5=compute_recall_at_k(log_probs_per_token, k=5),
      graph2text_map=compute_map_sklearn(log_probs_per_token, labels))
  return eval_out, state


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.

  Originally from tensor2tensor/tensor2tensor/utils/bleu_hook.py

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    BLEU score and n-gram precisions.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

    if random.random() < 0.01:
      print('==========')
      for k, v in overlap.items():
        if len(k) >= 3:
          print('%s : %d' % (str(k), v))

  # print(matches_by_order)
  # print(possible_matches_by_order)

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return bleu, precisions
