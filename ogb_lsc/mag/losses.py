# Copyright 2021 DeepMind Technologies Limited.
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

"""Losses and related utilities."""

from typing import Mapping, Tuple, Sequence, NamedTuple, Dict, Optional
import jax
import jax.numpy as jnp
import jraph
import numpy as np

# pylint: disable=g-bad-import-order
import datasets

LogsDict = Mapping[str, jnp.ndarray]


class Predictions(NamedTuple):
  node_indices: np.ndarray
  labels: np.ndarray
  predictions: np.ndarray
  logits: np.ndarray


def node_classification_loss(
    logits: jnp.ndarray,
    batch: datasets.Batch,
    extra_stats: bool = False,
) -> Tuple[jnp.ndarray, LogsDict]:
  """Gets node-wise classification loss and statistics."""
  log_probs = jax.nn.log_softmax(logits)
  loss = -jnp.sum(log_probs * batch.node_labels, axis=-1)

  num_valid = jnp.sum(batch.label_mask)
  labels = jnp.argmax(batch.node_labels, axis=-1)
  is_correct = (jnp.argmax(log_probs, axis=-1) == labels)
  num_correct = jnp.sum(is_correct * batch.label_mask)
  loss = jnp.sum(loss * batch.label_mask) / (num_valid + 1e-8)
  accuracy = num_correct / (num_valid + 1e-8)

  entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1))

  stats = {
      'classification_loss': loss,
      'prediction_entropy': entropy,
      'accuracy': accuracy,
      'num_valid': num_valid,
      'num_correct': num_correct,
  }
  if extra_stats:
    for k in range(1, 6):
      stats[f'top_{k}_correct'] = topk_correct(logits, labels,
                                               batch.label_mask, k)
  return loss, stats


def get_predictions_labels_and_logits(
    logits: jnp.ndarray,
    batch: datasets.Batch,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Gets prediction labels and logits."""
  mask = batch.label_mask > 0.
  indices = batch.node_indices[mask]
  logits = logits[mask]
  predictions = jnp.argmax(logits, axis=-1)
  labels = jnp.argmax(batch.node_labels[mask], axis=-1)
  return indices, predictions, labels, logits


def topk_correct(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    valid_mask: jnp.ndarray,
    topk: int,
) -> jnp.ndarray:
  """Calculates top-k accuracy."""
  pred_ranking = jnp.argsort(logits, axis=1)[:, ::-1]
  pred_ranking = pred_ranking[:, :topk]
  is_correct = jnp.any(pred_ranking == labels[:, jnp.newaxis], axis=1)
  return (is_correct * valid_mask).sum()


def ensemble_predictions_by_probability_average(
    predictions_list: Sequence[Predictions]) -> Predictions:
  """Ensemble predictions by ensembling the probabilities."""
  _assert_consistent_predictions(predictions_list)
  all_probs = np.stack([
      jax.nn.softmax(predictions.logits, axis=-1)
      for predictions in predictions_list
  ],
                       axis=0)
  ensembled_logits = np.log(all_probs.mean(0))
  return predictions_list[0]._replace(
      logits=ensembled_logits, predictions=np.argmax(ensembled_logits, axis=-1))


def get_accuracy_dict(predictions: Predictions) -> Dict[str, float]:
  """Returns the accuracy dict."""
  output_dict = {}
  output_dict['num_valid'] = predictions.predictions.shape[0]
  matches = (predictions.labels == predictions.predictions)
  output_dict['accuracy'] = matches.mean()

  pred_ranking = jnp.argsort(predictions.logits, axis=1)[:, ::-1]
  for k in range(1, 6):
    matches = jnp.any(
        pred_ranking[:, :k] == predictions.labels[:, None], axis=1)
    output_dict[f'top_{k}_correct'] = matches.mean()
  return output_dict


def bgrl_loss(
    first_online_predictions: jnp.ndarray,
    second_target_projections: jnp.ndarray,
    second_online_predictions: jnp.ndarray,
    first_target_projections: jnp.ndarray,
    symmetrize: bool,
    valid_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, LogsDict]:
  """Implements BGRL loss."""
  first_side_node_loss = jnp.sum(
      jnp.square(
          _l2_normalize(first_online_predictions, axis=-1) -
          _l2_normalize(second_target_projections, axis=-1)),
      axis=-1)
  if symmetrize:
    second_side_node_loss = jnp.sum(
        jnp.square(
            _l2_normalize(second_online_predictions, axis=-1) -
            _l2_normalize(first_target_projections, axis=-1)),
        axis=-1)
    node_loss = first_side_node_loss + second_side_node_loss
  else:
    node_loss = first_side_node_loss
  loss = (node_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)
  return loss, dict(bgrl_loss=loss)


def get_corrupted_view(
    graph: jraph.GraphsTuple,
    feature_drop_prob: float,
    edge_drop_prob: float,
    rng_key: jnp.ndarray,
) -> jraph.GraphsTuple:
  """Returns corrupted graph view."""
  node_key, edge_key = jax.random.split(rng_key)

  def mask_feature(x):
    mask = jax.random.bernoulli(node_key, 1 - feature_drop_prob, x.shape)
    return x * mask

  # Randomly mask features with fixed probability.
  nodes = jax.tree_map(mask_feature, graph.nodes)

  # Simulate dropping of edges by changing genuine edges to self-loops on
  # the padded node.
  num_edges = graph.senders.shape[0]
  last_node_idx = graph.n_node.sum() - 1
  edge_mask = jax.random.bernoulli(edge_key, 1 - edge_drop_prob, [num_edges])
  senders = jnp.where(edge_mask, graph.senders, last_node_idx)
  receivers = jnp.where(edge_mask, graph.receivers, last_node_idx)
  # Note that n_edge will now be invalid since edges in the middle of the list
  # will correspond to the final graph. Set n_edge to None to ensure we do not
  # accidentally use this.
  return graph._replace(
      nodes=nodes,
      senders=senders,
      receivers=receivers,
      n_edge=None,
  )


def _assert_consistent_predictions(predictions_list: Sequence[Predictions]):
  first_predictions = predictions_list[0]
  for predictions in predictions_list:
    assert np.all(predictions.node_indices == first_predictions.node_indices)
    assert np.all(predictions.labels == first_predictions.labels)
    assert np.all(
        predictions.predictions == np.argmax(predictions.logits, axis=-1))


def _l2_normalize(
    x: jnp.ndarray,
    axis: Optional[int] = None,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
  return x * jax.lax.rsqrt(
      jnp.sum(jnp.square(x), axis=axis, keepdims=True) + epsilon)
