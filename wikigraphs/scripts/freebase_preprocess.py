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
"""Preprocess freebase data and pair with wikitext."""

import os

from absl import app
from absl import flags
from absl import logging

from wikigraphs.data import io_tools
from wikigraphs.data import wikitext


FLAGS = flags.FLAGS
flags.DEFINE_string('freebase_dir', '', 'Directory that containns Freebase'
                    ' graphs.')
flags.DEFINE_string('output_dir', '', 'Path to output directory to store the'
                    ' paired dataset.')


def pair_graphs_with_wikitext(subset: str, graph_dir: str, output_dir: str):
  """Pair graphs with wikitext articles, and write to output directory."""
  logging.info('Pairing graphs from the %s set from %s with wikitext.',
               subset, graph_dir)
  graphs = list(io_tools.graphs_from_file(
      os.path.join(graph_dir, f'{subset}.gz')))
  title2graph = {
      io_tools.normalize_freebase_string(g.title).replace(' ', ''): g
      for g in graphs}
  n_graphs = len(graphs)

  # Use raw version of the wikitext data as the tokenized version has <unk> in
  # titles which is bad for matching.  We will handle the <unk>s through the
  # tokenizer to make sure our data are equivalent to that of the tokenized
  # version of wikitext-103.
  wikitext_articles = list(wikitext.RawDataset(subset=subset, version='raw'))
  n_wiki = len(wikitext_articles)
  logging.info('Loaded %d graphs and %d wikitext articles in total.',
               n_graphs, n_wiki)

  # Keep track of the article titles in the dataset.  Unfortunately wikitext-103
  # has about 1% of duplicated articles, we want to take care of that.
  retrieved_titles = set()
  pairs = []
  n_duplicates = 0
  for a in wikitext_articles:
    title = wikitext.normalize_title(a.title).replace(' ', '')
    g = title2graph.get(title, None)
    if g is not None:
      if title not in retrieved_titles:
        retrieved_titles.add(title)
        pairs.append(io_tools.GraphTextPair(
            center_node=g.center,
            title=g.title,
            edges=g.edges,
            text=a.text))
      else:
        n_duplicates += 1

  n_pairs = len(pairs)
  logging.info('Matched %d/%d = %.1f%% of wikitext articles,'
               ' and %d/%d = %.1f%% of graphs.',
               n_pairs, n_wiki, float(n_pairs) / n_wiki * 100,
               n_pairs, n_graphs, float(n_pairs) / n_graphs * 100)
  logging.info('Detected %d/%d = %.1f%% of duplicated wikitext articles.',
               n_duplicates, n_wiki, float(n_duplicates) / n_wiki * 100)

  io_tools.write_pairs_to_gzip_txt_file(
      os.path.join(output_dir, f'{subset}.gz'), pairs)


def main(_):
  for subset in ['train', 'valid', 'test']:
    pair_graphs_with_wikitext(subset, FLAGS.freebase_dir, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
