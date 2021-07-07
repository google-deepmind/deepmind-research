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
r"""Tool to visualize graphs.

You need to have the command line tool `dot` installed locally, for example by
`sudo apt-get install graphviz`.

Example usage:
python visualize_graph.py \
  --logtostderr --graph_ids=0:48 --truncate_limit=500 --layout=fdp
"""

import html
import os
import textwrap

from absl import app
from absl import flags
from absl import logging

from wikigraphs.data import io_tools
from wikigraphs.data import paired_dataset as pd


FLAGS = flags.FLAGS
flags.DEFINE_string('subset', 'valid', 'Which subset to choose graphs from.')
flags.DEFINE_string('graph_ids', '', 'A comma-separated string of graph IDs'
                    ' (0-based), for example `1,2,3`.  Or alternatively a'
                    ' range, e.g. `0:10` which is equivalent to'
                    ' `0,1,2,3,...,9`.')
flags.DEFINE_string('version', 'max256', 'Which version of data to load.')
flags.DEFINE_string('data_dir', '', 'Path to a directory that contains the raw'
                    ' paired data, if provided.')
flags.DEFINE_string('output_dir', '/tmp/graph_vis', 'Output directory to save'
                    ' the visualized graphs.')
flags.DEFINE_integer('truncate_limit', -1, 'Maximum length for graph nodes in'
                     ' visualization.')
flags.DEFINE_string('layout', 'fdp', 'Which one of the dot layout to use.')


def truncate(s: str) -> str:
  if FLAGS.truncate_limit > 0 and len(s) > FLAGS.truncate_limit:
    s = s[:FLAGS.truncate_limit] + '...'
  return s


def format_label(s: str, width: int = 40) -> str:
  """Format a node / edge label."""
  s = io_tools.normalize_freebase_string(s)
  s = truncate(s)
  lines = s.split('\\n')
  output_lines = []
  for line in lines:
    line = html.escape(line)
    if width > 0:
      output_lines += textwrap.wrap(line, width)
    else:
      output_lines.append(line)
  return '<' + '<br/>'.join(output_lines) + '>'


def graph_to_dot(graph_text_pair: io_tools.GraphTextPair) -> str:
  """Convert a graph to a dot file."""
  dot = ['digraph {', 'node [shape=rect];']
  graph = pd.Graph.from_edges(graph_text_pair.edges)
  center_node_id = graph.node2id(graph_text_pair.center_node)

  for i, n in enumerate(graph.nodes()):
    color = '#f5dc98' if i == center_node_id else (
        '#b0ffad' if not(n[0] == '"' and n[-1] == '"') else '#ffffff')
    label = format_label(n)
    dot.append(f'{i} [ label = {label}, fillcolor="{color}", style="filled"];')

  for i, j, e in graph.edges():
    dot.append(f'{i} -> {j} [ label = {format_label(e, width=0)} ];')
  dot.append('}')
  return '\n'.join(dot)


def visualize_graph(graph_text_pair: io_tools.GraphTextPair,
                    graph_id: int,
                    output_dir: str):
  """Visualize a graph and save the visualization to the specified directory."""
  dot = graph_to_dot(graph_text_pair)
  output_file = os.path.join(output_dir, f'{graph_id}.dot')
  logging.info('Writing output to %s', output_file)
  with open(output_file, 'w') as f:
    f.write(dot)
  pdf_output = os.path.join(output_dir, f'{graph_id}.pdf')
  os.system(f'dot -K{FLAGS.layout} -Tpdf -o {pdf_output} {output_file}')


def main(_):
  logging.info('Loading the %s set of data.', FLAGS.subset)
  pairs = list(pd.RawDataset(subset=FLAGS.subset,
                             data_dir=FLAGS.data_dir or None,
                             shuffle_data=False,
                             version=FLAGS.version))
  logging.info('Loaded %d graph-text pairs.')

  if ':' in FLAGS.graph_ids:
    start, end = [int(i) for i in FLAGS.graph_ids.split(':')]
    graph_ids = list(range(start, end))
  else:
    graph_ids = [int(i) for i in FLAGS.graph_ids.split(',')]
  logging.info('Visualizing graphs with ID %r', graph_ids)

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  for gid in graph_ids:
    visualize_graph(pairs[gid], gid, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
