# WikiGraphs

This package provides tools to download the [WikiGraphs dataset](https://arxiv.org/abs/2107.09556)
[1], collected by pairing each Wikipedia article from [WikiText-103](https://arxiv.org/pdf/1609.07843.pdf)
[2] with a knowledge graph (a subgraph from [Freebase knowledge graph](https://dl.acm.org/doi/pdf/10.1145/1376616.1376746?casa_token=H2ggPTDMoZUAAAAA:7wBhO9hnOzNKoJyMH0PcpVQZ6Vg6Ud6hObiDJTzLCGRiBwmYFjOFSXrG5PcKLStu5-n4_OfkPJtbisQ)
[3]). The baseline code to reproduce results in [1] is included as well. We hope
this can spur more interest in developing models that can generate long text
conditioned on graph and generate graphs given text.

## Setup Jax environment

[Jax](https://github.com/google/jax#installation),
[Haiku](https://github.com/deepmind/dm-haiku#installation),
[Optax](https://optax.readthedocs.io/en/latest/#installation), and
[Jraph](https://github.com/deepmind/jraph) are needed for this package. It has
been developed and tested on python 3 with the following packages:

*   Jax==0.2.13
*   Haiku==0.0.5.dev
*   Optax==0.0.6
*   Jraph==0.0.1.dev

Other packages required can be installed via:

```bash
pip install -r requirements.txt
```

Note: you may need to use `pip3` to select pip for python 3 and `--user` option
to install the packages to avoid permission issues.

## Installation

```bash
pip install -e .
```

## Preparing the data

### Download the data

You can download and unzip the data by running the following command:

```bash
bash scripts/download.sh
```

This will put the downloaded WikiText-103 data in a temporary directory
`/tmp/data` with the tokenized WikiText-103 data in `/tmp/data/wikitext-103` and
the raw data in `/tmp/data/wikitext-103-raw`.

This script will also download our processed Freebase knowledge graph data in a
temporary directory `/tmp/data/freebase`.

### Build vocabularies

For WikiText-103, run the following command to generate a vocabulary file:

```bash
python scripts/build_vocab.py \
  --vocab_file_path=/tmp/data/wikitext-vocab.csv \
  --data_dir=/tmp/data/wikitext-103
```

You can change the default file paths but make sure you make them consistent.

### Pair Freebase graphs with WikiText

You can run the following command to pair the Freebase graphs with WikiText-103
articles:

```bash
python scripts/freebase_preprocess.py \
  --freebase_dir=/tmp/data/freebase/max256 \
  --output_dir=/tmp/data/wikigraphs/max256
```

where the `freebase_dir` `/tmp/data/freebase/max256` is the directory that
contains the Fsreebase graphs, which should have files `train.gz`, `valid.gz`
and `test.gz` in it; and `output_dir` is the directory that will contain the
generated paired Freebase-WikiText data.

Note: you may need to use `python3` to select python 3 if you have both python 2
and 3 on your system.

Given that there are the following number of articles in WikiText-103:

Subset | #articles
------ | ---------
Train  | 28472*
Valid  | 60
Test   | 60

*Official number is 28475 but we were only able to find 28472 articles in
training set.

Our dataset covers around 80% of the WikiText articles:

Max graph size               | 256   | 512   | 1024
---------------------------- | ----- | ----- | -----
\#articles in training set   | 23431 | 23718 | 23760
Trainining set coverage      | 82.3% | 83.3% | 83.5%
\#articles in validation set | 48    | 48    | 48
Validation set coverage      | 80%   | 80%   | 80%
\#articles in test set       | 43    | 43    | 43
Test set coverage            | 71.7% | 71.7% | 71.7%

### Build vocabulary for WikiGraphs

You can build the vocabulary for the graph data (the max256 version) by running
the following command:

```bash
python scripts/build_vocab.py \
  --vocab_file_path=/tmp/data/graph-vocab.csv \
  --data_dir=/tmp/data/wikigraphs \
  --version=max256 \
  --data_type=graph \
  --threshold=15
```

This gives us a vocabulary of size 31,087, with each token included in the
vocabulary appearing at least 15 times.

You also need to build a separate text vocabulary for the WikiGraphs data, as
our training set does not cover 100% of WikiText-103.

```bash
python scripts/build_vocab.py \
  --vocab_file_path=/tmp/data/text-vocab.csv \
  --data_dir=/tmp/data/wikigraphs \
  --version=max256 \
  --data_type=text \
  --threshold=3
```

Here we choose threshold 3 which is also used by the original WikiText-103 data,
this gives us a vocabulary size of 238,068, only slightly smaller than the
original vocabulary size.

Note that when loading these vocabularies to build tokenizers, our tokenizers
will add a few extra tokens, like `<bos>`, `<pad>`, so the final vocab size
might be slightly different from the numbers above, depending on which tokenizer
you choose to use.

We only showcase how to build the vocabulary for the max256 version. The above
steps can be easily changed for the max512 and max1024 version.

## Loading the dataset

We provide JAX modules to load the WikiGraphs dataset. There are three classes
in `wikigraphs/data/paired_dataset.py`:

* `TextOnlyDataset`: loads only the text part of the WikiGraphs data
* `Bow2TextDataset`: loads text and the paired graph representated as one big
bag-of-words (BoW) on all nodes and edges from the graph
* `Graph2TextDataset`: returns text and the paired graph in which each node or
edge is represented by a BoW

Different versions of the dataset can be accessed by changing the `version`
argument in each class. For more detailed usage please refer to
`wikigraphs/data/paired_dataset_test.py`. Besides, the original WikiText dataset
can be loaded via the `Dataset` class in `wikigraphs/data/wikitext.py`.

Note: you may want to change the default data directory if you prefer to place
it elsewhere.

## Run baseline models

To quickly test-run a small model with 1 GPU:

```base
python main.py --model_type=graph2text \
  --dataset=freebase2wikitext \
  --checkpoint_dir=/tmp/graph2text \
  --job_mode=train \
  --train_batch_size=2 \
  --gnn_num_layers=1 \
  --num_gpus=1
```

To run the default baseline unconditional TransformerXL on Wikigraphs with 8
GPUs:

```base
python main.py --model_type=text \
  --dataset=freebase2wikitext \
  --checkpoint_dir=/tmp/text \
  --job_mode=train \
  --train_batch_size=64 \
  --gnn_num_layers=1 \
  --num_gpus=8
```

To run the default baseline BoW-based TransformerXL on Wikigraphs with 8
GPUs:

```base
python main.py --model_type=bow2text \
  --dataset=freebase2wikitext \
  --checkpoint_dir=/tmp/bow2text \
  --job_mode=train \
  --train_batch_size=64 \
  --gnn_num_layers=1 \
  --num_gpus=8
```

To run the default baseline Nodes-only GNN-based TransformerXL on Wikigraphs
with 8 GPUs:

```base
python main.py --model_type=bow2text \
  --dataset=freebase2wikitext \
  --checkpoint_dir=/tmp/bow2text \
  --job_mode=train \
  --train_batch_size=64 \
  --gnn_num_layers=0 \
  --num_gpus=8
```

To run the default baseline GNN-based TransformerXL on Wikigraphs with 8
GPUs:

```base
python main.py --model_type=graph2text \
  --dataset=freebase2wikitext \
  --checkpoint_dir=/tmp/graph2text \
  --job_mode=train \
  --train_batch_size=64 \
  --gnn_num_layers=1 \
  --num_gpus=8
```

We ran our experiments in the paper using 8 Nvidia V100 GPUs. Reduce the batch
size if the model does not fit into memory. To allow for batch parallization for
the GNN-based (graph2text) model, we pad graphs to the largest graph in the
batch. The full run takes almost 4 days. BoW- and nodes-based models can be
trained within 14 hours because there is no additional padding.

To evaluate the model on the validation set (this only uses 1 GPU):

```base
python main.py --model_type=graph2text \
  --dataset=freebase2wikitext \
  --checkpoint_dir=/tmp/graph2text \
  --job_mode=eval \
  --eval_subset=valid
```

To generate 960 samples from the model using the graphs in the validation set
(using 8 GPUs):

```base
python main.py --model_type=graph2text \
  --dataset=freebase2wikitext \
  --checkpoint_dir=/tmp/graph2text \
  --job_mode=sample \
  --eval_subset=valid \
  --num_gpus=8 \
  --num_samples=960
```

To compute the rBLEU score of the generated samples:

```base
python scripts/compute_bleu_score.py --dataset=freebase2wikitext \
  --checkpoint_dir=/tmp/graph2text
```

To compute the retrieval scores:

```base
python main.py --dataset=freebase2wikitext \
  --job_mode=retrieve \
  --checkpoint_dir=/tmp/graph2text
```

## Citing WikiGraphs

To cite this work:

```
@inproceedings{wang2021wikigraphs,
  title={WikiGraphs: A Wikipedia Text-Knowledge Graph Paired Dataset},
  author={Wang, Luyu and Li, Yujia and Aslan, Ozlem and Vinyals, Oriol},
  booktitle={Proceedings of the Graph-Based Methods for Natural Language Processing (TextGraphs)},
  pages={67--82},
  year={2021}
}
```

## License

All code copyright 2021 DeepMind Technologies Limited

Code is licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain a copy
of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

[WikiGraphs](https://arxiv.org/abs/2107.09556) [1] is licensed under the terms
of the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
license.

[WikiText-103 data](https://arxiv.org/pdf/1609.07843.pdf) [2] (unchanged) is
licensed by Salesforce.com, Inc. under the terms of the Creative Commons
Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license. You can find
details about CC BY-SA 4.0 at:

    https://creativecommons.org/licenses/by-sa/4.0/legalcode

[Freebase data](https://dl.acm.org/doi/pdf/10.1145/1376616.1376746?casa_token=H2ggPTDMoZUAAAAA:7wBhO9hnOzNKoJyMH0PcpVQZ6Vg6Ud6hObiDJTzLCGRiBwmYFjOFSXrG5PcKLStu5-n4_OfkPJtbisQ)
[3] is licensed by Google LLC under the terms of the Creative
Commons CC BY 4.0 license. You may obtain a copy of the License at:

    https://creativecommons.org/licenses/by/4.0/legalcode

## References

1.  L. Wang, Y. Li, O. Aslan, and O. Vinyals, "[WikiGraphs: a wikipedia -
knowledge graph paired dataset](https://arxiv.org/abs/2107.09556)",
in Proceedings of the Graph-based Methods for Natural Language Processing
(TextGraphs), pages 67-82, 2021.
2.  S. Merity, C. Xiong, J. Bradbury, and R. Socher, "[Pointer sentinel mixture
models](https://arxiv.org/pdf/1609.07843.pdf)",
arXiv: 1609.07843, 2016.
3.  K. Bollacker, C. Evans, P. Paritosh, T. Sturge, and J. Taylor,
"[Freebase: a collaboratively created graph database for structuring human
knowledge](https://dl.acm.org/doi/pdf/10.1145/1376616.1376746?casa_token=H2ggPTDMoZUAAAAA:7wBhO9hnOzNKoJyMH0PcpVQZ6Vg6Ud6hObiDJTzLCGRiBwmYFjOFSXrG5PcKLStu5-n4_OfkPJtbisQ)",
in Proceedings of ACM SIGMOD international conference on Managementof data,
pages 1247–1250, 2008.
