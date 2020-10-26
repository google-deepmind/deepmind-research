# ScratchGAN

This is the example code for the following NeurIPS 2019 paper. If you use the
code here please cite this paper:

    @article{DBLP:journals/corr/abs-1905-09922,
      author    = {Cyprien de Masson d'Autume and
                   Mihaela Rosca and
                   Jack W. Rae and
                   Shakir Mohamed},
      title     = {Training language GANs from Scratch},
      journal   = {CoRR},
      volume    = {abs/1905.09922},
      year      = {2019},
      url       = {http://arxiv.org/abs/1905.09922},
      archivePrefix = {arXiv},
      eprint    = {1905.09922},
      timestamp = {Wed, 29 May 2019 11:27:50 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-09922},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }


## Contents

The code contains:

  * `generators.py`: implementation of the generator.
  * `discriminator_nets.py`: implementation of the discriminator.
  * `eval_metrics.py`: implementation of the FED metric.
  * `losses.py`: implementation of the RL loss for the generator.
  * `reader.py`: data reader / tokenizer.
  * `experiment.py`: main training script.

The data contains:

  * `{train,valid,test}.json`: the EMNLP2017 News dataset.
  * `glove_emnlp2017.txt`: the relevant subset of GloVe embeddings.

## Running

    ./scratchgan/run.sh
