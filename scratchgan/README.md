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

Download the data and place it in the directory specified by `data_dir` flag:

    mkdir -p /tmp/emnlp2017
    curl https://storage.googleapis.com/deepmind-scratchgan-data/train.json --output /tmp/emnlp2017/train.json
    curl https://storage.googleapis.com/deepmind-scratchgan-data/valid.json --output /tmp/emnlp2017/valid.json
    curl https://storage.googleapis.com/deepmind-scratchgan-data/test.json --output /tmp/emnlp2017/test.json
    curl https://storage.googleapis.com/deepmind-scratchgan-data/glove_emnlp2017.txt --output /tmp/emnlp2017/glove_emnlp2017.txt

Create and activate a virtual environment if needed:

    virtualenv scratchgan-venv
    source scratchgan-venv/bin/activate

Install requirements:

    pip install -r scratchgan/requirements.txt

Run training and evaluation jobs:

    python2 -m scratchgan.experiment --mode="train" &
    python2 -m scratchgan.experiment --mode="evaluate_pair" &

The evaluation code is designed to run in parallel with the training.

The training code saves checkpoints periodically, the evaluation code
looks for new checkpoints and evaluate them.
