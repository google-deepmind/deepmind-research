# ScratchGAN

This is the example code for the following NeurIPS 2019 paper. If you use the
code here please cite this paper:

> Cyprien de Masson d'Autume, Mihaela Rosca, Jack Rae, Shakir Mohamed
  *Training Language GANs from Scratch*.  NeurIPS 2019.  [\[arXiv\]](https://arxiv.org/abs/1905.09922).


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

Place the data files in the directory specified by `data_dir` flag.

Create and activate a virtual environment if needed:

    virtualenv scratchgan-venv
    source scratchgan-venv/bin/activate

Install requirements:

    pip install -r scratchgan/requirements.txt

Run training and evaluation jobs:

    python2 scratchgan.experiment.py --mode="train" &
    python2 scratchgan.experiment.py --mode="evaluate_pair" &

The evaluation code is designed to run in parallel with the training.

The training code saves checkpoints periodically, the evaluation code
looks for new checkpoints and evaluate them.
