# DeepMind entry for MAG240M-LSC

This repository contains DeepMind's entry to the [MAG240M-LSC](https://ogb.stanford.edu/kddcup2021/mag240m/) (academic graph) track of the
[OGB Large-Scale Challenge](https://ogb.stanford.edu/kddcup2021/) (OGB-LSC).

For full details regarding this entry, please see our [technical report](https://arxiv.org/abs/2107.09422).

## DeepMind MAG Team ("Academic")

(in alphabetical order)

- Ravichandra Addanki
- Peter Battaglia
- David Budden
- Andreea Deac
- Jonathan Godwin
- Thomas Keck
- Alvaro Sanchez-Gonzalez
- Jacklynn Stott
- Shantanu Thakoor
- Petar Veličković

## Performance

Our final test set performance was achieved by pooling an ensemble of 10 folds.
See [technical report](https://arxiv.org/abs/2107.09422) for details.

Each model was trained for < 72 hours using 4x Google Cloud TPUv4 and 1x AMD
EPYC 7B12 64-core CPU @2.25GHz.

Inference takes < 12 hours on 4x NVIDIA V100 16GB GPU and 1x Intel Xeon Gold
6148 20-core CPU @2.40GHz.

# Running our model

## Setup

You can set up Python virtual environment (you might need to install the
`python3-venv` package first) with all needed dependencies inside the forked
`deepmind_research` repository using:

```bash
python3 -m venv /tmp/mag_venv
source /tmp/mag_venv/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -r ogb_lsc/mag/requirements.txt
```

Use the following command to get a jaxlib version built compatible with V100 GPUs.
```bash
pip install --upgrade jax jaxlib==0.1.67+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
See https://github.com/google/jax/issues/5231 for details.


## Download and pre-process data


**1. Download the dataset using the contest toolkit ([here](https://ogb.stanford.edu/kddcup2021/mag240m/#dataset)) to a local directory
`ROOT`.**

**2. Run this script to reorganize the data into a flat directory structure with
transparent names.**

```bash
/bin/bash organize_data.sh -r ROOT
```

Once this completes, a new directory `ROOT/mag240m_kddcup2021/raw` will be
created, with contents:

- `node_feat.npy`
- `node_label.npy`
- `node_year.npy`
- `author_affiliated_with_institution_edges.npy`
- `author_writes_paper_edges.npy`
- `paper_cites_paper_edges.npy`
- `train_idx.npy`
- `valid_idx.npy`
- `test_idx.npy`

We refer to this as the "raw" data.

**3. Run the preprocessing code.**

```bash
/bin/bash run_preprocessing.sh -r ROOT
```

The pre-processing is both time- and memory-consuming, and should only be run
to verify the full pipeline. You can download the pre-processed data using the
following script, for use in training and evaluating models:

```bash
python3 download_mag.py --task_root=${HOME}/mag --payload="data"
```


## Reproducing our final results

We have provided pre-trained weights of our final submission for convenience.
They can be downloaded with:

```bash
python3 download_mag.py --task_root=${HOME}/mag --payload="models"
```

Then to reproduce our final results, please run:

```bash
/bin/bash run_preprocessing.sh -r ${HOME}/mag/
```

## Retraining our model

Disclaimer: This script is provided for illustrative purposes. It is not
practical for actual training since it only uses a single machine, and likely
requires reducing the batch size and/or model size to fit on a single GPU.

To train a model, please run:

```bash
/bin/bash run_training.sh -r ${HOME}/mag/
```

To simply validate that the code is running correctly on your hardware setup,
consider setting `debug=True` in `config.py`, which trains a smaller model.


# Citation

To cite this work (together with our PCQM4M-LSC entry):

```latex
@article{deepmind2021ogb,
  author = {Ravichandra Addanki and Peter Battaglia and David Budden and Andreea
    Deac and Jonathan Godwin and Thomas Keck and Wai Lok Sibon Li and Alvaro
    Sanchez-Gonzalez and Jacklynn Stott and Shantanu Thakoor and Petar
    Veli\v{c}kovi\'{c}},
  title = {Large-scale graph representation learning with very deep GNNs and
    self-supervision},
  year = {2021},
  journal={arXiv preprint arXiv:2107.09422},
}
```

Our technical report can be found [here](https://arxiv.org/abs/2107.09422).
