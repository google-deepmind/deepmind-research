# AlphaFold

This package provides an implementation of the contact prediction network,
associated model weights and CASP13 dataset as published in Nature.

Any publication that discloses findings arising from using this source code must
cite *AlphaFold: Protein structure prediction using potentials from deep
learning* by Andrew W. Senior, Richard Evans, John Jumper, James Kirkpatrick,
Laurent Sifre, Tim Green, Chongli Qin, Augustin Žídek, Alexander W. R. Nelson,
Alex Bridgland, Hugo Penedones, Stig Petersen, Karen Simonyan, Steve Crossan,
Pushmeet Kohli, David T. Jones, David Silver, Koray Kavukcuoglu, Demis Hassabis.

## Setup

### Dependencies

*   Python 3.6+.
*   [Abseil 0.8.0+](https://github.com/abseil/abseil-py)
*   [Numpy 1.16+](https://numpy.org)
*   [Six 1.12+](https://pypi.org/project/six/)
*   [Sonnet 1.35+](https://github.com/deepmind/sonnet)
*   [TensorFlow 1.14](https://tensorflow.org). Not compatible with TensorFlow
    2.0+.
*   [TensorFlow Probability 0.7.0](https://www.tensorflow.org/probability)

You can set up Python virtual environment with these dependencies inside the
forked `deepmind_research` repository using:

```shell
python3 -m venv alphafold_venv
source alphafold_venv/bin/activate
pip install -r alphafold_casp13/requirements.txt
```

### Input data

The dataset can be downloaded from
[Google Cloud Storage](https://console.cloud.google.com/storage/browser/alphafold_casp13_data).

Download it e.g. using `wget`:

```shell
wget https://storage.googleapis.com/alphafold_casp13_data/casp13_data.zip
```

The zip file contains 1 directory for each CASP13 target and a `LICENSE.md`
file. Each target directory contains the following files:

1.  `TARGET.tfrec` file. This is a
    [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) file
    with serialized tf.train.Example protocol buffers that contain the features
    needed to run the model.
1.  `contacts/TARGET.pickle` file(s) with the predicted distogram.
1.  `contacts/TARGET.rr` file(s) with the contact map derived from the predicted
    distogram. The RR format is described on the
    [CASP website](http://predictioncenter.org/casp13/index.cgi?page=format#RR).

Note that for **T0999** the target was manually split based on hits in HHSearch
into 5 sub-targets, hence there are 5 distograms
(`contacts/T0999s{1,2,3,4,5}.pickle`) and 5 RR files
(`contacts/T0999s{1,2,3,4,5}.rr`).

The `contacts/` folder is not needed to run the model, these files are included
only for convenience so that you don't need to run the inference for CASP13
targets to get the contact map.

### Model checkpoints

The model checkpoints can be downloaded from
[Google Cloud Storage](https://console.cloud.google.com/storage/browser/alphafold_casp13_data).

Download them e.g. using `wget`:

```shell
wget https://storage.googleapis.com/alphafold_casp13_data/alphafold_casp13_weights.zip
```

The zip file contains:

1.  A directory `873731`. This contains the weights for the distogram model.
1.  A directory `916425`. This contains the weights for the background distogram
    model.
1.  A directory `941521`. This contains the weights for the torsion model.
1.  `LICENSE.md`. The model checkpoints have a non-commercial license which is
    defined in this file.

Each directory with model weights contains a number of different model
configurations. Each model has a config file and associated weights. There is
only one torsion model. Each model directory also contains a stats file that is
used for feature normalization specific to that model.

## Distogram prediction

### Running the system

You can use the `run_eval.sh` script to run the entire Distogram prediction
system. There are a few steps you need to start with:

1.  Download the input data as described above. Unpack the data in the
    directory with the code.
1.  Download the model checkpoints as described above. Unpack the data.
1.  In `run_eval.sh` set the following:
    *   `DISTOGRAM_MODEL` to the path to the directory with the distogram model.
    *   `BACKGROUND_MODEL` to the path to the directory with the background
        model.
    *   `TORSION_MODEL` to the path to the directory with the torsion model.
    *   `TARGET` to the path to the directory with the target input data.

Then run `alphafold_casp13/run_eval.sh` from the `deepmind_research` parent
directory (you will get errors if you try running `run_eval.sh` directly from
the `alphafold_casp13` directory).

The contact prediction works in the following way:

1.  4 replicas (by *replica* we mean a configuration file describing the network
    architecture and a snapshot with the network weights), each with slightly
    different model configuration, are launched to predict the distogram.
1.  4 replicas, each with slightly different model configuration are launched to
    predict the background distogram.
1.  1 replica is launched to predict the torsions.
1.  The predictions from the different replicas are averaged together using
    `ensemble_contact_maps.py`.
1.  The predictions for the 64 × 64 distogram crops are pasted together using
    `paste_contact_maps.py`.

When running `run_eval.sh` the output has the following directory structure:

*   **distogram/**: Contains 4 subfolders, one for each replica. Each of these
    contain the predicted ASA, secondary structure and a pickle file with the
    distogram for each crop. It also contains an `ensemble` directory with the
    ensembled distograms.
*   **background_distogram/**: Contains 4 subfolders, one for each replica. Each
    of these contain a pickle file with the background distogram for each crop.
    It also contains an `ensemble` directory with the ensembled background
    distograms.
*   **torsion/**: Contains 1 subfolder as there was only a single replica. This
    folder contains contains the predicted ASA, secondary structure, backbone
    torsions and a pickle file with the distogram for each crop. It also
    contains an `ensemble` directory with the ensembled torsions.
*   **pasted/**: Contains distograms obtained from the ensembled distograms by
    pasting. An RR contact map file is computed from this pasted distogram.
    **This is the final distogram that was used in the subsequent AlphaFold
    folding pipeline in CASP13.**

## Data splits

We used a version of [PDB](https://www.rcsb.org/) downloaded on 2018-03-15. The
train/test split can be found in the `train_domains.txt` and `test_domains.txt`
files.

Disclaimer: This is not an official Google product.
