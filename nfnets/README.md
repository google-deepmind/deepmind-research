# Code for Normalizer-Free Networks
This repository contains code for the ICLR 2021 paper
["Characterizing signal propagation to close the performance gap in unnormalized
ResNets,"](https://arxiv.org/abs/2102.06171) by Andrew Brock, Soham De, and
Samuel L. Smith, and the arXiv preprint ["High-Performance Large-Scale Image
Recognition Without Normalization"](http://dpmd.ai/06171) by
Andrew Brock, Soham De, Samuel L. Smith, and Karen Simonyan.


## Running this code
Using `run.sh` will create and activate a virtualenv, install all necessary
dependencies and run a test program to ensure that you can import all the
modules and take a single experiment step. To train with this code, use this
virtualenv and use one of the experiment.py files in combination with
[JAXline](https://github.com/deepmind/jaxline). The provided
demo Colab can be run online, or by starting a jupyter notebook within this
virtualenv.

Note that you will need a local copy of ImageNet compatible with the TFDS format
used in dataset.py in order to train on ImageNet.


## Pre-Trained Weights

We provide pre-trained weights for NFNet-F0 through F5 (trained without SAM),
and for NFNet-F6 trained with SAM. All models are pre-trained on ImageNet for
360 epochs at batch size 4096, and are provided as numpy files containing
parameter trees compatible with haiku. In utils.py we provide a
`load_haiku_file` function which loads these parameter trees, and
`flatten_haiku_tree` to convert these to flat dictionaries
which may prove easier to port to other frameworks. Note that we do not provide
model `states`, as these models, lacking batchnorm, do not have running stats.
Note also that the conv layer weights are in the format HWIO, so for frameworks
like PyTorch which use OIHW you'll need to swap the axes appropriately to the
layout you use.


| Model | #FLOPs | #Params | Top-1 | Top-5 | TPUv3 Train | GPU Train | link |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
F0 | 12.38B | 71.5M | 83.6 | 96.8 | 73.3ms | 56.7ms | [haiku](https://storage.googleapis.com/dm-nfnets/F0_haiku.npz)
F1 | 35.54B | 132.6M | 84.7 | 97.1 | 158.5ms | 133.9ms | [haiku](https://storage.googleapis.com/dm-nfnets/F1_haiku.npz)
F2 | 62.59B | 193.8M | 85.1 | 97.3 | 295.8ms | 226.3ms | [haiku](https://storage.googleapis.com/dm-nfnets/F2_haiku.npz)
F3 | 114.76B | 254.9M | 85.7 | 97.5 | 532.2ms | 524.5ms | [haiku](https://storage.googleapis.com/dm-nfnets/F3_haiku.npz)
F4 | 215.24B | 316.1M | 85.9 | 97.6 | 1033.3ms | 1190.6ms | [haiku](https://storage.googleapis.com/dm-nfnets/F4_haiku.npz)
F5 | 289.76B | 377.2M | 86.0 | 97.6 | 1398.5ms | 2177.1ms | [haiku](https://storage.googleapis.com/dm-nfnets/F5_haiku.npz)
F6+SAM | 377.28B | 438.4M | 86.5 | 97.9 | 2774.1ms | - | [haiku](https://storage.googleapis.com/dm-nfnets/F6_haiku.npz)


## Demo Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/deepmind-research/blob/master/nfnets/nfnet_demo_colab.ipynb)

We also include a Colab notebook with a demo showing how to run an NFNet to
classify an image.


## Giving Credit

If you use this code in your work, we ask you to please cite one or both of the
following papers.

The reference for the Normalizer-Free structure and NF-ResNets or NF-Regnets:

```
@inproceedings{brock2021characterizing,
  author={Andrew Brock and Soham De and Samuel L. Smith},
  title={Characterizing signal propagation to close the performance gap in
  unnormalized ResNets},
  booktitle={9th International Conference on Learning Representations, {ICLR}},
  year={2021}
}
```

The reference for Adaptive Gradient Clipping (AGC) and the NFNets models:

```
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:2102.06171},
  year={2021}
}
```

