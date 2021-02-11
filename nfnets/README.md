# Code for Normalizer-Free Networks
This repository contains code for the ICLR 2021 paper
"Characterizing signal propagation to close the performance gap in unnormalized
ResNets," by Andrew Brock, Soham De, and Samuel L. Smith, and the arXiv preprint
"High-Performance Large-Scale Image Recognition Without Normalization" by
Andrew Brock, Soham De, Samuel L. Smith, and Karen Simonyan.


## Running this code
Install using pip install -r requirements.txt and use one of the experiment.py
files in combination with [JAXline](https://github.com/deepmind/jaxline) to
train models. Optionally copy test.py into
a dir one level up and run it to ensure you can take a single experiment step
with fake data.

Note that you will need a local copy of ImageNet compatible with the TFDS format
used in dataset.py in order to train on ImageNet.

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
  journal={arXiv preprint arXiv:},
  year={2021}
}
```

