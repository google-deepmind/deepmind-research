# Code for Normalizer-Free ResNets (Brock, De, Smith, ICLR2021)
This repository contains code for the ICLR 2021 paper
"Characterizing signal propagation to close the performance gap in unnormalized
ResNets," by Andrew Brock, Soham De, and Samuel L. Smith.


## Running this code
Install using pip install -r requirements.txt and use one of the experiment.py
files in combination with [JAXline](https://github.com/deepmind/jaxline) to
train models. Optionally copy test.py into
a dir one level up and run it to ensure you can take a single experiment step
with fake data.

Note that you will need a local copy of ImageNet compatible with the TFDS format
used in dataset.py in order to train on ImageNet.

## Giving Credit

If you use this code in your work, we ask you to cite this paper:

```
@inproceedings{brock2021characterizing,
  author={Andrew Brock and Soham De and Samuel L. SMith},
  title={Characterizing signal propagation to close the performance gap in
  unnormalized ResNets},
  booktitle={9th International Conference on Learning Representations, {ICLR}},
  year={2021}
}
```
