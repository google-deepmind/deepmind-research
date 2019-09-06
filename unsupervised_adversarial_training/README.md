# Unsupervised Adversarial Training (UAT)

This repository contains the trained model and dataset used for Unsupervised
Adversarial Training (UAT) from the paper
[Are Labels Required for Improving Adversarial Robustness?](https://arxiv.org/abs/1905.13725)

This is not an official Google product.

## Contents

This repo serves two primary functions:

* Data release: We share indices for the 80 Million Tiny Images Dataset subset
used in our experiments, and a utility for loading the data.
* Model release: We have released our top-performing model on TF-Hub, and
include an example demonstrating how to use it.

## Running the code

### Using the model

Our model is available via
[TF-Hub](https://tfhub.dev/deepmind/unsupervised-adversarial-training/cifar10/wrn_106/1).
For example usage, refer to `quick_eval_cifar.py`. The preferred method of
running this script is through `run.sh`, which will set up a virtual
environment, install the dependendencies, and run the evaluation script, which
will print the adversarial accuracy of the model.

```bash
cd /path/to/deepmind_research
unsupervised_adversarial_training/run.sh
```

### Viewing the dataset

First, download the 80 Million Tiny Images Dataset image binary from the
official web page: http://horatio.cs.nyu.edu/mit/tiny/data/index.html

Note this file is very large, and requires 227 GB of disc space.

The file `tiny_200K_idxs.txt` indicates which images from the dataset form the
80M@200K training set used in the paper. For example usage, refer to
`save_example_images.py`.

To view example images from this dataset, use the command:

```bash
cd /path/to/deepmind_research
python -m unsupervised_adversarial_training.save_example_images \
  --data_bin_path=/path/to/tiny_images.bin
```

This will save the first 100 images to the directory
`unsupervised_adversarial_training/images`.

## Citing this work

If you use this code in your work, please cite the accompanying paper:

```
@article{uat2019,
  title={Are Labels Required for Improving Adversarial Robustness?},
  author={Uesato, Jonathan and Alayrac, Jean-Baptiste, and Huang, Po-Sen and
  Stanforth, Robert and Fawzi, Alhussein and Kohli, Pushmeet},
  journal={arXiv preprint arXiv:1905.13725},
  year={2019}
}
```
