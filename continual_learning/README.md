# Continual learning with pre-trained encoders and ensembles of classifiers
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/continual_learning/encoders_and_ensembles.ipynb)

This repository contains a notebook implementation of a classifier ensemble memory model that mitigates catastrophic forgetting.

The code was written by Murray Shanahan.

The model comprises
*   a pre-trained encoder, trained on a different dataset from the target dataset, and
*   a memory with fixed randomised keys and k-nearest neighbour lookup, where
*   each memory location stores the parameters of a trainable local classifier, and
*   the ensemble's output is the mean output of the k selected classifiers weighted according to the distance of their keys from the encoded input

The model is demonstrated on MNIST, where the encoder is pre-trained on Omniglot. The continual learning setting is
*   Task-free. The models doesn't know about task boundaries
*   Online. The dataset is ony seen once, and there are no epochs
*   Incremental class learning. Evaluation is always on 10-way classification

The code accompanies the paper:

Shanahan, M., Kaplanis, C. & Mitrovic, J. (2021). Encoders and Ensembles for Task-Free Continual Learning. ArXiv preprint: https://arxiv.org/abs/2105.13327

## Running the experiments

The easiest way to run the code is using the publicly available [Colab](https://colab.research.google.com) kernel. Colaboratory is a free Jupyter notebook environment provided by Google that requires no setup and runs entirely in the cloud. (A GPU runtime is needed to train in a reasonable time.) The notebook is self-contained, and will load all necessary libraries automatically if run in Colaboratory.

Click "Run all" in the "Runtime" menu to train on 5-way split MNIST ("high data" setting), as described in the paper. Adjusting the "schedule_type" in the config will allow you to try out different benchmarks, such as a 10-way split.

## Contact

If you have any feedback, or would like to get in touch regarding the code or the architecture, you can reach out to mshanahan@deepmind.com.

## Disclaimer

This is not an officially supported Google product.
