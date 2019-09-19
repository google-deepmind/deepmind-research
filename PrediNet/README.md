# The PrediNet Architecture and Relations Game Datasets

This repository contains a notebook implementation of the PrediNet architecture,
as well as the baselines and sample datasets.

The code was written by Murray Shanahan, Marta Garnelo and Kyriacos Nikiforou.

The `PrediNet.ipynb` notebook includes an overview of the PrediNet
architecture and the code to reproduce the *multi-task experiment* presented in the paper
[*An Explicitly Relational Neural Network Architecture*](https://arxiv.org/pdf/1905.10307.pdf).
Additional details of the model and experiments can be found in the paper.

Six `.npz` files contain downsampled versions of the datasets required to train
and evaluate the various models on the multi-task experiment. The training set (`*_pentos.npz`),
containing pentominoes, is a NPZ NumPy archive with the following fields:

*  `images`: (50000 x 36 x 36, 3) Images in RGB.
*  `labels`: (50000 x 2) Labels for the images.
*  `tasks`:  (50000 x 1) Task ids that denote which relation must hold between the
   objects in the images.

Two additional `.npz` files are provided for each task, one with hexominoes (`*_hexos.npz`)
and one with striped squares (`*_stripes.npz`). These can be used for testing.

The full Relations Game datasets composed of 250,000 samples for pentominoes - and 50,000 samples
for hexominoes and stripes - can be found [here](https://console.cloud.google.com/storage/browser/relations-game-datasets)

## Running the experiments

The easiest way to run the code is using the publicly available [Colab](https://colab.research.google.com) kernel.
You can run simply by clicking on [PrediNet Notebook](https://colab.research.google.com/github/deepmind/deepmind-research/blob/master/PrediNet/PrediNet.ipynb)

Colaboratory is a free Jupyter notebook environment provided by Google that requires no setup and runs entirely in the cloud.
You will need the following dependencies to run the code, with instructions on how to install them.
The code was tested with the versions shown in brackets

*   TensorFlow 2 (2.0.0-rc0 with gpu support) - not installed
You can install TensorFlow 2.0 beta version by running the following  command in
Colab:

```
!pip install "tensorflow-gpu>=2.0.0rc0" --pre
```

*   Sonnet 2 (2.0.0b0) - not installed
You can install Sonnet 2 by running the following comment in Colab.

```
!pip install "dm-sonnet>=2.0.0b0" --pre
```

Alternatively, you can open the `.ipynb` files using
[Jupyter notebook](http://jupyter.org/install.html). If you do this you will
also have to set up a local kernel that includes the libraries above.

## Citing Predicate Networks

If you use this code in your work, please cite us as follows:

Shanahan, M., Nikiforou, K., Creswell, A., Kaplanis, C., Barrett, D.,
& Garnelo, M. (2019). *An Explicitly Relational Neural Network Architecture*.
arXiv preprint arXiv:1905.10307.


## Contact

If you have any feedback, or would like to get in touch with us, you can reach out to us
at mshanahan@google.com and knikiforou@google.com.

## Disclaimer

This is not an officially supported Google product.
