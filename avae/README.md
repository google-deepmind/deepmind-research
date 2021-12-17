# The Autoencoding Variational Autoencoder

This is the code for the models in NeurIPS Submission [AVAE](https://papers.nips.cc/paper/2020/file/ac10ff1941c540cd87c107330996f4f6-Paper.pdf)

Folder contains code to train AVAE model in JAX, and we will be uploading
evaluation setup soon.

Code files in the folder
 - checkpointer.py: Checkpointing abstraction
 - data_iterators.py: Datasets to be used
 - decoders.py: VAE decoder network architectures
 - encoders.py: VAE encoder network architectures
 - kl.py: KL computation between 2 gaussians
 - train.py: Function to train given ELBO, network and data
 - train_main.py: Main file to train AVAE
 - vae.py: VAE model defining various ELBOs

## Setup

To set up a Python3 virtual environment with the required dependencies, run:

```shell
python -m venv avae_env
source avae_env/bin/activate
pip install --upgrade pip
pip install -r avae/requirements.txt
```

## Running AVAE training

Following command will run AVAE training for ColorMnist dataset using MLP
network architectures.

```shell
python -m avae.train_main \
  --dataset='color_mnist' \
  --latent_dim=64 \
  --checkpoint_dir='/tmp/avae_checkpoints' \
  --checkpoint_filename='color_mnist_mlp_avae' \
  --rho=0.975 \
  --encoder='color_mnist_mlp_encoder' \
  --decoder='color_mnist_mlp_decoder'
```

## References

### Citing our work

If you use that code for your research, please consider citing our paper:

```bibtex
@article{cemgil2020autoencoding,
  title={The Autoencoding Variational Autoencoder},
  author={Cemgil, Taylan and Ghaisas, Sumedh and Dvijotham, Krishnamurthy and Gowal, Sven and Kohli, Pushmeet},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
