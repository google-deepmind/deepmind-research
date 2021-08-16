# Adversarial Robustness

This repository contains the code needed to evaluate models trained in
[Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples](https://arxiv.org/abs/2010.03593)
(Gowal et al., 2020) and in
[Fixing Data Augmentation to Improve Adversarial Robustness](https://arxiv.org/abs/2103.01946)
(Rebuffi et al., 2021).


## Contents

We have released our top-performing models in two formats compatible with
[JAX](https://github.com/google/jax) and [PyTorch](https://pytorch.org/).
This repository also contains our model definitions.

## Running the code

### Downloading a model

Download a model from links listed in the following table.
Clean and robust accuracies are measured on the full test set.
The robust accuracy is measured using
[AutoAttack](https://github.com/fra31/auto-attack).
The following table contains the models from **Gowal et al., 2020**.

| dataset | norm | radius | architecture | extra data | clean | robust | link |
|---|:---:|:---:|:---:|:---:|---:|---:|:---:|
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-70-16 | &#x2713; | 91.10% | 65.88% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn70-16_with.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn70-16_with.pt)
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-28-10 | &#x2713; | 89.48% | 62.80% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn28-10_with.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn28-10_with.pt)
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-70-16 | &#x2717; | 85.29% | 57.20% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn70-16_without.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn70-16_without.pt)
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-34-20 | &#x2717; | 85.64% | 56.86% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn34-20_without.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn34-20_without.pt)
| CIFAR-10 | &#8467;<sub>2</sub> | 128 / 255 | WRN-70-16 | &#x2713; | 94.74% | 80.53% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_l2_wrn70-16_with.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_l2_wrn70-16_with.pt)
| CIFAR-10 | &#8467;<sub>2</sub> | 128 / 255 | WRN-70-16 | &#x2717; | 90.90% | 74.50% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_l2_wrn70-16_without.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_l2_wrn70-16_without.pt)
| CIFAR-100 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-70-16 | &#x2713; | 69.15% | 36.88% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn70-16_with.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn70-16_with.pt)
| CIFAR-100 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-70-16 | &#x2717; | 60.86% | 30.03% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn70-16_without.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn70-16_without.pt)
| MNIST | &#8467;<sub>&infin;</sub> | 0.3 | WRN-28-10 | &#x2717; | 99.26% | 96.34% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/mnist_linf_wrn28-10_without.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/mnist_linf_wrn28-10_without.pt)

The following table contains the models from **Rebuffi et al., 2021**.

| dataset | norm | radius | architecture | extra data | clean | robust | link |
|---|:---:|:---:|:---:|:---:|---:|---:|:---:|
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-106-16 | &#x2717; | 88.50% | 64.64% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn106-16_cutmix_ddpm_v2.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn106-16_cutmix_ddpm_v2.pt)
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-70-16 | &#x2717; | 88.54% | 64.25% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn70-16_cutmix_ddpm_v2.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn70-16_cutmix_ddpm_v2.pt)
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-28-10 | &#x2717; | 87.33% | 60.75% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn28-10_cutmix_ddpm_v2.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn28-10_cutmix_ddpm_v2.pt)
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | ResNet-18 | &#x2717; | 83.53% | 56.66% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_resnet18_ddpm.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_resnet18_ddpm.pt)
| CIFAR-10 | &#8467;<sub>2</sub> | 128 / 255 | WRN-70-16 | &#x2717; | 92.41% | 80.42% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_l2_wrn70-16_cutmix_ddpm_v2.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_l2_wrn70-16_cutmix_ddpm_v2.pt)
| CIFAR-10 | &#8467;<sub>2</sub> | 128 / 255 | WRN-28-10 | &#x2717; | 91.79% | 78.80% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_l2_wrn28-10_cutmix_ddpm_v2.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_l2_wrn28-10_cutmix_ddpm_v2.pt)
| CIFAR-10 | &#8467;<sub>2</sub> | 128 / 255 | ResNet-18 | &#x2717; | 90.33% | 75.86% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_l2_resnet18_cutmix_ddpm.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_l2_resnet18_cutmix_ddpm.pt)
| CIFAR-100 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-70-16 | &#x2717; | 63.56% | 34.64% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn70-16_cutmix_ddpm.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn70-16_cutmix_ddpm.pt)
| CIFAR-100 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-28-10 | &#x2717; | 62.41% | 32.06% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn28-10_cutmix_ddpm.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn28-10_cutmix_ddpm.pt)
| CIFAR-100 | &#8467;<sub>&infin;</sub> | 8 / 255 | ResNet-18 | &#x2717; | 56.87% | 28.50% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_resnet18_ddpm.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_resnet18_ddpm.pt)

### Installing

The following has been tested using Python 3.9.2.
Using `run.sh` will create and activate a virtualenv, install all necessary
dependencies and run a test program to ensure that you can import all the
modules.

```
# Run from the parent directory.
sh adversarial_robustness/run.sh
```

To run the provided code, use this virtualenv:

```
source /tmp/adversarial_robustness_venv/bin/activate
```

You may want to edit `requirements.txt` before running `run.sh` if GPU support
is needed (e.g., use `jaxline==0.1.67+cuda111`). See JAX's installation
[instructions](https://github.com/google/jax#installation) for more details.

### Using pre-trained models

Once downloaded, a model can be evaluated by running the `eval.py` script in
either the `jax` or `pytorch` folders. E.g.:

```
cd jax
python3 eval.py \
  --ckpt=${PATH_TO_CHECKPOINT} --depth=70 --width=16 --dataset=cifar10
```

These models are also directly available within
[RobustBench](https://github.com/RobustBench/robustbench#model-zoo-quick-tour)'s
model zoo.

### Training your own model

We also provide a training pipeline that reproduces results from both
publications. This pipeline uses [Jaxline](https://github.com/deepmind/jaxline)
and is written using [JAX](https://github.com/google/jax) and
[Haiku](https://github.com/deepmind/dm-haiku). To train a model, modify the
configuration in the `get_config()` function of `jax/experiment.py` and issue
the following command from within the virtualenv created above:

```
cd jax
python3 train.py --config=experiment.py
```

The training pipeline can run with multiple worker machines and multiple devices
(either GPU or TPU). See [Jaxline](https://github.com/deepmind/jaxline) for more
details.

We do not provide a PyTorch implementation of our training pipeline. However,
you may find one on GitHub, e.g.,
[adversarial_robustness_pytorch](https://github.com/imrahulr/adversarial_robustness_pytorch)
(by Rahul Rade).

## Datasets

### Extracted dataset

Gowal et al. (2020) use samples extracted from
[TinyImages-80M](https://groups.csail.mit.edu/vision/TinyImages/).
Unfortunately, since then, the official TinyImages-80M dataset has been
withdrawn (due to the presence of offensive images). As such, we cannot provide
a download link to our extrated data until we have manually verified that all
extracted images are not offensive. If you want to reproduce our setup, consider
the generated datasets below. We are also happy to help, so feel free to reach
out to Sven Gowal directly.

### Generated datasets

Rebuffi et al. (2021) use samples generated by a Denoising Diffusion
Probabilistic Model [(DDPM; Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
to improve robustness. The DDPM is solely trained on the original training data
and does not use additional external data. The following table links to datasets
of 1M **generated** samples for CIFAR-10, CIFAR-100 and SVHN.

| dataset | model | size | link |
|---|---|:---:|:---:|
| CIFAR-10 | DDPM | 1M | [npz](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_ddpm.npz) |
| CIFAR-100 | DDPM | 1M | [npz](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_ddpm.npz) |
| SVHN | DDPM | 1M | [npz](https://storage.googleapis.com/dm-adversarial-robustness/svhn_ddpm.npz) |

To load each dataset, use NumPy. E.g.:

```
npzfile = np.load('cifar10_ddpm.npz')
images = npzfile['image']
labels = npzfile['label']
```

## Citing this work

If you use this code (or any derived code), data or these models in your work,
please cite the relevant accompanying paper:

```
@article{gowal2020uncovering,
  title={Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples},
  author={Gowal, Sven and Qin, Chongli and Uesato, Jonathan and Mann, Timothy and Kohli, Pushmeet},
  journal={arXiv preprint arXiv:2010.03593},
  year={2020},
  url={https://arxiv.org/pdf/2010.03593}
}
```

and/or

```
@article{rebuffi2021fixing,
  title={Fixing Data Augmentation to Improve Adversarial Robustness},
  author={Rebuffi, Sylvestre-Alvise and Gowal, Sven and Calian, Dan A. and Stimberg, Florian and Wiles, Olivia and Mann, Timothy},
  journal={arXiv preprint arXiv:2103.01946},
  year={2021},
  url={https://arxiv.org/pdf/2103.01946}
}
```

## Disclaimer

This is not an official Google product.
