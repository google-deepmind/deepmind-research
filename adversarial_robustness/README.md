# Adversarial Robustness

This repository contains the code needed to evaluate models trained in
[Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples](https://arxiv.org/abs/2010.03593)


## Contents

We have released our top-performing models in two formats compatible with
[JAX](https://github.com/google/jax) and [PyTorch](https://pytorch.org/).
This repository also contains our model definitions.

## Running the example code

### Downloading a model

Download a model from links listed in the following table.
Clean and robust accuracies are measured on the full test set.
The robust accuracy is measured using
[AutoAttack](https://github.com/fra31/auto-attack).

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

### Using the model

Once downloaded, a model can be evaluated (clean accuracy) by running the
`eval.py` script in either the `jax` or `pytorch` folders. E.g.:

```
cd jax
python3 eval.py \
  --ckpt=${PATH_TO_CHECKPOINT} --depth=70 --width=16 --dataset=cifar10
```


## Citing this work

If you use this code or these models in your work, please cite the accompanying
paper:

```
@article{gowal2020uncovering,
  title={Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples},
  author={Gowal, Sven and Qin, Chongli and Uesato, Jonathan and Mann, Timothy and Kohli, Pushmeet},
  journal={arXiv preprint arXiv:2010.03593},
  year={2020},
  url={https://arxiv.org/pdf/2010.03593}
}
```

## Disclaimer

This is not an official Google product.
