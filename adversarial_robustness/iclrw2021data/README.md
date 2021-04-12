# Adversarial Robustness

This repository contains the code needed to evaluate models trained in
[Data Augmentation Can Improve Robustness](https://storage.googleapis.com/dm-adversarial-robustness/rebuffi2021data.pdf)
which has been accepted at
[ICLR 2021 Security and Safety in Machine Learning Systems Workshop](https://aisecure-workshop.github.io/aml-iclr2021/).


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
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-70-16 | &#x2713; | 92.23% | 66.58% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn70-16_cutmix_external.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn70-16_cutmix_external.pt)
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-70-16 | &#x2717; | 87.25% | 60.07% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn70-16_cutmix.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn70-16_cutmix.pt)
| CIFAR-10 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-28-10 | &#x2717; | 86.09% | 57.61% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn28-10_cutmix.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_linf_wrn28-10_cutmix.pt)
| CIFAR-100 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-70-16 | &#x2717; | 65.76% | 32.43% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn70-16_cutmix.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn70-16_cutmix.pt)
| CIFAR-100 | &#8467;<sub>&infin;</sub> | 8 / 255 | WRN-28-10 | &#x2717; | 62.97% | 29.80% | [jax](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn28-10_cutmix.npy), [pt](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn28-10_cutmix.pt)

### Using the model

Once downloaded, a model can be evaluated (clean accuracy) by running the
`eval.py` script in either the `jax` or `pytorch` folders. E.g.:

```
cd jax
python3 eval.py \
  --ckpt=${PATH_TO_CHECKPOINT} --depth=70 --width=16 --dataset=cifar10
```


## Citing this work

If you use this code or these models in your work, please cite the complete
version which combines data augmentation with generated samples:

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
