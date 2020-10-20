# Geometric Manifold Component Estimator (GEOMANCER)

This package provides an implementation of the Geometric Manifold Component
Estimator, or GEOMANCER, as described in [Disentangling by Subspace Diffusion
(2020)](https://arxiv.org/abs/2006.12982). GEOMANCER is a nonparametric
algorithm for disentangling, somewhat similar in spirit to Laplacian Eigenmaps
or Vector Diffusion Maps, except instead of producing an embedding for the data,
it produces a set of subspaces around each data point, one subspace for each
disentangled factor of variation in the data. This differs from more common
algorithms for disentangling that originated in the deep learning community,
such as the beta-VAE, TCVAE or FactorVAE, which learn a nonlinear embedding and
probabilistic generative model of the data. GEOMANCER is intended for data where
the individual factors of variation might be more than one dimensional, for
instance 3D rotations. At the moment, GEOMANCER works best when some ground
truth information about the metric in the data space is available, for instance
knowledge of the "true" nearest neighbors around each point, and we do not
recommend running GEOMANCER directly on unstructured data from high-dimensional
spaces. We are providing the code here to enable the interested researcher to
get some hands-on experience with the ideas around differential geometry,
holonomy and higher-order graph connection Laplacians we explore in the paper.


## Installation

To install the package locally in a new virtual environment run:
```bash
python3 -m venv geomancer
source geomancer/bin/activate
git clone https://github.com/deepmind/deepmind-research.git .
cd deepmind-research/geomancer
pip install -e .
```

## Example

To run, simply load or generate an array of data, and call the `fit` function:

```
import numpy as np
import geomancer

# Generate data from a product of two spheres
data = []
for i in range(2):
  foo = np.random.randn(1000, 3)
  data.append(foo / np.linalg.norm(foo, axis=1, keepdims=True))
data = np.concatenate(data, axis=1)

# Run GEOMANCER. The underlying manifold is 4-dimensional.
components, spectrum = geomancer.fit(data, 4)
```

If ground truth information about the tangent spaces is available in a space
that is aligned with the data, then the performance can be evaluated using the
`eval_aligned` function. If ground truth data is only available in an unaligned
space, for instance if the embedding used to generate the data is not the same
as the space in which the data is observed, then the `eval_unaligned` function
can be used, which requires both the data and disentangled tangent vectors in
the ground truth space. Examples of both evaluation metrics are given in the
demo in `train.py`.


## Demo on Synthetic Manifolds

The file `train.py` runs GEOMANCER on a product of manifolds that can be
specified by the user. The number of data points to train on is given by the
`--npts` flag, while the specification of the manifold is given by the
`--specification` flag. The `--rotate` flag specifies whether a random rotation
should be applied to the data. If false, `eval_aligned` will be used to evaluate
the result. If true, `eval_unaligned` will be used to evaluate the result.

For instance, to run on the product of the sphere in 2 and 4 dimensions and the
special orthogonal group in 3 dimensions, run:

```
python3 train.py --specification='S^2','S^4','SO(3)' --npts=100000
```

This passes a list of strings as the manifold specification flag. Note that a
manifold this large will require a large amount of data to work and may require
hours or days to run. The default example should run in just a few minutes.

The demo plots 3 different outputs:
1. The eigenvalue spectrum of the 2nd-order graph Laplacian. This should have
a large gap in the spectrum at the eigenvalue equal to the number of
submanifolds.
2. The basis vectors for each disentangled subspace around one point.
3. The ground truth basis vectors for the disentangled subspaces at the same
point. If `--rotate=False`, and GEOMANCER has sufficient data, each basis matrix
should span the same subspace as the results in the second plot.

## Giving Credit

If you use this code in your work, we ask you to cite this paper:

```
@article{pfau2020disentangling,
  title={Disentangling by Subspace Diffusion},
  author={Pfau, David and Higgins, Irina and Botev, Aleksandar and Racani\`ere,
  S{\'e}bastian},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```

## Disclaimer

This is not an official Google product.
