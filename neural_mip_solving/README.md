# Neural MIP Solving - NN Verification Dataset

This is the “Neural Network Verification” dataset used in the paper

[Solving Mixed Integer Programs Using Neural Networks (Nair et al., 2020)](https://arxiv.org/abs/2012.13349).

It contains a set of mixed integer programs (MIPs) for the problem of verifying
a neural network’s robustness to perturbations to its inputs. The MIP
formulation is described in the paper
[On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models (Gowal et al., 2018)](https://arxiv.org/abs/1810.12715).

This dataset corresponds to MIPs defined for
verifying a neural network with the architecture labelled as “small” in Table 1
of Gowal et al., 2018, and trained on the MNIST image dataset. The code used to
train the neural network to be verified is available at
https://github.com/deepmind/interval-bound-propagation. The MIPs are split into
the same training, validation, and test sets as that in Nair et al., 2020.


## Dataset Location

The dataset is available in the following
[link](https://storage.cloud.google.com/neural-mip-solving/nn_verification.tar.gz)

## Dataset Metadata

The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">Neural Network Verification Dataset</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/deepmind/deepmind-research/tree/master/neural_mip_solving</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://github.com/deepmind/deepmind-research/tree/master/neural_mip_solving</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">
      This dataset contains a set of mixed integer programs (MIPs) for the
      problem of verifying a neural network’s robustness to perturbations of its
      inputs. The MIPs are encoded in LP format.</code></td>
  </tr>
  <tr>
    <td>license</td>
    <td><code itemprop="license">https://creativecommons.org/licenses/by/4.0/legalcode
  </code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">DeepMind</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/DeepMind</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>citation</td>
    <td><code itemprop="citation">https://arxiv.org/abs/2012.13349</code></td>
  </tr>
</table>
</div>


## Citing this work

If you use this dataset in your work, we ask you to cite this paper:

```
@misc{nair2020solving,
      title={Solving Mixed Integer Programs Using Neural Networks},
      author={Vinod Nair and Sergey Bartunov and Felix Gimeno and Ingrid von Glehn and Pawel Lichocki and Ivan Lobov and Brendan O'Donoghue and Nicolas Sonnerat and Christian Tjandraatmadja and Pengming Wang and Ravichandra Addanki and Tharindi Hapuarachchi and Thomas Keck and James Keeling and Pushmeet Kohli and Ira Ktena and Yujia Li and Oriol Vinyals and Yori Zwols},
      year={2020},
      eprint={2012.13349},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```

## License

This dataset is made available under the terms of the Creative Commons
Attribution 4.0 International (CC BY 4.0) license.

You can find details at: https://creativecommons.org/licenses/by/4.0/legalcode

## Disclaimer

This is not an officially supported Google product.
