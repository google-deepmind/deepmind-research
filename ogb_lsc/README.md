# DeepMind entry for OGB-LSC

This repository contains DeepMind's entry to the [PCQM4M-LSC](https://ogb.stanford.edu/kddcup2021/pcqm4m/) (quantum chemistry) and
[MAG240M-LSC](https://ogb.stanford.edu/kddcup2021/mag240m/) (academic graph)
tracks of the [OGB Large-Scale Challenge](https://ogb.stanford.edu/kddcup2021/)
(OGB-LSC).

For full details regarding this entry, please see our [technical report](https://arxiv.org/abs/2107.09422).

## Code structure

* `pcq/`: Scripts for training, evaluating on the [PCQ dataset](https://ogb.stanford.edu/docs/graphprop/).
* `mag/`: Scripts for training, evaluating on the [MAG dataset](https://ogb.stanford.edu/docs/nodeprop/).

## Citation

To cite this work:

```latex
@article{deepmind2021ogb,
  author = {Ravichandra Addanki and Peter Battaglia and David Budden and Andreea
    Deac and Jonathan Godwin and Thomas Keck and Wai Lok Sibon Li and Alvaro
    Sanchez-Gonzalez and Jacklynn Stott and Shantanu Thakoor and Petar
    Veli\v{c}kovi\'{c}},
  title = {Large-scale graph representation learning with very deep GNNs and
    self-supervision},
  year = {2021},
  journal={arXiv preprint arXiv:2107.09422},
}
```
