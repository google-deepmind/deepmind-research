# Targeted free energy estimation via learned mappings

This repository contains supporting data for our publication
([journal](https://doi.org/10.1063/5.0018903), [arXiv](https://arxiv.org/abs/2002.04913)).
Here, we provide
- molecular dynamics (MD) datasets underlying the results reported in our paper,
- a LAMMPS input script to generate these datasets, and
- the data plotted in Fig. 5 of our paper to facilitate comparison.

## Abstract

Free energy perturbation (FEP) was proposed by Zwanzig more than six decades ago
as a method to estimate free energy differences, and has since inspired a huge
body of related methods that use it as an integral building block. Being an
importance sampling based estimator, however, FEP suffers from a severe
limitation: the requirement of sufficient overlap between distributions.
One strategy to mitigate this problem, called Targeted Free Energy Perturbation,
uses a high-dimensional mapping in configuration space to increase overlap of
the underlying distributions. Despite its potential, this method has attracted
only limited attention due to the formidable challenge of formulating a
tractable mapping. Here, we cast Targeted FEP as a machine learning problem in
which the mapping is parameterized as a neural network that is optimized so as
to increase overlap. We develop a new model architecture that respects
permutational and periodic symmetries often encountered in atomistic simulations
and test our method on a fully-periodic solvation system. We demonstrate that
our method leads to a substantial variance reduction in free energy estimates
when compared against baselines, without requiring any additional data.

## Dataset

We generated the datasets using the open-source MD package
[LAMMPS](https://lammps.sandia.gov). The prototypical solvation problem of study
consists of a solute particle immersed in a liquid comprising 125 solvent
particles. The solvent-solvent interactions are modelled using a Lennard-Jones
potential and the solute-solvent interactions via a Weeks-Chandler-Andersen
(WCA) potential. Further simulation details can be found in the LAMMPS script
provided (see below) and in our [paper](https://arxiv.org/abs/2002.04913)
(see Sec. 4 and Appendix B).

### Download

You can download the compressed datasets (~3.8GB) using the command:
> wget https://storage.googleapis.com/learned_free_energy_estimation/learned_free_energy_estimation_datasets.tar.bz2
or by copying the above link directly into your browser.

Once the archive `learned_free_energy_estimation_datasets.tar.bz2` is
downloaded, you can extract it with the command:
> tar -xvf learned_free_energy_estimation_datasets.tar.bz2

### Data format

The archive contains a total of 40 files:
- 10 train datasets for ensemble *A* (`ensemble_a_train_<<index>>.dat`),
- 10 train datasets for ensemble *B* (`ensemble_b_train_<<index>>.dat`),
- 10 test datasets for ensemble *A* (`ensemble_a_test_<<index>>.dat`) and
- 10 test datasets for ensemble *B* (`ensemble_b_test_<<index>>.dat`).

Each file is text-based and stored in a LAMMPS compatible format (see [dump command](https://lammps.sandia.gov/doc/dump.html)). Train datasets contain 90k records
each and test datasets contain 10k records, totalling 1M records for each
ensemble.

Each record contains 135 lines and is structured as follows:
- lines 1-9: Header information.
- lines 10-135: A matrix with shape `[126, 5]` containing the
  - `id` (column 1),
  - `type` (column 2) and
  - `x, y, z` coordinates (columns 3-5)

  of all particles.

For information on how the data was generated and partitioned into the final
datasets we refer to Sec. 4 and Appendix B of our [paper](https://arxiv.org/abs/2002.04913).


## LAMMPS script

The file `lammps.dat` contains a sample input script to generate data from
ensemble *A*. You can generate data from ensemble *B* by updating the value of
the solute radius, as suggested in the inline comment. For more information on
how the datasets were post-processed and partitioned, we refer to Sec. 4 and
Appendix B of our [paper](https://arxiv.org/abs/2002.04913).

## Figures

The subdirectory `figures` contains 4 files:
- `figure_5a_work_values.dat`: contains data underlying the histogram of work values in Fig. 5a.
- `figure_5b_df_bar.dat`: contains the BAR estimate of dF in Fig. 5b.
- `figure_5b_df_lbar.dat`: contains the LBAR estimate of dF in Fig. 5b.
- `figure_5b_df_mbar.dat`: contains the MBAR estimate of dF in Fig. 5b.

## Reference

If you find this repository helpful for your research, please cite our publication:

```
@article{Wirnsberger2020,
  title={Targeted free energy estimation via learned mappings},
  author={Wirnsberger, Peter and Ballard, Andrew J and Papamakarios, George and
          Abercrombie, Stuart and Racanière, Sébastien and Pritzel, Alexander and
          Jimenez Rezende, Danilo and Blundell, Charles},
  journal={J. Chem. Phys.},
  volume={153},
  number={14},
  pages={144112},
  year={2020},
  doi={10.1063/5.0018903}
}
```


## Disclaimer
This is not an official Google product.
