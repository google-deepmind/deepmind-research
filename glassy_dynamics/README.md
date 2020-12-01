# Unveiling the predictive power of static structure in glassy systems

This repository contains an open source implementation of the graph neural
network model described in our
[paper](http://dx.doi.org/10.1038/s41567-020-0842-8).
The model can be trained using the training binary included in this repository,
and the dataset published with our paper.

Pretrained model checkpoints and the dataset are available via the
[google cloud platform](https://console.cloud.google.com/storage/browser/deepmind-research-glassy-dynamics).


## Abstract

Despite decades of theoretical studies, the nature of the glass transition
remains elusive and debated, while the existence of structural predictors of its
dynamics is a major open question. Recent approaches propose inferring
predictors from a variety of human-defined features using machine learning.
Here we determine the long time evolution of a glassy system solely from the
initial particle positions and without any hand-crafted features, using graph
neural networks as a powerful model. We show that this method outperforms
current state-of-the-art methods, generalizing over a wide range of
temperatures, pressures, and densities. In shear experiments, it predicts the
locations of rearranging particles. The structural predictors learned by our
network exhibit a correlation length which increases with larger timescales to
reach the size of our system. Beyond glasses, our method could apply to many
other physical systems that map to a graph of local interaction.


## Dataset

The dataset was generated with the LAMMPS molecular dynamics package.
The simulated system has periodic boundaries and is a binary mixture of 4096
large (A) and small (B) particles that interact via a 6-12 Lennard-Jones
potential.
The interaction coefficients are set for a typical Kob-Andersen configuration.

### Download

The dataset (and model checkpoints) can be downloaded using [gsutil](https://cloud.google.com/storage/docs/downloading-objects).
To download the entire GCP bucket (~100GB) use:
> gsutil -m cp -R gs://deepmind-research-glassy-dynamics .



### Data format

The data is stored in Python's pickle format protocol version 3.
Each file contains the data for one of the equilibrated systems in a Python
dictionary. The dictionary contains the following entries:

  - `positions` the particle positions of the equilibrated system.
  - `types` the particle types (0 == type A and 1 == type B) of the equilibrated
     system.
  - `box` the dimensions of the periodic cubic simulation box.
  - `time` the logarithmically sampled time points.
  - `time_indices` the indices of the time points for which the sampled
     trajectories on average reach a certain value of the intermediate
     scattering function.
  - `is_values` the values of the intermediate scattering function associated
     with each time index.
  - `trajectory_start_velocities` the velocities drawn from a Boltzmann
     distribution at the start of each trajectory.
  - `trajectory_target_positions` the positions of the particles for each of
     the trajectories at selected time points (as defined by the `time_indices`
     array and the corresponding values of the intermediate scattering function
     stored in `is_values`).
  - `metadata` a dictionary containing additional metadata:
    - `temperature` the temperature at which the system was equilibrated.
    - `pressure` the pressure at which the system was equilibrated.
    - `fluid` the type of fluid which was simulated (Kob-Andersen).

All units are in Lennard-Jones units. The positions are stored in the absolute
coordinate system i.e. they are outside of the simulation box if the particle
crossed a periodic boundary during the simulation.


## Reference

If this repository is helpful for your research please cite the following
publication:

[Unveiling the predictive power of static structure in glassy systems](http://dx.doi.org/10.1038/s41567-020-0842-8)
V. Bapst, T. Keck, A. Grabska-Barwińska, C. Donner, E. D. Cubuk,
S. S. Schoenholz, A. Obika, A. W. R. Nelson, T. Back, D. Hassabis and P. Kohli


## Disclaimer
This is not an official Google product.

