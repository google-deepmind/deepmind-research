# Learning Mesh-Based Simulation with Graph Networks (ICLR 2021)

Video site: [sites.google.com/view/meshgraphnets](https://sites.google.com/view/meshgraphnets)

Paper: [arxiv.org/abs/2010.03409](https://arxiv.org/abs/2010.03409)

If you use the code here please cite this paper:

    @inproceedings{pfaff2021learning,
      title={Learning Mesh-Based Simulation with Graph Networks},
      author={Tobias Pfaff and
              Meire Fortunato and
              Alvaro Sanchez-Gonzalez and
              Peter W. Battaglia},
      booktitle={International Conference on Learning Representations},
      year={2021}
    }

## Setup

Prepare environment, install dependencies:

    virtualenv --python=python3.6 "${ENV}"
    ${ENV}/bin/activate
    pip install -r meshgraphnets/requirements.txt

Download a dataset:

    mkdir -p ${DATA}
    bash meshgraphnets/download_dataset.sh flag_simple ${DATA}

## Running the model

Train a model:

    python -m meshgraphnets.run_model --mode=train --model=cloth \
        --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/flag_simple

Generate some trajectory rollouts:

    python -m meshgraphnets.run_model --mode=eval --model=cloth \
        --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/flag_simple \
        --rollout_path=${DATA}/rollout_flag.pkl

Plot a trajectory:

    python -m meshgraphnets.plot_cloth --rollout_path=${DATA}/rollout_flag.pkl

## Datasets

Datasets can be downloaded using the script `download_dataset.sh`. They contain
a metadata file describing the available fields and their shape, and tfrecord
datasets for train, valid and test splits.
Dataset names match the naming in the paper.
The following datasets are available:

    airfoil
    cylinder_flow
    deforming_plate
    flag_minimal
    flag_simple
    flag_dynamic
    sphere_simple
    sphere_dynamic

`flag_minimal` is a truncated version of flag_simple, and is only used for
integration tests.
