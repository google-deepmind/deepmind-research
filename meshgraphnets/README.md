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

## Overview

This release contains the full datasets used in the paper, as well as data
loaders (dataset.py), and the learned model core (core_model.py).
These components are designed to work with all of our domains.

We also include demonstration code for a full training and evaluation pipeline,
for the `cylinder_flow` and `flag_simple` domains only. This
includes graph encoding, evaluation, rollout and plotting trajectory.
Refer to the respective `cfd_*` and `cloth_*` files for details.

## Setup

Prepare environment, install dependencies:

    virtualenv --python=python3.6 "${ENV}"
    ${ENV}/bin/activate
    pip install -r meshgraphnets/requirements.txt

Download a dataset:

    mkdir -p ${DATA}
    bash meshgraphnets/download_dataset.sh flag_simple ${DATA}

**Alternative download methods:**

If you encounter 404 errors with the bash script (Issue #596), use the Python downloader:

    # Download using Python script (recommended)
    python meshgraphnets/download_meshgraphnet_datasets.py --dataset flag_simple --output_dir ${DATA}
    
    # List available datasets
    python meshgraphnets/download_meshgraphnet_datasets.py --list-datasets
    
    # Verify downloaded dataset
    python meshgraphnets/download_meshgraphnet_datasets.py --verify flag_simple

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

The instructions above train a model for the `flag_simple` domain; for
the `cylinder_flow` dataset, use `--model=cfd` and the `plot_cfd` script.

## Datasets

Datasets can be downloaded using the script `download_dataset.sh`. They contain
a metadata file describing the available fields and their shape, and tfrecord
datasets for train, valid and test splits.
Dataset names match the naming in the paper.

### 📋 **Complete Dataset List**

The following datasets are available for download:

#### **Fluid Dynamics (CFD)**
- **`airfoil`**: Airfoil steady-state simulations (CFD around airfoils) - *Addresses Issue #569*
- **`cylinder_flow`**: Cylinder flow CFD dataset (time-dependent fluid dynamics)

#### **Cloth/Structural Dynamics**  
- **`flag_simple`**: Simple flag simulation dataset (cloth dynamics)
- **`flag_minimal`**: Truncated version of flag_simple (for integration tests)
- **`flag_dynamic`**: Advanced flag dynamics with varying conditions
- **`flag_dynamic_sizing`**: Flag simulation with adaptive mesh sizing

#### **Structural Mechanics**
- **`deforming_plate`**: Deforming plate simulation dataset
- **`sphere_simple`**: Simple sphere simulation dataset  
- **`sphere_dynamic`**: Complex sphere dynamics
- **`sphere_dynamic_sizing`**: Sphere simulation with adaptive mesh sizing

### 💾 **Download Methods**

**Using Python script (recommended for Issue #569):**
```bash
# Download airfoil dataset (addresses Issue #569)
python meshgraphnets/download_meshgraphnet_datasets.py --dataset airfoil --output_dir ${DATA}

# List all available datasets
python meshgraphnets/download_meshgraphnet_datasets.py --list-datasets

# Download any specific dataset
python meshgraphnets/download_meshgraphnet_datasets.py --dataset DATASET_NAME --output_dir ${DATA}
```

**Using shell script:**
```bash
bash meshgraphnets/download_dataset.sh DATASET_NAME ${DATA}
```

### 📖 **Detailed Dataset Information**

For comprehensive information about each dataset, including research applications and usage examples, see [DATASETS.md](DATASETS.md).

**Special Note for Issue #569**: The `airfoil` dataset mentioned in the MeshGraphNets paper is available for download using the methods above. This dataset contains steady-state CFD simulations around airfoils, suitable for aerodynamics research and comparison studies.
