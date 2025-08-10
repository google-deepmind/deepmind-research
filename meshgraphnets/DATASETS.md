# MeshGraphNets Dataset Guide

This document provides comprehensive information about MeshGraphNets datasets, specifically addressing Issue #569 regarding the AirFoil Steady State dataset access.

## Available Datasets

Based on the MeshGraphNets paper and repository, the following datasets are available:

### 🌊 **Fluid Dynamics (CFD) Datasets**

#### 1. **airfoil** - Airfoil Steady State Dataset
- **Description**: Computational Fluid Dynamics simulations around airfoils
- **Type**: Steady-state flow simulations
- **Use Case**: Research on aerodynamics, airfoil performance analysis
- **Paper Reference**: Mentioned as comparison dataset in MeshGraphNets paper
- **Domain**: Fluid dynamics with complex boundary conditions

#### 2. **cylinder_flow** - Cylinder Flow Dataset  
- **Description**: CFD simulations of fluid flow around cylindrical obstacles
- **Type**: Time-dependent flow simulations
- **Use Case**: Fluid dynamics research, wake formation studies
- **Recommended Model**: `--model=cfd`

### 🧵 **Cloth/Structural Dynamics Datasets**

#### 3. **flag_simple** - Simple Flag Simulation
- **Description**: Cloth dynamics simulation of flag motion
- **Type**: Structural dynamics with simple boundary conditions
- **Use Case**: Cloth simulation, deformable object modeling
- **Recommended Model**: `--model=cloth`

#### 4. **flag_minimal** - Minimal Flag Dataset
- **Description**: Truncated version of flag_simple
- **Type**: Testing/integration dataset (smaller size)
- **Use Case**: Quick testing, integration tests

#### 5. **flag_dynamic** - Dynamic Flag Simulation
- **Description**: More complex flag dynamics with varying conditions
- **Type**: Advanced cloth dynamics
- **Use Case**: Research on complex cloth behavior

#### 6. **flag_dynamic_sizing** - Dynamic Flag with Sizing Field
- **Description**: Flag simulation with adaptive mesh sizing
- **Type**: Adaptive mesh refinement research
- **Use Case**: Learning optimal mesh sizing strategies

### 🏗️ **Structural Mechanics Datasets**

#### 7. **deforming_plate** - Deforming Plate Simulation
- **Description**: Structural deformation simulation of plates
- **Type**: Solid mechanics simulation
- **Use Case**: Structural analysis, material deformation research

#### 8. **sphere_simple** - Simple Sphere Simulation  
- **Description**: Basic sphere deformation/dynamics
- **Type**: Simple structural dynamics
- **Use Case**: Basic deformable object research

#### 9. **sphere_dynamic** - Dynamic Sphere Simulation
- **Description**: Complex sphere dynamics with varying conditions
- **Type**: Advanced structural dynamics
- **Use Case**: Complex deformable object modeling

#### 10. **sphere_dynamic_sizing** - Dynamic Sphere with Sizing Field
- **Description**: Sphere simulation with adaptive mesh sizing
- **Type**: Adaptive mesh refinement for spherical objects
- **Use Case**: Learning mesh sizing for complex geometries

## Download Instructions

### Method 1: Using Python Download Script (Recommended)

```bash
# List all available datasets
python meshgraphnets/download_meshgraphnet_datasets.py --list-datasets

# Download airfoil dataset
python meshgraphnets/download_meshgraphnet_datasets.py --dataset airfoil --output_dir ./data

# Download cylinder_flow dataset  
python meshgraphnets/download_meshgraphnet_datasets.py --dataset cylinder_flow --output_dir ./data

# Verify downloaded dataset
python meshgraphnets/download_meshgraphnet_datasets.py --verify ./data
```

### Method 2: Using Shell Script

```bash
# Download airfoil dataset
bash meshgraphnets/download_dataset.sh airfoil ./data

# Download cylinder_flow dataset
bash meshgraphnets/download_dataset.sh cylinder_flow ./data
```

## Dataset Structure

Each dataset contains:
- `meta.json`: Metadata describing fields and shapes
- `train.tfrecord`: Training data
- `valid.tfrecord`: Validation data  
- `test.tfrecord`: Test data

## Research Usage

### For AirFoil Steady State Research (Issue #569)

The **airfoil** dataset is specifically designed for:
- Computational Fluid Dynamics research
- Airfoil performance analysis
- Steady-state flow simulation studies
- Comparison with other CFD methods

### Training Models

```bash
# Train CFD model on airfoil dataset
python -m meshgraphnets.run_model --mode=train --model=cfd \
    --checkpoint_dir=./checkpoints --dataset_dir=./data/airfoil

# Train CFD model on cylinder_flow dataset
python -m meshgraphnets.run_model --mode=train --model=cfd \
    --checkpoint_dir=./checkpoints --dataset_dir=./data/cylinder_flow

# Train cloth model on flag datasets
python -m meshgraphnets.run_model --mode=train --model=cloth \
    --checkpoint_dir=./checkpoints --dataset_dir=./data/flag_simple
```

### Evaluation and Visualization

```bash
# Generate rollouts for airfoil simulations
python -m meshgraphnets.run_model --mode=eval --model=cfd \
    --checkpoint_dir=./checkpoints --dataset_dir=./data/airfoil \
    --rollout_path=./results/airfoil_rollout.pkl

# Plot CFD results
python -m meshgraphnets.plot_cfd --rollout_path=./results/airfoil_rollout.pkl
```

## Troubleshooting

### Common Issues

1. **Dataset not found (404 errors)**
   - Solution: Use the updated download scripts that fix broken URL issues
   - See Issue #596 fix for details

2. **Large download sizes**
   - Airfoil dataset: ~2-3GB
   - Cylinder flow: ~4-5GB  
   - Flag datasets: ~1-2GB each
   - Ensure sufficient disk space

3. **Network timeouts**
   - Use Python download script with retry logic
   - Download during off-peak hours
   - Check internet connection stability

### Paper References

- **MeshGraphNets Paper**: [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
- **Airfoil Dataset**: Used for comparison studies in computational fluid dynamics
- **Repository**: [deepmind/deepmind-research/meshgraphnets](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets)

## Contributing

If you encounter issues with dataset access:
1. Check this guide for troubleshooting steps
2. Verify your download commands match the examples
3. Report specific error messages with dataset names
4. Include system information (OS, Python version, network conditions)

---

**Note**: This guide addresses Issue #569 regarding AirFoil Steady State dataset access. The airfoil dataset is available through the standard MeshGraphNets download mechanisms once the download script fixes are applied.
