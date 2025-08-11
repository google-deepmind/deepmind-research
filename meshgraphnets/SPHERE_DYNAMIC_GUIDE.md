# MeshGraphNets Sphere Dynamic Implementation

## Issue #529 Solution

This directory provides the missing implementation for **sphere_dynamic** examples from the MeshGraphNets paper, addressing [Issue #529](https://github.com/google-deepmind/deepmind-research/issues/529).

## 🆕 **New Files Added**

### Core Implementation
- **`sphere_model.py`** - Model architecture for sphere dynamics simulation
- **`sphere_eval.py`** - Evaluation metrics and rollout functions for spheres
- **`plot_sphere.py`** - Visualization tools for sphere trajectories

### Updated Files
- **`run_model.py`** - Added support for `--model=sphere` parameter

## 🚀 **Quick Start**

### 1. Download the sphere_dynamic dataset
```bash
mkdir -p ${DATA}
bash meshgraphnets/download_dataset.sh sphere_dynamic ${DATA}
```

### 2. Train a sphere dynamics model
```bash
python -m meshgraphnets.run_model --mode=train --model=sphere \
    --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/sphere_dynamic
```

### 3. Generate trajectory rollouts
```bash
python -m meshgraphnets.run_model --mode=eval --model=sphere \
    --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/sphere_dynamic \
    --rollout_path=${DATA}/rollout_sphere.pkl
```

### 4. Visualize results
```bash
# Interactive 3D visualization
python -m meshgraphnets.plot_sphere --rollout_path=${DATA}/rollout_sphere.pkl

# Save as GIF animation
python -m meshgraphnets.plot_sphere --rollout_path=${DATA}/rollout_sphere.pkl \
    --save_gif --output_path=sphere_animation.gif
```

## 📊 **Model Architecture**

### Sphere Model Features
- **3D Dynamics**: Full 3D position and velocity tracking
- **Verlet Integration**: Stable numerical integration for sphere dynamics
- **Boundary Handling**: Proper treatment of kinematic vs. normal nodes
- **Volume Preservation**: Specialized metrics for sphere volume conservation

### Key Differences from Cloth Model
- **3D Focus**: Optimized for spherical geometry and 3D deformations
- **Volume Metrics**: Additional evaluation metrics for volume preservation
- **Center of Mass**: Tracking of sphere center of mass for stability analysis

## 🔧 **Technical Details**

### Input Features
- **Node Features**: 3D velocity + node type (3 + NodeType.SIZE dimensions)
- **Edge Features**: Relative positions in world/mesh space + distances (7 dimensions)

### Output
- **Acceleration**: 3D acceleration vectors for next timestep prediction

### Loss Function
- **MSE Loss**: Mean squared error on positions (normal nodes only)
- **Volume Preservation**: Implicit through network architecture
- **Boundary Conditions**: Kinematic nodes remain fixed

## 📈 **Evaluation Metrics**

The sphere evaluation provides comprehensive metrics:

### Standard Metrics
- **MSE at horizons**: 1, 5, 10, 20, 50, 100, 200 steps
- **Trajectory comparison**: Ground truth vs. prediction

### Sphere-Specific Metrics
- **Center of Mass Error**: Tracking of sphere center stability
- **Volume Preservation Error**: Variance in radial distances
- **Surface Deformation**: Node-wise displacement analysis

## 🎨 **Visualization Features**

### Interactive 3D Plots
- **Side-by-side comparison**: Ground truth vs. prediction
- **Real-time MSE display**: Error metrics during animation
- **Customizable views**: 3D rotation and zoom

### Export Options
- **GIF Animation**: Save animations for presentations
- **Frame Export**: Individual frame extraction
- **Data Export**: Trajectory data for external analysis

## 📋 **Dataset Requirements**

The sphere_dynamic dataset should contain:
- **world_pos**: 3D node positions over time
- **mesh_pos**: Reference mesh positions
- **cells**: Triangle connectivity
- **node_type**: Node classification (normal/kinematic)
- **prev|world_pos**: Previous timestep positions
- **target|world_pos**: Next timestep positions (ground truth)

## 🔄 **Integration with Existing Code**

This implementation seamlessly integrates with the existing MeshGraphNets framework:

```python
# Use exactly like cloth or cfd models
python -m meshgraphnets.run_model --model=sphere [other_args]
```

### Parameter Configuration
```python
# Sphere parameters in run_model.py
'sphere': dict(
    noise=0.003,        # Training noise level
    gamma=0.1,          # Noise decay
    field='world_pos',  # Primary field
    history=True,       # Use velocity history
    size=3,             # 3D coordinates
    batch=1,            # Batch size
    model=sphere_model,
    evaluator=sphere_eval
)
```

## 🐛 **Troubleshooting**

### Common Issues

1. **Dataset not found**
   ```bash
   # Ensure dataset is downloaded
   bash meshgraphnets/download_dataset.sh sphere_dynamic ${DATA}
   ```

2. **Visualization errors**
   ```bash
   # Install required packages
   pip install matplotlib numpy
   ```

3. **Memory issues**
   ```bash
   # Reduce batch size in parameters
   # Use smaller rollout sequences
   ```

## 🔗 **Related Work**

- **Paper**: [Learning Mesh-Based Simulation with Graph Networks (ICLR 2021)](https://arxiv.org/abs/2010.03409)
- **Project Page**: [sites.google.com/view/meshgraphnets](https://sites.google.com/view/meshgraphnets)
- **Original Issue**: [#529](https://github.com/google-deepmind/deepmind-research/issues/529)

## 🙏 **Acknowledgments**

This implementation was created to address the community need for sphere_dynamic examples, as raised in Issue #529 by @jjdunlop and discussed by @ambreshbiradar9.

## 📄 **License**

Same as the original MeshGraphNets code - Apache License 2.0.
