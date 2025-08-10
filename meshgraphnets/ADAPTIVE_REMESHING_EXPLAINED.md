# MeshGraphNet Adaptive Remeshing: Technical Explanation

**Issue #519 Resolution**: This document provides comprehensive answers to the technical questions about MeshGraphNet's adaptive remeshing mechanics, addressing confusion about node count changes, training procedures, and loss computation with variable topologies.

## Overview

MeshGraphNet implements adaptive remeshing during both training and inference to optimize mesh resolution based on local field gradients and simulation requirements. This document explains the technical mechanics behind adaptive remeshing as described in Section 3.2 of the MeshGraphNet paper.

## 🔍 Core Questions Answered

### 1. **Does remeshing change the number of nodes?**

**✅ YES** - Remeshing operations fundamentally change the mesh topology and node count:

#### Remeshing Operations:
- **Edge Splitting**: Creates new nodes at edge midpoints when local error exceeds threshold
- **Edge Collapse**: Removes nodes by merging adjacent vertices when resolution is too high  
- **Node Insertion**: Adds nodes in regions requiring higher resolution
- **Node Removal**: Eliminates nodes in over-resolved regions

#### Node Count Dynamics:
```python
# Simplified remeshing logic
def adaptive_remesh(mesh, sizing_field):
    """
    Adaptive remeshing based on sizing field R
    sizing_field: Per-node desired edge length
    """
    new_nodes = []
    new_edges = []
    
    for edge in mesh.edges:
        current_length = edge.length
        desired_length = sizing_field[edge.nodes].mean()
        
        if current_length > 1.4 * desired_length:
            # SPLIT: Create new node at edge midpoint
            new_node = create_midpoint_node(edge)
            new_nodes.append(new_node)
            
        elif current_length < 0.6 * desired_length:
            # COLLAPSE: Remove one endpoint
            remove_node(edge.endpoint_with_lower_priority())
    
    return updated_mesh_with_variable_node_count
```

### 2. **Is remeshing performed during training?**

**✅ YES** - Remeshing is performed during training for datasets with `*_sizing` suffix:

#### Training Datasets with Remeshing:
- `flag_dynamic_sizing` 
- `sphere_dynamic_sizing`
- Any dataset containing sizing field annotations

#### Training Procedure:
```python
def training_step_with_remeshing(model, mesh_t, target_t_plus_1):
    """
    Training step with adaptive remeshing
    """
    # 1. Predict next state AND sizing field
    predicted_state, predicted_sizing = model(mesh_t)
    
    # 2. Apply remesher R to get new mesh topology  
    remeshed_mesh = remesher_R(mesh_t, predicted_sizing)
    
    # 3. Interpolate ground truth to new mesh topology
    interpolated_target = interpolate_ground_truth(
        original_target=target_t_plus_1,
        original_mesh=mesh_t, 
        new_mesh=remeshed_mesh
    )
    
    # 4. Compute loss on remeshed topology
    loss = compute_loss(
        prediction=predicted_state,
        target=interpolated_target,
        mesh=remeshed_mesh
    )
    
    return loss
```

## 🧮 Loss Computation with Variable Topology

### Challenge: Ground Truth Interpolation

The most complex aspect is computing loss when node counts change between prediction and target:

#### Problem:
- **Original target**: Defined on mesh with N nodes
- **Predicted state**: Defined on remeshed mesh with M nodes (M ≠ N)
- **Solution**: Interpolate ground truth to new mesh topology

### Implementation Strategy:

#### 1. **Spatial Interpolation**
```python
def interpolate_ground_truth(original_target, original_mesh, new_mesh):
    """
    Interpolate ground truth fields to new mesh topology
    using spatial interpolation methods
    """
    interpolated_fields = {}
    
    for field_name, field_values in original_target.items():
        if field_name in ['velocity', 'pressure', 'displacement']:
            # Use barycentric interpolation for continuous fields
            interpolated_fields[field_name] = barycentric_interpolate(
                source_mesh=original_mesh,
                target_mesh=new_mesh, 
                source_values=field_values
            )
        elif field_name == 'node_type':
            # Use nearest neighbor for discrete fields
            interpolated_fields[field_name] = nearest_neighbor_interpolate(
                source_mesh=original_mesh,
                target_mesh=new_mesh,
                source_values=field_values
            )
    
    return interpolated_fields

def barycentric_interpolate(source_mesh, target_mesh, source_values):
    """
    Barycentric interpolation for continuous fields
    """
    interpolated_values = []
    
    for new_node in target_mesh.nodes:
        # Find containing triangle in original mesh
        containing_triangle = find_containing_triangle(
            point=new_node.position,
            mesh=source_mesh
        )
        
        if containing_triangle is not None:
            # Compute barycentric coordinates
            barycentric_coords = compute_barycentric_coordinates(
                point=new_node.position,
                triangle=containing_triangle
            )
            
            # Interpolate value using barycentric weights
            interpolated_value = sum(
                coord * source_values[vertex_id] 
                for coord, vertex_id in zip(barycentric_coords, containing_triangle.vertices)
            )
        else:
            # Fallback to nearest neighbor for boundary cases
            interpolated_value = nearest_neighbor_value(new_node, source_mesh, source_values)
        
        interpolated_values.append(interpolated_value)
    
    return interpolated_values
```

#### 2. **Conservative Interpolation for Physical Quantities**
```python
def conservative_interpolate(source_mesh, target_mesh, source_values):
    """
    Conservative interpolation preserving physical quantities
    (e.g., mass, momentum, energy)
    """
    # Ensure conservation of integral quantities
    source_integral = integrate_over_mesh(source_mesh, source_values)
    
    # Standard interpolation
    interpolated_values = barycentric_interpolate(source_mesh, target_mesh, source_values)
    
    # Apply conservation constraint
    target_integral = integrate_over_mesh(target_mesh, interpolated_values)
    conservation_factor = source_integral / target_integral
    
    return interpolated_values * conservation_factor
```

### 3. **Loss Function with Topology Changes**

```python
def compute_variable_topology_loss(prediction, interpolated_target, mesh):
    """
    Compute loss handling variable mesh topology
    """
    # Standard MSE loss on interpolated ground truth
    field_loss = tf.reduce_mean(
        tf.square(prediction.fields - interpolated_target.fields)
    )
    
    # Regularization based on mesh quality
    mesh_quality_loss = compute_mesh_quality_loss(mesh)
    
    # Sizing field consistency loss
    sizing_consistency_loss = compute_sizing_consistency_loss(
        predicted_sizing=prediction.sizing_field,
        actual_edge_lengths=compute_edge_lengths(mesh)
    )
    
    total_loss = (
        field_loss + 
        0.01 * mesh_quality_loss + 
        0.1 * sizing_consistency_loss
    )
    
    return total_loss
```

## 🔬 Sizing Field Prediction

### Node Type: SIZE

In `common.py`, we see:
```python
class NodeType(enum.IntEnum):
  NORMAL = 0
  OBSTACLE = 1
  AIRFOIL = 2
  HANDLE = 3
  INFLOW = 4
  OUTFLOW = 5
  WALL_BOUNDARY = 6
  SIZE = 9  # ← Sizing field indicator
```

The `SIZE` node type indicates nodes that predict local mesh sizing requirements.

### Sizing Field Architecture
```python
def create_sizing_aware_model():
    """
    Model architecture that predicts both physics and sizing
    """
    # Standard physics prediction
    physics_decoder = create_physics_decoder(output_dim=physics_dims)
    
    # Sizing field prediction (one value per node)
    sizing_decoder = create_sizing_decoder(output_dim=1)
    
    def forward(graph):
        # Shared graph processing
        processed_graph = process_graph(graph)
        
        # Dual outputs
        physics_output = physics_decoder(processed_graph)
        sizing_output = sizing_decoder(processed_graph)
        
        return {
            'physics': physics_output,
            'sizing_field': sizing_output,
            'node_types': graph.node_features[:, -1]  # SIZE nodes
        }
    
    return forward
```

## 🎯 Training vs Inference Differences

### Training Mode
- **Remeshing**: Applied when sizing field datasets are used
- **Ground Truth**: Interpolated to match remeshed topology
- **Loss**: Computed on variable topology with interpolated targets
- **Objective**: Learn to predict both physics and optimal mesh sizing

### Inference Mode  
- **Remeshing**: Applied at every timestep using predicted sizing field
- **No Ground Truth**: Model operates autonomously
- **Adaptation**: Mesh evolves based on simulation needs
- **Efficiency**: Computational resources focused on important regions

## 📊 Practical Implementation Notes

### 1. **Interpolation Quality**
- Higher-order interpolation schemes improve accuracy
- Conservative interpolation maintains physical consistency
- Boundary handling requires special attention

### 2. **Mesh Quality Control**
- Aspect ratio limits prevent degenerate elements
- Minimum/maximum edge length constraints ensure stability
- Smoothing operations maintain mesh regularity

### 3. **Computational Efficiency**
- Spatial data structures (KD-trees, octrees) accelerate interpolation
- Caching interpolation weights reduces overhead
- Adaptive remeshing frequency balances accuracy vs speed

## 🔍 Mathematical Formulation

### Loss Function with Remeshing
```
L_total = L_physics + λ₁ L_sizing + λ₂ L_quality

where:
L_physics = ||f_θ(G_t) - I(y_{t+1}, G_t → G_t')||²
L_sizing = ||R_predicted - R_optimal||²  
L_quality = Σ quality_metrics(G_t')

I(·) = interpolation operator
G_t → G_t' = mesh topology change due to remeshing
```

### Interpolation Operator
```
I(y, G_original → G_remeshed) = Σᵢ wᵢ(x) yᵢ

where wᵢ(x) are interpolation weights for position x
```

## 🚀 Benefits of Adaptive Remeshing

1. **Resolution Independence**: Model adapts mesh density to simulation needs
2. **Computational Efficiency**: Focus computation on important regions
3. **Accuracy Preservation**: Maintain precision in critical areas
4. **Scalability**: Handle complex geometries with variable resolution requirements

## 📚 Related Code Files

- `common.py`: Node types including SIZE
- `core_model.py`: Graph network architecture
- `dataset.py`: Data loading and field handling
- `*_sizing` datasets: Training data with remeshing annotations

## 🔗 References

- MeshGraphNet Paper: [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
- Section 3.2: "ADAPTIVE REMESHING"
- Implementation: `meshgraphnets/` directory

---

**This explanation resolves Issue #519 by providing comprehensive technical details about MeshGraphNet's adaptive remeshing mechanics, including node count changes, training procedures, and loss computation strategies with variable mesh topologies.**
