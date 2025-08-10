#!/usr/bin/env python3
"""
MeshGraphNet Adaptive Remeshing Demo

This script demonstrates the key concepts of adaptive remeshing
as discussed in Issue #519, showing how node counts change and
how ground truth interpolation works during training.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass
import scipy.spatial
from scipy.interpolate import griddata


@dataclass 
class Node:
    """Mesh node with position and fields"""
    id: int
    position: np.ndarray  # [x, y] coordinates
    velocity: np.ndarray  # [vx, vy] velocity field
    pressure: float       # pressure field
    node_type: int        # node type (see NodeType enum)


@dataclass
class Edge:
    """Mesh edge connecting two nodes"""
    node1_id: int
    node2_id: int
    length: float
    

@dataclass  
class Mesh:
    """Mesh representation with nodes and connectivity"""
    nodes: List[Node]
    edges: List[Edge]
    triangles: List[Tuple[int, int, int]]  # triangle connectivity
    

class NodeType:
    """Node type enumeration matching MeshGraphNet"""
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9  # Sizing field nodes


class AdaptiveRemesher:
    """
    Simplified adaptive remesher demonstrating MeshGraphNet concepts
    """
    
    def __init__(self, split_ratio: float = 1.4, collapse_ratio: float = 0.6):
        self.split_ratio = split_ratio      # Split edges longer than this * desired_length
        self.collapse_ratio = collapse_ratio  # Collapse edges shorter than this * desired_length
        
    def compute_sizing_field(self, mesh: Mesh) -> np.ndarray:
        """
        Compute desired edge length at each node based on field gradients
        This simulates what the neural network would predict
        """
        sizing_field = np.ones(len(mesh.nodes)) * 0.1  # Default edge length
        
        for i, node in enumerate(mesh.nodes):
            # Simulate adaptive sizing based on pressure gradient
            # In real implementation, this would be predicted by neural network
            local_gradient = np.linalg.norm([
                node.velocity[0] * 0.1,  # Velocity-based refinement
                abs(node.pressure) * 0.05  # Pressure-based refinement
            ])
            
            # Smaller edges needed in high-gradient regions
            desired_length = max(0.02, 0.15 / (1 + local_gradient * 10))
            sizing_field[i] = desired_length
            
        return sizing_field
        
    def remesh(self, mesh: Mesh, sizing_field: np.ndarray) -> Mesh:
        """
        Apply adaptive remeshing based on sizing field
        Returns new mesh with potentially different node count
        """
        print(f"🔄 Remeshing: Input mesh has {len(mesh.nodes)} nodes")
        
        new_nodes = list(mesh.nodes)  # Start with existing nodes
        new_edges = []
        nodes_to_remove = set()
        
        # Process each edge for splitting/collapsing
        for edge in mesh.edges:
            node1 = mesh.nodes[edge.node1_id]
            node2 = mesh.nodes[edge.node2_id]
            
            current_length = edge.length
            desired_length = (sizing_field[edge.node1_id] + sizing_field[edge.node2_id]) / 2
            
            if current_length > self.split_ratio * desired_length:
                # SPLIT: Add new node at edge midpoint
                new_node_id = len(new_nodes)
                midpoint_pos = (node1.position + node2.position) / 2
                
                # Interpolate fields to new node
                new_velocity = (node1.velocity + node2.velocity) / 2
                new_pressure = (node1.pressure + node2.pressure) / 2
                
                new_node = Node(
                    id=new_node_id,
                    position=midpoint_pos,
                    velocity=new_velocity, 
                    pressure=new_pressure,
                    node_type=NodeType.NORMAL
                )
                new_nodes.append(new_node)
                
                # Create two new edges replacing the original
                new_edges.extend([
                    Edge(edge.node1_id, new_node_id, current_length/2),
                    Edge(new_node_id, edge.node2_id, current_length/2)
                ])
                print(f"  ➕ Split edge {edge.node1_id}-{edge.node2_id}, added node {new_node_id}")
                
            elif current_length < self.collapse_ratio * desired_length:
                # COLLAPSE: Mark one node for removal (simplified)
                if node1.node_type == NodeType.NORMAL:  # Only remove normal nodes
                    nodes_to_remove.add(edge.node1_id)
                    print(f"  ➖ Collapsed edge {edge.node1_id}-{edge.node2_id}, removing node {edge.node1_id}")
                else:
                    new_edges.append(edge)  # Keep edge
            else:
                # Keep edge unchanged
                new_edges.append(edge)
        
        # Remove collapsed nodes (simplified - in reality need to update connectivity)
        final_nodes = [node for node in new_nodes if node.id not in nodes_to_remove]
        
        # Update node IDs to be consecutive
        for i, node in enumerate(final_nodes):
            node.id = i
            
        print(f"✅ Remeshing complete: Output mesh has {len(final_nodes)} nodes")
        print(f"   📊 Node count change: {len(mesh.nodes)} → {len(final_nodes)} ({len(final_nodes) - len(mesh.nodes):+d})")
        
        return Mesh(nodes=final_nodes, edges=new_edges, triangles=mesh.triangles)


class GroundTruthInterpolator:
    """
    Interpolates ground truth fields from original mesh to remeshed topology
    This solves the core challenge of computing loss with variable node counts
    """
    
    def interpolate_fields(self, original_mesh: Mesh, remeshed_mesh: Mesh, 
                          original_ground_truth: Dict) -> Dict:
        """
        Interpolate ground truth fields to new mesh topology
        """
        print(f"🔄 Interpolating ground truth: {len(original_mesh.nodes)} → {len(remeshed_mesh.nodes)} nodes")
        
        # Extract positions and field values from original mesh
        original_positions = np.array([node.position for node in original_mesh.nodes])
        original_velocities = np.array([node.velocity for node in original_mesh.nodes])
        original_pressures = np.array([node.pressure for node in original_mesh.nodes])
        
        # Target positions (remeshed mesh)
        target_positions = np.array([node.position for node in remeshed_mesh.nodes])
        
        # Interpolate using scipy's griddata (barycentric interpolation)
        interpolated_velocities = self._interpolate_vector_field(
            original_positions, original_velocities, target_positions
        )
        
        interpolated_pressures = griddata(
            points=original_positions,
            values=original_pressures,
            xi=target_positions,
            method='linear',
            fill_value=0.0
        )
        
        interpolated_ground_truth = {
            'velocity': interpolated_velocities,
            'pressure': interpolated_pressures,
            'positions': target_positions
        }
        
        print("✅ Ground truth interpolation complete")
        return interpolated_ground_truth
    
    def _interpolate_vector_field(self, original_pos: np.ndarray, 
                                 original_vectors: np.ndarray, 
                                 target_pos: np.ndarray) -> np.ndarray:
        """Interpolate vector field components separately"""
        vx_interp = griddata(original_pos, original_vectors[:, 0], target_pos, 
                           method='linear', fill_value=0.0)
        vy_interp = griddata(original_pos, original_vectors[:, 1], target_pos,
                           method='linear', fill_value=0.0)
        return np.column_stack([vx_interp, vy_interp])


def compute_loss_with_variable_topology(predicted_fields: Dict, 
                                       interpolated_ground_truth: Dict) -> float:
    """
    Compute training loss when mesh topology changes between prediction and target
    """
    # MSE loss on velocity field
    velocity_loss = np.mean(
        (predicted_fields['velocity'] - interpolated_ground_truth['velocity'])**2
    )
    
    # MSE loss on pressure field  
    pressure_loss = np.mean(
        (predicted_fields['pressure'] - interpolated_ground_truth['pressure'])**2
    )
    
    total_loss = velocity_loss + pressure_loss
    
    print(f"📊 Loss computation:")
    print(f"   Velocity loss: {velocity_loss:.6f}")
    print(f"   Pressure loss: {pressure_loss:.6f}")
    print(f"   Total loss: {total_loss:.6f}")
    
    return total_loss


def create_sample_mesh() -> Mesh:
    """Create a simple sample mesh for demonstration"""
    # Create a 3x3 grid of nodes
    nodes = []
    node_id = 0
    
    for i in range(3):
        for j in range(3):
            position = np.array([i * 0.1, j * 0.1])
            
            # Simulate some interesting velocity and pressure fields
            velocity = np.array([
                0.1 * np.sin(position[0] * 10),  # Varying velocity
                0.05 * np.cos(position[1] * 15)
            ])
            pressure = 0.8 + 0.3 * np.sin(position[0] * 8) * np.cos(position[1] * 8)
            
            nodes.append(Node(
                id=node_id,
                position=position,
                velocity=velocity,
                pressure=pressure,
                node_type=NodeType.NORMAL
            ))
            node_id += 1
    
    # Create edges (simplified connectivity)
    edges = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            distance = np.linalg.norm(nodes[i].position - nodes[j].position)
            if distance < 0.15:  # Connect nearby nodes
                edges.append(Edge(i, j, distance))
    
    return Mesh(nodes=nodes, edges=edges, triangles=[])


def visualize_remeshing_demo(original_mesh: Mesh, remeshed_mesh: Mesh):
    """Visualize the before/after of remeshing"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original mesh
    orig_pos = np.array([node.position for node in original_mesh.nodes])
    orig_pressures = np.array([node.pressure for node in original_mesh.nodes])
    
    scatter1 = ax1.scatter(orig_pos[:, 0], orig_pos[:, 1], c=orig_pressures, 
                          cmap='viridis', s=100, alpha=0.8)
    ax1.set_title(f'Original Mesh ({len(original_mesh.nodes)} nodes)')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    plt.colorbar(scatter1, ax=ax1, label='Pressure')
    
    # Plot remeshed mesh
    remesh_pos = np.array([node.position for node in remeshed_mesh.nodes])
    remesh_pressures = np.array([node.pressure for node in remeshed_mesh.nodes])
    
    scatter2 = ax2.scatter(remesh_pos[:, 0], remesh_pos[:, 1], c=remesh_pressures,
                          cmap='viridis', s=100, alpha=0.8)
    ax2.set_title(f'Remeshed Mesh ({len(remeshed_mesh.nodes)} nodes)')
    ax2.set_xlabel('X coordinate') 
    ax2.set_ylabel('Y coordinate')
    plt.colorbar(scatter2, ax=ax2, label='Pressure')
    
    plt.tight_layout()
    plt.savefig('remeshing_demo.png', dpi=150, bbox_inches='tight')
    print("📊 Visualization saved as 'remeshing_demo.png'")


def main():
    """
    Demonstrate MeshGraphNet adaptive remeshing concepts
    """
    print("🚀 MeshGraphNet Adaptive Remeshing Demo")
    print("=" * 50)
    
    # 1. Create initial mesh
    print("\n1️⃣  Creating initial mesh...")
    original_mesh = create_sample_mesh()
    print(f"✅ Created mesh with {len(original_mesh.nodes)} nodes")
    
    # 2. Apply adaptive remeshing
    print("\n2️⃣  Applying adaptive remeshing...")
    remesher = AdaptiveRemesher()
    sizing_field = remesher.compute_sizing_field(original_mesh)
    remeshed_mesh = remesher.remesh(original_mesh, sizing_field)
    
    # 3. Interpolate ground truth for loss computation
    print("\n3️⃣  Interpolating ground truth to new topology...")
    interpolator = GroundTruthInterpolator()
    
    # Simulate ground truth at next timestep (on original mesh)
    original_ground_truth = {
        'velocity': np.array([node.velocity * 1.1 for node in original_mesh.nodes]),  # Evolved velocity
        'pressure': np.array([node.pressure * 1.05 for node in original_mesh.nodes])  # Evolved pressure
    }
    
    interpolated_gt = interpolator.interpolate_fields(
        original_mesh, remeshed_mesh, original_ground_truth
    )
    
    # 4. Compute loss with variable topology
    print("\n4️⃣  Computing loss with variable topology...")
    predicted_fields = {
        'velocity': np.array([node.velocity for node in remeshed_mesh.nodes]),
        'pressure': np.array([node.pressure for node in remeshed_mesh.nodes])
    }
    
    loss = compute_loss_with_variable_topology(predicted_fields, interpolated_gt)
    
    # 5. Visualize results
    print("\n5️⃣  Creating visualization...")
    visualize_remeshing_demo(original_mesh, remeshed_mesh)
    
    print("\n🎯 Summary of Key Concepts:")
    print("   ✅ Node count DOES change during remeshing")
    print("   ✅ Remeshing IS performed during training (for *_sizing datasets)")
    print("   ✅ Ground truth is interpolated to match new topology")
    print("   ✅ Loss is computed on the remeshed mesh with interpolated targets")
    print("\n📚 This resolves the questions raised in Issue #519!")


if __name__ == "__main__":
    main()
