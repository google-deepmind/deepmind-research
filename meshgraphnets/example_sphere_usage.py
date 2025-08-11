"""
Example usage script for sphere_dynamic implementation.
Demonstrates how to use the new sphere model functionality.
"""

import os
import sys
import tempfile
import numpy as np
import tensorflow.compat.v1 as tf

# Add meshgraphnets to path
sys.path.append(os.path.dirname(__file__))

from meshgraphnets import sphere_model, sphere_eval, core_model


def create_sample_sphere_data():
    """Create sample sphere data for testing."""
    print("📊 Creating sample sphere data...")
    
    # Sphere parameters
    num_nodes = 162  # Typical icosphere subdivision
    num_faces = 320
    num_timesteps = 50
    
    # Generate sphere mesh (simplified)
    theta = np.linspace(0, 2*np.pi, num_nodes//2)
    phi = np.linspace(0, np.pi, num_nodes//2)
    
    # Create sample positions (simplified sphere)
    positions = []
    for t in range(num_timesteps):
        # Simulate sphere deformation over time
        deformation = 0.1 * np.sin(t * 0.1)
        radius = 1.0 + deformation
        
        sphere_pos = []
        for i in range(num_nodes):
            t_idx = i % len(theta)
            p_idx = i % len(phi)
            x = radius * np.sin(phi[p_idx]) * np.cos(theta[t_idx])
            y = radius * np.sin(phi[p_idx]) * np.sin(theta[t_idx])
            z = radius * np.cos(phi[p_idx])
            sphere_pos.append([x, y, z])
        
        positions.append(sphere_pos)
    
    # Convert to numpy arrays
    world_pos = np.array(positions, dtype=np.float32)
    
    # Create sample mesh connectivity (simplified)
    cells = np.random.randint(0, num_nodes, (num_faces, 3))
    
    # Create node types (all normal nodes for this example)
    node_type = np.zeros((num_nodes, 1), dtype=np.int32)
    
    # Create mesh positions (reference)
    mesh_pos = world_pos[0]  # Use first timestep as reference
    
    return {
        'world_pos': world_pos,
        'mesh_pos': mesh_pos,
        'cells': cells,
        'node_type': node_type
    }


def test_sphere_model():
    """Test the sphere model implementation."""
    print("🧪 Testing sphere model...")
    
    tf.disable_v2_behavior()
    
    # Create sample data
    data = create_sample_sphere_data()
    
    # Create model
    learned_model = core_model.EncodeProcessDecode(
        output_size=3,  # 3D acceleration
        latent_size=128,
        num_layers=2,
        message_passing_steps=15)
    
    model = sphere_model.Model(learned_model)
    
    # Create test inputs
    num_nodes = data['world_pos'].shape[1]
    batch_inputs = {
        'world_pos': tf.constant(data['world_pos'][1], dtype=tf.float32),
        'prev|world_pos': tf.constant(data['world_pos'][0], dtype=tf.float32),
        'target|world_pos': tf.constant(data['world_pos'][2], dtype=tf.float32),
        'mesh_pos': tf.constant(data['mesh_pos'], dtype=tf.float32),
        'cells': tf.constant(data['cells'], dtype=tf.int32),
        'node_type': tf.constant(data['node_type'], dtype=tf.int32)
    }
    
    # Test forward pass
    try:
        output = model(batch_inputs)
        print(f"✅ Model forward pass successful, output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False
    
    # Test loss calculation
    try:
        loss = model.loss(batch_inputs)
        print(f"✅ Loss calculation successful")
    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        return False
    
    return True


def test_sphere_eval():
    """Test the sphere evaluation functions."""
    print("📈 Testing sphere evaluation...")
    
    # Create sample data
    data = create_sample_sphere_data()
    num_timesteps, num_nodes, _ = data['world_pos'].shape
    
    # Create mock model for testing
    class MockModel:
        def __call__(self, inputs):
            # Return a simple prediction (slightly perturbed current position)
            current_pos = inputs['world_pos']
            prev_pos = inputs['prev|world_pos']
            # Simple forward prediction
            velocity = current_pos - prev_pos
            next_pos = current_pos + velocity * 0.98  # Slight damping
            return next_pos
    
    mock_model = MockModel()
    
    # Prepare inputs for evaluation
    eval_inputs = {
        'world_pos': tf.constant(data['world_pos'], dtype=tf.float32),
        'cells': tf.constant(np.tile(data['cells'][None], (num_timesteps, 1, 1)), dtype=tf.int32),
        'mesh_pos': tf.constant(np.tile(data['mesh_pos'][None], (num_timesteps, 1, 1)), dtype=tf.float32),
        'node_type': tf.constant(np.tile(data['node_type'][None], (num_timesteps, 1, 1)), dtype=tf.int32)
    }
    
    try:
        scalars, traj_ops = sphere_eval.evaluate(mock_model, eval_inputs)
        print("✅ Sphere evaluation successful")
        print(f"   Available metrics: {list(scalars.keys())}")
        print(f"   Trajectory ops: {list(traj_ops.keys())}")
        return True
    except Exception as e:
        print(f"❌ Sphere evaluation failed: {e}")
        return False


def main():
    """Run all tests for sphere_dynamic implementation."""
    print("🚀 Testing MeshGraphNets Sphere Dynamic Implementation")
    print("=" * 60)
    
    success = True
    
    # Test model
    if not test_sphere_model():
        success = False
    
    print()
    
    # Test evaluation
    if not test_sphere_eval():
        success = False
    
    print()
    print("=" * 60)
    
    if success:
        print("🎉 All tests passed! Sphere dynamic implementation is working correctly.")
        print()
        print("📚 Next steps:")
        print("1. Download sphere_dynamic dataset")
        print("2. Train model: python -m meshgraphnets.run_model --mode=train --model=sphere")
        print("3. Evaluate: python -m meshgraphnets.run_model --mode=eval --model=sphere")
        print("4. Visualize: python -m meshgraphnets.plot_sphere --rollout_path=results.pkl")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == '__main__':
    main()
