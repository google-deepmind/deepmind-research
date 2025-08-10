#!/usr/bin/env python3
"""
TFRecord Dataset Generator for Learning to Simulate

This script addresses Issue #204 by providing a comprehensive solution
for generating train.tfrecord files from custom simulation data.

Usage:
    python generate_tfrecord_dataset.py --input_dir=/path/to/sim_data --output_dir=/path/to/output

Addresses the questions from Issue #204:
1. How to generate train.tfrecord?
2. How to read TFRecord contents?
3. How to create custom datasets for cloth simulation?
4. How to calculate velocity/acceleration statistics?
"""

import argparse
import functools
import json
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import tensorflow as tf
from pathlib import Path


class TFRecordDatasetGenerator:
    """
    Generates TFRecord datasets compatible with Learning to Simulate framework
    """
    
    def __init__(self, 
                 sequence_length: int = 1000,
                 dimensions: int = 3,
                 default_connectivity_radius: float = 0.015,
                 bounds: List[Tuple[float, float]] = None):
        """
        Initialize the TFRecord generator
        
        Args:
            sequence_length: Number of timesteps per trajectory
            dimensions: Spatial dimensions (2D or 3D)
            default_connectivity_radius: Graph connectivity radius
            bounds: Simulation bounds [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
        """
        self.sequence_length = sequence_length
        self.dimensions = dimensions
        self.default_connectivity_radius = default_connectivity_radius
        self.bounds = bounds or [(-1.0, 1.0)] * dimensions
        
        # Statistics for normalization
        self.velocity_stats = {"mean": None, "std": None}
        self.acceleration_stats = {"mean": None, "std": None}
        self.context_stats = {"mean": None, "std": None}
        
    def compute_statistics(self, trajectories: List[Dict]) -> Dict:
        """
        Compute velocity and acceleration statistics for normalization
        
        This addresses the question from Issue #204 about how vel_mean, vel_std, 
        acc_mean, and acc_std are calculated in metadata.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Dictionary containing computed statistics
        """
        print("🔢 Computing velocity and acceleration statistics...")
        
        all_velocities = []
        all_accelerations = []
        all_contexts = []
        
        for traj_idx, trajectory in enumerate(trajectories):
            positions = trajectory['positions']  # Shape: [time_steps, num_particles, dims]
            
            # Compute velocities as position differences (Δt = 1 as per paper)
            # vel_t = pos_{t+1} - pos_t
            velocities = positions[1:] - positions[:-1]
            all_velocities.append(velocities)
            
            # Compute accelerations as second derivatives
            # acc_t = pos_{t+1} - 2*pos_t + pos_{t-1}
            if len(positions) >= 3:
                accelerations = positions[2:] - 2*positions[1:-1] + positions[:-2]
                all_accelerations.append(accelerations)
            
            # Global context (if provided)
            if 'step_context' in trajectory:
                all_contexts.append(trajectory['step_context'])
                
            if traj_idx % 100 == 0:
                print(f"  Processed {traj_idx+1}/{len(trajectories)} trajectories")
        
        # Concatenate all data for statistics
        all_velocities = np.concatenate(all_velocities, axis=0)  # [total_steps, num_particles, dims]
        all_velocities = all_velocities.reshape(-1, self.dimensions)  # [total_particles, dims]
        
        if all_accelerations:
            all_accelerations = np.concatenate(all_accelerations, axis=0)
            all_accelerations = all_accelerations.reshape(-1, self.dimensions)
        else:
            all_accelerations = np.zeros_like(all_velocities)
        
        # Compute statistics
        vel_mean = np.mean(all_velocities, axis=0).tolist()
        vel_std = np.std(all_velocities, axis=0).tolist()
        acc_mean = np.mean(all_accelerations, axis=0).tolist()
        acc_std = np.std(all_accelerations, axis=0).tolist()
        
        self.velocity_stats = {"mean": vel_mean, "std": vel_std}
        self.acceleration_stats = {"mean": acc_mean, "std": acc_std}
        
        # Context statistics
        if all_contexts:
            all_contexts = np.concatenate(all_contexts, axis=0)
            context_mean = np.mean(all_contexts, axis=0).tolist()
            context_std = np.std(all_contexts, axis=0).tolist()
            self.context_stats = {"mean": context_mean, "std": context_std}
        
        print("✅ Statistics computation complete!")
        print(f"   Velocity mean: {vel_mean}")
        print(f"   Velocity std: {vel_std}")
        print(f"   Acceleration mean: {acc_mean}")
        print(f"   Acceleration std: {acc_std}")
        
        return {
            "vel_mean": vel_mean,
            "vel_std": vel_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "context_mean": context_mean if all_contexts else None,
            "context_std": context_std if all_contexts else None
        }
    
    def _serialize_trajectory(self, trajectory: Dict) -> bytes:
        """
        Serialize a single trajectory to TFRecord format
        
        Args:
            trajectory: Dictionary containing trajectory data
            
        Returns:
            Serialized tf.train.SequenceExample
        """
        positions = trajectory['positions']  # [time_steps, num_particles, dims]
        particle_types = trajectory.get('particle_types', np.zeros(positions.shape[1], dtype=np.int64))
        
        # Context features (constant for entire trajectory)
        context_features = {
            'key': tf.train.Feature(int64_list=tf.train.Int64List(value=[trajectory.get('key', 0)])),
            'particle_type': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[particle_types.tobytes()])
            )
        }
        
        # Sequence features (vary over time)
        sequence_features = {}
        
        # Positions
        position_list = []
        for time_step in range(positions.shape[0]):
            position_bytes = positions[time_step].astype(np.float32).tobytes()
            position_list.append(tf.train.Feature(bytes_list=tf.train.BytesList(value=[position_bytes])))
        
        sequence_features['position'] = tf.train.FeatureList(feature=position_list)
        
        # Step context (global features per timestep)
        if 'step_context' in trajectory:
            step_context = trajectory['step_context']  # [time_steps, context_dims]
            context_list = []
            for time_step in range(step_context.shape[0]):
                context_bytes = step_context[time_step].astype(np.float32).tobytes()
                context_list.append(tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_bytes])))
            
            sequence_features['step_context'] = tf.train.FeatureList(feature=context_list)
        
        # Create SequenceExample
        sequence_example = tf.train.SequenceExample(
            context=tf.train.Features(feature=context_features),
            feature_lists=tf.train.FeatureLists(feature_list=sequence_features)
        )
        
        return sequence_example.SerializeToString()
    
    def generate_tfrecords(self, 
                          trajectories: Dict[str, List[Dict]], 
                          output_dir: str,
                          dataset_name: str = "CustomDataset"):
        """
        Generate TFRecord files for train/valid/test splits
        
        Args:
            trajectories: Dict with keys 'train', 'valid', 'test' containing trajectory lists
            output_dir: Output directory for TFRecord files
            dataset_name: Name of the dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 Generating TFRecord dataset: {dataset_name}")
        print(f"   Output directory: {output_path}")
        
        # Compute statistics from training data
        if 'train' in trajectories:
            stats = self.compute_statistics(trajectories['train'])
        else:
            stats = self.compute_statistics(trajectories[list(trajectories.keys())[0]])
        
        # Generate metadata
        metadata = self._generate_metadata(trajectories, stats, dataset_name)
        
        # Write metadata.json
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Generated metadata.json")
        
        # Generate TFRecord files for each split
        for split, split_trajectories in trajectories.items():
            tfrecord_path = output_path / f'{split}.tfrecord'
            
            print(f"🔄 Generating {split}.tfrecord...")
            with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
                for traj_idx, trajectory in enumerate(split_trajectories):
                    serialized_example = self._serialize_trajectory(trajectory)
                    writer.write(serialized_example)
                    
                    if traj_idx % 100 == 0:
                        print(f"  Serialized {traj_idx+1}/{len(split_trajectories)} trajectories")
            
            print(f"✅ Generated {split}.tfrecord ({len(split_trajectories)} trajectories)")
        
        print(f"🎉 Dataset generation complete! Files saved to: {output_path}")
    
    def _generate_metadata(self, 
                          trajectories: Dict[str, List[Dict]], 
                          stats: Dict,
                          dataset_name: str) -> Dict:
        """Generate metadata.json file"""
        # Get dimensions from first trajectory
        first_traj = trajectories[list(trajectories.keys())[0]][0]
        num_particles = first_traj['positions'].shape[1]
        
        metadata = {
            "name": dataset_name,
            "dim": self.dimensions,
            "default_connectivity_radius": self.default_connectivity_radius,
            "sequence_length": self.sequence_length,
            "bounds": self.bounds,
            "vel_mean": stats["vel_mean"],
            "vel_std": stats["vel_std"],
            "acc_mean": stats["acc_mean"],
            "acc_std": stats["acc_std"]
        }
        
        # Add context statistics if available
        if stats["context_mean"] is not None:
            metadata["context_mean"] = stats["context_mean"]
            metadata["context_std"] = stats["context_std"]
        
        # Add split information
        for split, split_trajectories in trajectories.items():
            metadata[f"num_trajectories_{split}"] = len(split_trajectories)
        
        return metadata


class ClothSimulationLoader:
    """
    Example loader for cloth simulation data
    Addresses the specific cloth dataset creation question from Issue #204
    """
    
    @staticmethod
    def load_from_simulation_output(simulation_dir: str) -> Dict[str, List[Dict]]:
        """
        Load cloth simulation data from directory structure
        
        Expected structure:
        simulation_dir/
        ├── train/
        │   ├── trajectory_001.npz
        │   ├── trajectory_002.npz
        │   └── ...
        ├── valid/
        │   └── ...
        └── test/
            └── ...
        
        Each .npz file should contain:
        - positions: [time_steps, num_particles, 3] array
        - particle_types: [num_particles] array (optional)
        - step_context: [time_steps, context_dims] array (optional)
        """
        trajectories = {}
        
        for split in ['train', 'valid', 'test']:
            split_dir = Path(simulation_dir) / split
            if not split_dir.exists():
                print(f"⚠️ Warning: {split} directory not found, skipping...")
                continue
            
            split_trajectories = []
            npz_files = list(split_dir.glob('*.npz'))
            
            print(f"📂 Loading {split} data from {len(npz_files)} files...")
            
            for npz_file in npz_files:
                try:
                    data = np.load(npz_file)
                    
                    trajectory = {
                        'positions': data['positions'],
                        'key': int(npz_file.stem.split('_')[-1]) if '_' in npz_file.stem else 0
                    }
                    
                    # Optional fields
                    if 'particle_types' in data:
                        trajectory['particle_types'] = data['particle_types']
                    if 'step_context' in data:
                        trajectory['step_context'] = data['step_context']
                    
                    split_trajectories.append(trajectory)
                    
                except Exception as e:
                    print(f"❌ Error loading {npz_file}: {e}")
            
            trajectories[split] = split_trajectories
            print(f"✅ Loaded {len(split_trajectories)} {split} trajectories")
        
        return trajectories
    
    @staticmethod
    def create_sample_cloth_dataset(output_dir: str, num_trajectories: int = 10):
        """
        Create a sample cloth dataset for demonstration
        """
        print(f"🧵 Creating sample cloth dataset with {num_trajectories} trajectories...")
        
        trajectories = {'train': [], 'valid': [], 'test': []}
        
        # Generate synthetic cloth simulation data
        for split, num_traj in [('train', num_trajectories), ('valid', num_trajectories//5), ('test', num_trajectories//5)]:
            for traj_id in range(num_traj):
                # Simulate cloth as 10x10 grid of particles
                grid_size = 10
                num_particles = grid_size * grid_size
                time_steps = 100
                
                # Initialize cloth grid
                x = np.linspace(-0.5, 0.5, grid_size)
                y = np.linspace(-0.5, 0.5, grid_size)
                xx, yy = np.meshgrid(x, y)
                
                initial_positions = np.stack([
                    xx.flatten(),
                    yy.flatten(),
                    np.zeros(num_particles)  # Start flat
                ], axis=1)
                
                # Simulate cloth falling with some dynamics
                positions = np.zeros([time_steps, num_particles, 3])
                positions[0] = initial_positions
                
                for t in range(1, time_steps):
                    # Simple physics: gravity + some cloth-like deformation
                    gravity = np.array([0, 0, -0.001])
                    noise = np.random.normal(0, 0.0001, [num_particles, 3])
                    
                    # Keep top edge fixed (cloth hanging)
                    positions[t] = positions[t-1] + gravity + noise
                    positions[t][:grid_size] = initial_positions[:grid_size]  # Fixed top edge
                
                # Particle types (0=normal, 3=handle for fixed particles)
                particle_types = np.zeros(num_particles, dtype=np.int64)
                particle_types[:grid_size] = 3  # Top edge as handles
                
                trajectory = {
                    'positions': positions,
                    'particle_types': particle_types,
                    'key': traj_id
                }
                
                trajectories[split].append(trajectory)
        
        print("✅ Sample cloth dataset created!")
        return trajectories


def read_tfrecord_contents(tfrecord_path: str, metadata_path: str, max_examples: int = 5):
    """
    Read and display TFRecord contents in human-readable format
    
    Addresses the garbled code issue mentioned in Issue #204
    """
    print(f"📖 Reading TFRecord contents from: {tfrecord_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Import reading utilities (assuming they're available)
    try:
        from learning_to_simulate import reading_utils
        
        # Create dataset
        dataset = tf.data.TFRecordDataset([tfrecord_path])
        dataset = dataset.map(functools.partial(
            reading_utils.parse_serialized_simulation_example, metadata=metadata))
        
        print(f"🔍 Displaying first {max_examples} examples:")
        
        for i, (context, features) in enumerate(dataset.take(max_examples)):
            print(f"\n--- Example {i+1} ---")
            print(f"Context keys: {list(context.keys())}")
            print(f"Feature keys: {list(features.keys())}")
            
            if 'position' in features:
                pos_shape = features['position'].shape
                print(f"Position shape: {pos_shape}")
                print(f"Position sample (first particle, first 3 timesteps):")
                print(features['position'][:3, 0, :].numpy())
            
            if 'particle_type' in context:
                print(f"Particle types: {context['particle_type'].numpy()[:10]}...")
            
            if 'step_context' in features:
                print(f"Step context shape: {features['step_context'].shape}")
        
    except ImportError:
        print("❌ Could not import reading_utils. Using raw TFRecord parsing...")
        
        # Raw parsing for demonstration
        for i, serialized_example in enumerate(tf.data.TFRecordDataset([tfrecord_path]).take(max_examples)):
            print(f"\n--- Raw Example {i+1} ---")
            print(f"Serialized size: {len(serialized_example.numpy())} bytes")
            
            # Parse as SequenceExample
            sequence_example = tf.train.SequenceExample()
            sequence_example.ParseFromString(serialized_example.numpy())
            
            print("Context features:")
            for key, feature in sequence_example.context.feature.items():
                print(f"  {key}: {len(feature.bytes_list.value)} bytes")
            
            print("Sequence features:")
            for key, feature_list in sequence_example.feature_lists.feature_list.items():
                print(f"  {key}: {len(feature_list.feature)} timesteps")


def main():
    parser = argparse.ArgumentParser(description="Generate TFRecord datasets for Learning to Simulate")
    parser.add_argument('--input_dir', type=str, help='Input directory containing simulation data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for TFRecord files')
    parser.add_argument('--dataset_name', type=str, default='CustomDataset', help='Name of the dataset')
    parser.add_argument('--create_sample', action='store_true', help='Create sample cloth dataset')
    parser.add_argument('--read_tfrecord', type=str, help='Path to TFRecord file to read')
    parser.add_argument('--metadata_path', type=str, help='Path to metadata.json file')
    parser.add_argument('--sequence_length', type=int, default=1000, help='Sequence length')
    parser.add_argument('--dimensions', type=int, default=3, help='Spatial dimensions')
    
    args = parser.parse_args()
    
    if args.read_tfrecord:
        # Read existing TFRecord
        if not args.metadata_path:
            args.metadata_path = str(Path(args.read_tfrecord).parent / 'metadata.json')
        read_tfrecord_contents(args.read_tfrecord, args.metadata_path)
        return
    
    # Initialize generator
    generator = TFRecordDatasetGenerator(
        sequence_length=args.sequence_length,
        dimensions=args.dimensions
    )
    
    if args.create_sample:
        # Create sample cloth dataset
        trajectories = ClothSimulationLoader.create_sample_cloth_dataset(args.output_dir)
    elif args.input_dir:
        # Load from simulation output
        trajectories = ClothSimulationLoader.load_from_simulation_output(args.input_dir)
    else:
        print("❌ Error: Must specify either --input_dir or --create_sample")
        return
    
    # Generate TFRecord dataset
    generator.generate_tfrecords(trajectories, args.output_dir, args.dataset_name)
    
    print("\n🎯 Usage Examples:")
    print(f"# Train a model:")
    print(f"python -m learning_to_simulate.train --data_path={args.output_dir} --model_path=/tmp/models/{args.dataset_name}")
    print(f"\n# Read generated TFRecord:")
    print(f"python generate_tfrecord_dataset.py --read_tfrecord={args.output_dir}/train.tfrecord")


if __name__ == "__main__":
    main()
