#!/usr/bin/env python3
"""
TFRecord Reader Example - Issue #204 Solution

This script demonstrates how to read TFRecord files in human-readable format,
addressing the "garbled code" issue mentioned in the GitHub discussion.

Usage:
    python tfrecord_reader_example.py --tfrecord_path=/path/to/train.tfrecord --metadata_path=/path/to/metadata.json
"""

import argparse
import functools
import json
import numpy as np
import tensorflow as tf
from pathlib import Path


def read_tfrecord_detailed(tfrecord_path: str, metadata_path: str, max_examples: int = 3):
    """
    Read TFRecord file and display contents in detail
    
    This solves the issue where users see "garbled code" when opening TFRecord files
    """
    print("=" * 60)
    print("🔍 DETAILED TFRECORD READER - Issue #204 Solution")
    print("=" * 60)
    
    print(f"📂 Reading: {tfrecord_path}")
    print(f"📋 Metadata: {metadata_path}")
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("\n✅ Metadata loaded successfully!")
        print("📊 Dataset Info:")
        for key, value in metadata.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"   {key}: {value[:3]}... (length: {len(value)})")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return
    
    # Method 1: Using reading_utils (if available)
    print("\n" + "="*50)
    print("📖 METHOD 1: Using reading_utils")
    print("="*50)
    
    try:
        from learning_to_simulate import reading_utils
        
        dataset = tf.data.TFRecordDataset([tfrecord_path])
        dataset = dataset.map(functools.partial(
            reading_utils.parse_serialized_simulation_example, 
            metadata=metadata))
        
        print(f"🔍 Examining first {max_examples} trajectories:")
        
        for i, (context, features) in enumerate(dataset.take(max_examples)):
            print(f"\n--- Trajectory {i+1} ---")
            
            # Context (static per trajectory)
            print("📋 Context Features:")
            for key, value in context.items():
                if key == 'particle_type':
                    types = value.numpy()
                    unique_types = np.unique(types)
                    print(f"   {key}: {len(types)} particles, types: {unique_types}")
                    print(f"             Sample: {types[:10]}...")
                else:
                    print(f"   {key}: {value.numpy()}")
            
            # Features (time-varying)
            print("⏱️  Sequence Features:")
            for key, value in features.items():
                shape = value.shape
                if key == 'position':
                    print(f"   {key}: shape={shape}")
                    print(f"             Time steps: {shape[0]}")
                    print(f"             Particles: {shape[1]}")
                    print(f"             Dimensions: {shape[2]}")
                    print(f"             Sample positions (first particle, first 3 steps):")
                    sample_pos = value[:3, 0, :].numpy()
                    for t, pos in enumerate(sample_pos):
                        print(f"               t={t}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                elif key == 'step_context':
                    print(f"   {key}: shape={shape}")
                    print(f"             Sample context (first 3 steps): {value[:3].numpy()}")
                else:
                    print(f"   {key}: shape={shape}")
            
            # Calculate some statistics
            if 'position' in features:
                positions = features['position'].numpy()
                velocities = positions[1:] - positions[:-1]
                
                print("📊 Computed Statistics:")
                print(f"   Position range: [{np.min(positions):.4f}, {np.max(positions):.4f}]")
                print(f"   Velocity mean: {np.mean(velocities, axis=(0,1))}")
                print(f"   Velocity std: {np.std(velocities, axis=(0,1))}")
        
    except ImportError:
        print("⚠️ reading_utils not available, using raw parsing...")
    except Exception as e:
        print(f"❌ Error with reading_utils: {e}")
    
    # Method 2: Raw TFRecord parsing
    print("\n" + "="*50)
    print("🔧 METHOD 2: Raw TFRecord Parsing")
    print("="*50)
    
    try:
        raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
        
        print(f"🔍 Raw examination of first {max_examples} records:")
        
        for i, serialized_example in enumerate(raw_dataset.take(max_examples)):
            print(f"\n--- Raw Record {i+1} ---")
            
            # Parse as SequenceExample
            sequence_example = tf.train.SequenceExample()
            sequence_example.ParseFromString(serialized_example.numpy())
            
            print("📋 Context Features:")
            for key, feature in sequence_example.context.feature.items():
                if key == 'key':
                    value = feature.int64_list.value[0] if feature.int64_list.value else 0
                    print(f"   {key}: {value}")
                elif key == 'particle_type':
                    bytes_data = feature.bytes_list.value[0] if feature.bytes_list.value else b''
                    if bytes_data:
                        particle_types = np.frombuffer(bytes_data, dtype=np.int64)
                        print(f"   {key}: {len(particle_types)} particles")
                        print(f"             Types: {np.unique(particle_types)}")
                    else:
                        print(f"   {key}: No data")
                else:
                    print(f"   {key}: {len(feature.bytes_list.value)} bytes")
            
            print("⏱️  Sequence Features:")
            for key, feature_list in sequence_example.feature_lists.feature_list.items():
                num_timesteps = len(feature_list.feature)
                print(f"   {key}: {num_timesteps} timesteps")
                
                if key == 'position' and num_timesteps > 0:
                    # Decode first timestep
                    first_step_bytes = feature_list.feature[0].bytes_list.value[0]
                    first_step_data = np.frombuffer(first_step_bytes, dtype=np.float32)
                    
                    # Infer shape from metadata
                    dims = metadata.get('dim', 3)
                    num_particles = len(first_step_data) // dims
                    
                    positions = first_step_data.reshape(num_particles, dims)
                    print(f"             First timestep: {num_particles} particles x {dims}D")
                    print(f"             Sample: {positions[0]} ...")
        
    except Exception as e:
        print(f"❌ Error with raw parsing: {e}")
    
    # Method 3: Statistics verification
    print("\n" + "="*50)
    print("📊 METHOD 3: Statistics Verification")
    print("="*50)
    
    try:
        from learning_to_simulate import reading_utils
        
        dataset = tf.data.TFRecordDataset([tfrecord_path])
        dataset = dataset.map(functools.partial(
            reading_utils.parse_serialized_simulation_example, 
            metadata=metadata))
        
        all_velocities = []
        all_accelerations = []
        
        print("🔄 Computing statistics from TFRecord data...")
        
        for context, features in dataset.take(10):  # Sample subset for speed
            positions = features['position'].numpy()
            
            # Compute velocities (position differences)
            velocities = positions[1:] - positions[:-1]
            all_velocities.append(velocities.reshape(-1, metadata['dim']))
            
            # Compute accelerations (second derivatives)
            if len(positions) >= 3:
                accelerations = positions[2:] - 2*positions[1:-1] + positions[:-2]
                all_accelerations.append(accelerations.reshape(-1, metadata['dim']))
        
        if all_velocities:
            all_velocities = np.concatenate(all_velocities, axis=0)
            computed_vel_mean = np.mean(all_velocities, axis=0)
            computed_vel_std = np.std(all_velocities, axis=0)
            
            print("📊 Statistics Comparison:")
            print(f"   Metadata vel_mean: {metadata.get('vel_mean', 'N/A')}")
            print(f"   Computed vel_mean: {computed_vel_mean.tolist()}")
            print(f"   Metadata vel_std:  {metadata.get('vel_std', 'N/A')}")
            print(f"   Computed vel_std:  {computed_vel_std.tolist()}")
            
            if all_accelerations:
                all_accelerations = np.concatenate(all_accelerations, axis=0)
                computed_acc_mean = np.mean(all_accelerations, axis=0)
                computed_acc_std = np.std(all_accelerations, axis=0)
                
                print(f"   Metadata acc_mean: {metadata.get('acc_mean', 'N/A')}")
                print(f"   Computed acc_mean: {computed_acc_mean.tolist()}")
                print(f"   Metadata acc_std:  {metadata.get('acc_std', 'N/A')}")
                print(f"   Computed acc_std:  {computed_acc_std.tolist()}")
        
    except Exception as e:
        print(f"⚠️ Could not compute statistics: {e}")
    
    print("\n" + "="*60)
    print("✅ TFRecord reading complete!")
    print("💡 Key takeaways:")
    print("   • TFRecord files are binary format (not human-readable when opened directly)")
    print("   • Use TensorFlow's parsing functions to decode the data")
    print("   • Each record contains context (static) and sequence (time-varying) features")
    print("   • Statistics are computed across all particles, timesteps, and trajectories")
    print("="*60)


def compare_with_original_datasets():
    """
    Compare data format with original Learning to Simulate datasets
    """
    print("\n🔍 Understanding Original Dataset Format")
    print("="*50)
    
    print("📊 Original datasets structure:")
    print("   WaterDrop, Sand, Cloth, etc. all follow the same format:")
    print("   ")
    print("   Context (per trajectory):")
    print("   ├── key: trajectory ID")
    print("   └── particle_type: [N] particle types")
    print("   ")
    print("   Sequence (per timestep):")
    print("   ├── position: [T, N, D] particle positions")
    print("   └── step_context: [T, C] global context (optional)")
    print("   ")
    print("   Where:")
    print("   - T = time steps (1000 typically)")
    print("   - N = number of particles (varies)")
    print("   - D = dimensions (2 or 3)")
    print("   - C = context dimensions (varies)")


def main():
    parser = argparse.ArgumentParser(description="Read TFRecord files in human-readable format")
    parser.add_argument('--tfrecord_path', type=str, required=True, help='Path to TFRecord file')
    parser.add_argument('--metadata_path', type=str, help='Path to metadata.json (auto-detected if not provided)')
    parser.add_argument('--max_examples', type=int, default=3, help='Maximum number of examples to display')
    parser.add_argument('--compare_format', action='store_true', help='Show format comparison with original datasets')
    
    args = parser.parse_args()
    
    # Auto-detect metadata path if not provided
    if not args.metadata_path:
        tfrecord_path = Path(args.tfrecord_path)
        args.metadata_path = str(tfrecord_path.parent / 'metadata.json')
        print(f"🔍 Auto-detected metadata path: {args.metadata_path}")
    
    # Check if files exist
    if not Path(args.tfrecord_path).exists():
        print(f"❌ TFRecord file not found: {args.tfrecord_path}")
        return
    
    if not Path(args.metadata_path).exists():
        print(f"❌ Metadata file not found: {args.metadata_path}")
        return
    
    # Read the TFRecord file
    read_tfrecord_detailed(args.tfrecord_path, args.metadata_path, args.max_examples)
    
    # Show format comparison if requested
    if args.compare_format:
        compare_with_original_datasets()


if __name__ == "__main__":
    main()
