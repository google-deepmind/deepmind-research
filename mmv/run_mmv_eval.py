#!/usr/bin/env python3
"""
MMV (Self-Supervised MultiModal Versatile Networks) Runner

This script provides a robust way to run MMV models with proper device 
configuration and axis name handling for both single-device and multi-device setups.

Usage:
    python run_mmv_eval.py --checkpoint_path=/path/to/checkpoint --dataset_folder=/path/to/data
"""

import os
import sys
import warnings
from typing import Optional

def setup_jax_environment(force_single_device: bool = False) -> None:
    """Configure JAX environment for MMV."""
    
    # Set JAX configuration before importing JAX
    if force_single_device:
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Import JAX after environment setup
    import jax
    
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")
    
    # Configure for single device if needed
    if len(jax.devices()) == 1 or force_single_device:
        print("Running in single-device mode")
        # Ensure we don't use pmap-style operations
        return
    else:
        print("Running in multi-device mode")


def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    
    required_packages = {
        'jax': None,
        'jaxlib': None, 
        'haiku': 'dm-haiku',
        'tensorflow': None,
        'tensorflow_datasets': None,
        'sklearn': 'scikit-learn',
        'numpy': None
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            package_name = pip_name or package
            missing_packages.append(package_name)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def run_mmv_eval(checkpoint_path: str, 
                 dataset_folder: str,
                 force_single_device: bool = False,
                 **kwargs) -> None:
    """Run MMV evaluation with proper configuration."""
    
    # Setup environment
    setup_jax_environment(force_single_device)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies. Please install required packages.")
        return
    
    # Import MMV evaluation after environment setup
    try:
        from mmv import eval_ucf101
        import absl.app
        import absl.flags
        
        # Set flags
        absl.flags.FLAGS.checkpoint_path = checkpoint_path
        absl.flags.FLAGS.dataset_folder = dataset_folder
        
        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(absl.flags.FLAGS, key):
                setattr(absl.flags.FLAGS, key, value)
        
        # Run evaluation
        print("🚀 Starting MMV evaluation...")
        absl.app.run(eval_ucf101.main)
        
    except Exception as e:
        print(f"❌ Error running MMV evaluation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that checkpoint path exists and is valid")
        print("2. Ensure dataset folder is properly prepared")
        print("3. Try running with --force_single_device if you see axis name errors")
        print("4. Check JAX and Haiku versions are compatible")


def main():
    """Main command line interface."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run MMV evaluation with proper device configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on single device (recommended for debugging)
  python run_mmv_eval.py --checkpoint_path=./mmv_s3d.pkl --dataset_folder=./ucf101 --force_single_device
  
  # Run on available devices
  python run_mmv_eval.py --checkpoint_path=./mmv_s3d.pkl --dataset_folder=./ucf101
  
  # Check environment only
  python run_mmv_eval.py --check_only
        """
    )
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the MMV checkpoint file')
    parser.add_argument('--dataset_folder', type=str, required=True,
                       help='Path to the UCF101 dataset folder')
    parser.add_argument('--force_single_device', action='store_true',
                       help='Force single device mode to avoid axis name errors')
    parser.add_argument('--check_only', action='store_true',
                       help='Only check dependencies and environment')
    
    # Additional MMV parameters
    parser.add_argument('--eval_batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--num_test_windows', type=int, default=10,
                       help='Number of test windows to average')
    
    args = parser.parse_args()
    
    if args.check_only:
        print("🔍 Checking MMV environment...")
        setup_jax_environment(args.force_single_device)
        check_dependencies()
        return
    
    # Run MMV evaluation
    run_mmv_eval(
        checkpoint_path=args.checkpoint_path,
        dataset_folder=args.dataset_folder,
        force_single_device=args.force_single_device,
        eval_batch_size=args.eval_batch_size,
        num_test_windows=args.num_test_windows
    )


if __name__ == '__main__':
    main()
