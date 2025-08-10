#!/usr/bin/env python3
"""
H100 GPU Compatibility Test Script

This script tests whether the current environment is compatible with NVIDIA H100 GPUs
and provides detailed information about GPU capabilities and TensorFlow configuration.

Usage:
    python test_h100_compatibility.py
"""

import sys
import subprocess
import warnings
from typing import Dict, List, Tuple

def get_gpu_info() -> List[Dict]:
    """Get detailed GPU information using nvidia-ml-py."""
    try:
        import pynvml
        pynvml.nvmlInit()
        
        gpu_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get CUDA compute capability
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            
            gpus.append({
                'index': i,
                'name': name,
                'memory_total': memory_info.total // (1024**3),  # GB
                'memory_free': memory_info.free // (1024**3),   # GB
                'compute_capability': f"{major}.{minor}",
                'is_h100': 'H100' in name
            })
            
        return gpus
        
    except ImportError:
        print("⚠️ pynvml not installed. Install with: pip install pynvml")
        return []
    except Exception as e:
        print(f"❌ Error getting GPU info: {e}")
        return []


def check_cuda_version() -> Tuple[str, bool]:
    """Check CUDA version and H100 compatibility."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse CUDA version
        for line in output.split('\n'):
            if 'release' in line:
                version = line.split('release ')[1].split(',')[0]
                
                # H100 requires CUDA 11.8+
                major, minor = map(int, version.split('.'))
                is_compatible = major > 11 or (major == 11 and minor >= 8)
                
                return version, is_compatible
                
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Not found", False
    
    return "Unknown", False


def check_tensorflow() -> Dict:
    """Check TensorFlow version and GPU support."""
    try:
        import tensorflow as tf
        
        # Basic TensorFlow info
        tf_version = tf.__version__
        
        # Check GPU availability
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        
        # Check if TensorFlow can see GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        # TensorFlow version compatibility with H100
        version_parts = tf_version.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        # TensorFlow 2.8+ has better H100 support
        h100_compatible = major > 2 or (major == 2 and minor >= 8)
        
        return {
            'version': tf_version,
            'gpu_available': gpu_available,
            'gpu_count': len(gpus),
            'h100_compatible': h100_compatible,
            'gpu_devices': [device.name for device in gpus]
        }
        
    except ImportError:
        return {
            'version': 'Not installed',
            'gpu_available': False,
            'gpu_count': 0,
            'h100_compatible': False,
            'gpu_devices': []
        }


def check_cudnn() -> Tuple[str, bool]:
    """Check cuDNN version."""
    try:
        import tensorflow as tf
        
        # Try to get cuDNN version through TensorFlow
        build_info = tf.sysconfig.get_build_info()
        cudnn_version = build_info.get('cudnn_version', 'Unknown')
        
        # H100 requires cuDNN 8.6+
        if cudnn_version != 'Unknown':
            version_parts = cudnn_version.split('.')
            if len(version_parts) >= 2:
                major, minor = int(version_parts[0]), int(version_parts[1])
                is_compatible = major > 8 or (major == 8 and minor >= 6)
                return cudnn_version, is_compatible
        
        return cudnn_version, False
        
    except:
        return "Unknown", False


def print_compatibility_report():
    """Print a comprehensive compatibility report."""
    print("🔍 H100 GPU Compatibility Report")
    print("=" * 50)
    
    # GPU Information
    print("\n📊 GPU Information:")
    gpus = get_gpu_info()
    
    if gpus:
        h100_found = False
        for gpu in gpus:
            status = "🟢 H100" if gpu['is_h100'] else "🔵 Other"
            print(f"  {status} GPU {gpu['index']}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_free']}/{gpu['memory_total']} GB")
            print(f"    Compute: {gpu['compute_capability']}")
            
            if gpu['is_h100']:
                h100_found = True
        
        if h100_found:
            print("✅ H100 GPU detected!")
        else:
            print("ℹ️ No H100 GPU found (testing compatibility anyway)")
    else:
        print("❌ No GPU information available")
    
    # CUDA Version
    print("\n🛠️ CUDA Version:")
    cuda_version, cuda_compatible = check_cuda_version()
    status = "✅" if cuda_compatible else "❌"
    print(f"  {status} CUDA: {cuda_version}")
    if not cuda_compatible:
        print("    ⚠️ H100 requires CUDA 11.8+")
    
    # cuDNN Version
    print("\n📚 cuDNN Version:")
    cudnn_version, cudnn_compatible = check_cudnn()
    status = "✅" if cudnn_compatible else "❌"
    print(f"  {status} cuDNN: {cudnn_version}")
    if not cudnn_compatible:
        print("    ⚠️ H100 requires cuDNN 8.6+")
    
    # TensorFlow Information
    print("\n🧠 TensorFlow Configuration:")
    tf_info = check_tensorflow()
    
    status = "✅" if tf_info['h100_compatible'] else "❌"
    print(f"  {status} TensorFlow: {tf_info['version']}")
    
    if tf_info['gpu_available']:
        print(f"  ✅ GPU Support: {tf_info['gpu_count']} GPU(s) detected")
        for device in tf_info['gpu_devices']:
            print(f"    - {device}")
    else:
        print("  ❌ GPU Support: No GPUs detected by TensorFlow")
    
    if not tf_info['h100_compatible']:
        print("    ⚠️ Consider upgrading to TensorFlow 2.8+ for better H100 support")
    
    # Overall Recommendation
    print("\n💡 Recommendations:")
    
    overall_compatible = cuda_compatible and cudnn_compatible and tf_info['h100_compatible']
    
    if overall_compatible:
        print("✅ Your environment is H100 compatible!")
        print("  - You can run DeepMind research projects on H100 GPUs")
        print("  - Consider enabling mixed precision for better performance")
    else:
        print("⚠️ Your environment needs updates for H100 compatibility:")
        
        if not cuda_compatible:
            print("  - Update CUDA to 11.8+")
        if not cudnn_compatible:
            print("  - Update cuDNN to 8.6+")
        if not tf_info['h100_compatible']:
            print("  - Upgrade TensorFlow to 2.8+")
            print("  - For Enformer: pip install -r requirements-h100.txt")
    
    print("\n📖 For detailed instructions, see: H100-COMPATIBILITY.md")


def main():
    """Main function."""
    print_compatibility_report()


if __name__ == "__main__":
    main()
