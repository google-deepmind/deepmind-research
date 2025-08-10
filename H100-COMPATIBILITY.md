# H100 GPU Compatibility Guide for DeepMind Research

This document provides solutions for running DeepMind research projects on NVIDIA H100 GPUs (Hopper architecture).

## Problem

The H100 GPU requires specific CUDA and cuDNN versions that are not compatible with older TensorFlow versions used in many DeepMind research projects.

**H100 GPU Requirements:**
- CUDA 11.8+ (minimum)
- cuDNN 8.6+ (minimum)
- TensorFlow 2.8+ (recommended for full H100 support)

## Compatibility Matrix

| Project | Current TF Version | H100 Compatible | Recommended Version | Status |
|---------|-------------------|-----------------|---------------------|---------|
| Enformer | 2.5.0 | ⚠️ Limited | 2.10.0+ | Partial support |
| Sketchy | 2.0.0 | ❌ No | 2.10.0+ | Needs upgrade |
| Transporter | 1.13.1 | ❌ No | 2.10.0+ | Major upgrade needed |
| ScratchGAN | 1.15 | ❌ No | 2.10.0+ | Major upgrade needed |

## Solutions

### Option 1: Upgrade to H100-Compatible TensorFlow (Recommended)

For projects that can be upgraded, we provide updated requirements files with H100-compatible versions.

#### Enformer H100 Support
```bash
# Use the new H100-compatible requirements
pip install -r requirements-h100.txt
```

### Option 2: Docker Container with Proper CUDA Setup

For projects requiring specific TensorFlow versions, use containerized environments:

```dockerfile
# Example for TensorFlow 2.5.0 with H100 support
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04
# Install compatible TensorFlow version...
```

### Option 3: Environment-Specific Installation

Use conda environments with specific CUDA versions:

```bash
# Create H100 environment
conda create -n h100-env python=3.8
conda activate h100-env
conda install cudatoolkit=11.8 cudnn=8.6
pip install tensorflow==2.10.0  # H100 compatible
```

## Project-Specific Solutions

### Enformer Project

The Enformer project can be upgraded to TensorFlow 2.10+ for full H100 support:

1. **Use updated requirements**: `requirements-h100.txt`
2. **Verify compatibility**: Run the included test script
3. **Performance benefits**: ~2-3x faster training on H100

### Legacy Projects (TF 1.x)

For projects using TensorFlow 1.x:

1. **Containerization**: Use Docker with TF 1.x + CUDA 10.2
2. **CPU fallback**: Run on CPU for development/testing
3. **Migration guide**: Follow TF 2.x migration path

## Testing H100 Compatibility

Use the provided test script to verify GPU compatibility:

```bash
python test_h100_compatibility.py
```

## Performance Optimization

Once H100 compatibility is achieved:

1. **Mixed precision**: Enable automatic mixed precision (AMP)
2. **Memory optimization**: Use gradient checkpointing
3. **Batch size tuning**: Leverage H100's 80GB memory

## Migration Timeline

- **Phase 1**: Update Enformer and recent projects (TF 2.x)
- **Phase 2**: Containerize legacy projects (TF 1.x)
- **Phase 3**: Full migration for actively maintained projects

## Contributing

When adding new research code:
- Use TensorFlow 2.10+ for H100 compatibility
- Include both standard and H100-optimized requirements
- Test on both V100 and H100 if available

## References

- [NVIDIA H100 Developer Guide](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/)
- [TensorFlow GPU Support Guide](https://www.tensorflow.org/install/gpu)
- [CUDA Compatibility Matrix](https://docs.nvidia.com/deploy/cuda-compatibility/)
