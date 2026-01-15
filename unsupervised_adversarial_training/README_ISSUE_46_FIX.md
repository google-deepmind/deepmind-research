# Fix for Issue #46: TensorFlow GraphKeys Compatibility Error

## Problem Description

**Issue**: [#46 - tensorflow has no attribute GraphKeys](https://github.com/google-deepmind/deepmind-research/issues/46)

When trying to run the unsupervised adversarial training code with modern TensorFlow versions (2.x), users encounter the following error:

```
AttributeError: module 'tensorflow' has no attribute 'GraphKeys'
```

This occurs because TensorFlow 2.x moved `GraphKeys` to `tf.compat.v1.GraphKeys`, but the CleverHans library (used for adversarial attacks) expects the old TensorFlow 1.x API where `tf.GraphKeys` was directly available.

## Root Cause

1. **TensorFlow API Changes**: TensorFlow 2.x moved many v1 APIs to the `tf.compat.v1` namespace
2. **CleverHans Dependency**: The code uses CleverHans library which internally tries to access `tf.GraphKeys`
3. **Version Mismatch**: Modern TensorFlow installations don't expose `GraphKeys` at the top level

## Solution Overview

This fix provides a comprehensive compatibility layer that:

1. **Detects TensorFlow version** and automatically applies necessary compatibility shims
2. **Adds missing APIs** like `tf.GraphKeys` back to the main TensorFlow namespace
3. **Handles CleverHans dependencies** including the deprecated `tensorflow-addons` requirement
4. **Provides diagnostics** to help users troubleshoot their environment
5. **Maintains backward compatibility** with TensorFlow 1.x setups

## Files Modified

### 1. `tensorflow_compatibility.py` (NEW)
- **Purpose**: Comprehensive TensorFlow compatibility layer
- **Key Functions**:
  - `setup_tensorflow_compatibility()`: Main setup function to call before importing CleverHans
  - `check_cleverhans_compatibility()`: Validates CleverHans installation and handles dependencies
  - `diagnose_environment()`: Provides detailed environment diagnostics and recommendations

### 2. `quick_eval_cifar.py` (MODIFIED)
- **Changes**: Added import and setup of compatibility layer at the top of the file
- **Impact**: Ensures GraphKeys compatibility is established before CleverHans imports

### 3. `requirements.txt` (UPDATED)
- **Changes**: Updated dependency specifications for modern environments
- **Notes**: Removed tensorflow-addons dependency (deprecated), clarified TensorFlow version options

### 4. `test_compatibility.py` (NEW)
- **Purpose**: Comprehensive test suite to validate the fix
- **Tests**: TensorFlow import, compatibility layer, GraphKeys access, CleverHans import, full integration

## Installation Instructions

### For TensorFlow 2.x (Recommended)

```bash
# Install modern TensorFlow with compatibility layer
pip install tensorflow>=2.4.0,<3.0
pip install cleverhans==3.1.0
pip install numpy>=1.19.0 scipy>=1.5.0 absl-py>=0.10.0
```

### For TensorFlow 1.x (Legacy)

```bash
# Install legacy TensorFlow (if required)
pip install tensorflow>=1.15,<2
pip install cleverhans==3.1.0
pip install numpy>=1.19.0 scipy>=1.5.0 absl-py>=0.10.0
```

## Usage

The compatibility layer is automatically activated when running the main script:

```bash
# Run the evaluation script (compatibility is handled automatically)
python quick_eval_cifar.py --help
```

For manual setup in other scripts:

```python
# Import and setup compatibility before using CleverHans
from tensorflow_compatibility import setup_tensorflow_compatibility
setup_tensorflow_compatibility()

# Now you can safely import CleverHans
import cleverhans
from cleverhans import attacks
```

## Validation

Run the test suite to verify everything works:

```bash
python test_compatibility.py
```

Expected output:
```
🚀 Running TensorFlow GraphKeys Compatibility Tests
==================================================
✅ PASS: TensorFlow Import
✅ PASS: Compatibility Layer  
✅ PASS: GraphKeys Access
✅ PASS: CleverHans Import
✅ PASS: Full Integration
==================================================
🎉 All tests passed! Issue #46 is resolved.
```

## Technical Details

### How the Fix Works

1. **Version Detection**: Automatically detects TensorFlow version using `tf.__version__`
2. **API Bridging**: For TF 2.x, adds `tf.GraphKeys = tf.compat.v1.GraphKeys`
3. **Compatibility Mode**: Enables `tf.compat.v1.disable_v2_behavior()` for broader compatibility
4. **Dependency Handling**: Creates mock `tensorflow-addons` module if needed
5. **Error Prevention**: Validates environment and provides clear error messages

### Compatibility Matrix

| TensorFlow Version | CleverHans Version | Status | Notes |
|-------------------|--------------------|--------|-------|
| 1.15.x | 3.0.1, 3.1.0 | ✅ Native | No compatibility layer needed |
| 2.4.x - 2.18.x | 3.1.0 | ✅ With Layer | Use tensorflow_compatibility.py |
| 2.19.x+ | 3.1.0 | ⚠️ Testing | May require updates |

### Environment Variables

Optional environment variables for fine-tuning:

```bash
# Disable oneDNN optimizations if getting numerical warnings
export TF_ENABLE_ONEDNN_OPTS=0

# Enable more detailed TensorFlow logging
export TF_CPP_MIN_LOG_LEVEL=0
```

## Common Issues and Solutions

### Issue: `ModuleNotFoundError: No module named 'tensorflow_addons'`
**Solution**: This is handled automatically by the compatibility layer. The deprecated `tensorflow-addons` is mocked.

### Issue: `ImportError: cannot import name 'GraphKeys' from 'tensorflow'`
**Solution**: Ensure you import and call `setup_tensorflow_compatibility()` before importing CleverHans.

### Issue: Numerical warnings about oneDNN
**Solution**: These are harmless performance warnings. Set `TF_ENABLE_ONEDNN_OPTS=0` to disable.

### Issue: Python version compatibility warnings
**Solution**: The code works with Python 3.8+. For older Python versions, consider using TensorFlow 1.15.

## Testing on Different Environments

The fix has been tested on:

- ✅ **Python 3.12** + TensorFlow 2.18.1 + CleverHans 3.1.0
- ✅ **Windows 10/11** with PowerShell
- ✅ **Anaconda/Miniconda** environments

For other environments, run `python test_compatibility.py` to validate.

## Contributing

If you encounter issues with this fix:

1. Run `python test_compatibility.py` and share the output
2. Include your TensorFlow version (`python -c "import tensorflow as tf; print(tf.__version__)"`)
3. Include your CleverHans version (`python -c "import cleverhans; print(cleverhans.__version__)"`)
4. Include your Python version (`python --version`)

## References

- **Original Issue**: [DeepMind Research #46](https://github.com/google-deepmind/deepmind-research/issues/46)
- **TensorFlow Migration Guide**: [TF 1.x to 2.x Migration](https://www.tensorflow.org/guide/migrate)
- **CleverHans Documentation**: [CleverHans Library](https://github.com/cleverhans-lab/cleverhans)
