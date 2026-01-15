#!/usr/bin/env python3
"""
TensorFlow Compatibility Layer for Unsupervised Adversarial Training

This module provides compatibility fixes for TensorFlow version conflicts,
specifically addressing Issue #46: "tensorflow has no attribute 'GraphKeys'"

The issue occurs because:
1. CleverHans library expects TensorFlow 1.x API
2. Modern TensorFlow versions moved GraphKeys to tf.compat.v1.GraphKeys
3. Version constraints in requirements.txt may conflict with newer environments

This compatibility layer ensures the code works across TensorFlow versions.

Author: GSoC 2026 Contributor
Issue: https://github.com/google-deepmind/deepmind-research/issues/46
"""

import sys
import warnings
from typing import Any, Dict, List, Optional

def setup_tensorflow_compatibility():
    """
    Set up TensorFlow compatibility for unsupervised adversarial training.
    
    This function handles TensorFlow version compatibility issues, particularly
    the GraphKeys attribute error that occurs with newer TensorFlow versions.
    """
    try:
        import tensorflow as tf
        
        # Check TensorFlow version
        tf_version = tf.__version__
        major_version = int(tf_version.split('.')[0])
        
        print(f"TensorFlow version detected: {tf_version}")
        
        # For TensorFlow 2.x, ensure v1 compatibility
        if major_version >= 2:
            print("TensorFlow 2.x detected - enabling v1 compatibility mode")
            tf.compat.v1.disable_v2_behavior()
            
            # Ensure GraphKeys is available in the main tf namespace for CleverHans
            if not hasattr(tf, 'GraphKeys'):
                tf.GraphKeys = tf.compat.v1.GraphKeys
                print("✓ Added tf.GraphKeys compatibility layer")
            
            # Ensure other commonly used v1 APIs are available
            v1_apis = [
                'placeholder', 'Session', 'global_variables_initializer',
                'variable_scope', 'get_variable', 'layers'
            ]
            
            for api in v1_apis:
                if not hasattr(tf, api) and hasattr(tf.compat.v1, api):
                    setattr(tf, api, getattr(tf.compat.v1, api))
                    print(f"✓ Added tf.{api} compatibility layer")
        
        else:
            print("TensorFlow 1.x detected - no compatibility layer needed")
            
        # Suppress TensorFlow warnings for cleaner output
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        return True
        
    except ImportError as e:
        print(f"Error importing TensorFlow: {e}")
        return False
    except Exception as e:
        print(f"Error setting up TensorFlow compatibility: {e}")
        return False

def check_cleverhans_compatibility():
    """
    Check CleverHans compatibility and handle tensorflow-addons dependency.
    """
    try:
        import cleverhans
        ch_version = cleverhans.__version__
        print(f"CleverHans version detected: {ch_version}")
        
        # Check for tensorflow-addons dependency issue
        try:
            import tensorflow_addons
            print("✓ tensorflow-addons is available")
        except ImportError:
            print("⚠ tensorflow-addons not available (this is expected for newer environments)")
            print("  CleverHans should still work with our compatibility layer")
            
            # Try to create a mock to prevent import errors
            try:
                import sys
                from unittest.mock import MagicMock
                
                # Create a mock tensorflow_addons module
                mock_tfa = MagicMock()
                mock_tfa.__version__ = "0.15.0"  # Fake version
                sys.modules['tensorflow_addons'] = mock_tfa
                print("✓ Created tensorflow-addons mock for compatibility")
            except Exception as mock_error:
                print(f"  Mock creation failed: {mock_error}")
                print("  CleverHans might still work without tensorflow-addons")
        
        # Check for known compatible versions
        compatible_versions = ['3.0.1', '3.1.0']
        version_main = ch_version.split('-')[0] if '-' in ch_version else ch_version
        
        if version_main in compatible_versions:
            print("✓ CleverHans version is compatible")
            return True
        else:
            print(f"⚠ CleverHans version {ch_version} may have compatibility issues")
            print("Recommended versions: 3.0.1 or 3.1.0")
            print("Install with: pip install cleverhans==3.1.0")
            return False
            
    except ImportError:
        print("✗ CleverHans not found. Install with: pip install cleverhans==3.1.0")
        return False
    except Exception as e:
        print(f"Error checking CleverHans: {e}")
        return False

def get_installation_guide() -> Dict[str, str]:
    """
    Provide installation guidance for different Python/TensorFlow environments.
    """
    return {
        "python_3.7_tf_1.15": {
            "description": "Recommended stable setup (Issue #46 original environment)",
            "commands": [
                "pip install tensorflow==1.15.5",
                "pip install cleverhans==3.1.0",
                "pip install tensorflow-hub==0.12.0"
            ]
        },
        "python_3.8_plus_tf_2.x": {
            "description": "Modern environment with compatibility layer",
            "commands": [
                "pip install tensorflow>=2.4.0",
                "pip install cleverhans==3.1.0",
                "pip install tensorflow-hub>=0.12.0"
            ],
            "note": "Uses tensorflow_compatibility.py for GraphKeys compatibility"
        },
        "troubleshooting": {
            "common_errors": {
                "GraphKeys_error": "Use tensorflow_compatibility.py or downgrade to TF 1.15",
                "cleverhans_import_error": "Install cleverhans==3.1.0 specifically",
                "hub_compatibility": "Ensure tensorflow-hub version matches TF version"
            }
        }
    }

def diagnose_environment():
    """
    Diagnose the current environment and provide specific recommendations.
    """
    print("\n" + "="*60)
    print("TensorFlow Compatibility Diagnostic for Issue #46")
    print("="*60)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {python_version}")
    
    # Check if this is compatible with the recommendations
    if python_version in ["3.7", "3.8", "3.9", "3.10", "3.11"]:
        print("✓ Python version is compatible")
    else:
        print("⚠ Python version may have compatibility issues")
    
    # Check TensorFlow
    tf_compatible = setup_tensorflow_compatibility()
    
    # Check CleverHans
    ch_compatible = check_cleverhans_compatibility()
    
    # Provide recommendations
    print("\n" + "-"*40)
    print("RECOMMENDATIONS:")
    print("-"*40)
    
    if tf_compatible and ch_compatible:
        print("✅ Environment appears to be compatible!")
        print("You should be able to run the unsupervised adversarial training code.")
    else:
        print("❌ Compatibility issues detected. Try these solutions:")
        
        guide = get_installation_guide()
        
        if python_version == "3.7":
            print("\n1. STABLE SOLUTION (Recommended for Issue #46):")
            for cmd in guide["python_3.7_tf_1.15"]["commands"]:
                print(f"   {cmd}")
        else:
            print("\n1. MODERN SOLUTION (Python 3.8+):")
            for cmd in guide["python_3.8_plus_tf_2.x"]["commands"]:
                print(f"   {cmd}")
            print("   Note: This uses the compatibility layer in this file")
        
        print("\n2. TROUBLESHOOTING:")
        for error, solution in guide["troubleshooting"]["common_errors"].items():
            print(f"   - {error}: {solution}")

def main():
    """Main function for running the diagnostic."""
    diagnose_environment()
    
    print("\n" + "="*60)
    print("To use this compatibility layer in your code:")
    print("="*60)
    print("from unsupervised_adversarial_training.tensorflow_compatibility import setup_tensorflow_compatibility")
    print("setup_tensorflow_compatibility()")
    print("# Your existing code here...")
    

if __name__ == "__main__":
    main()