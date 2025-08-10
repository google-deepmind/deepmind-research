#!/usr/bin/env python3
"""Test script to verify TensorFlow GraphKeys compatibility fix for Issue #46.

This script tests that the compatibility layer correctly resolves the GraphKeys
AttributeError that occurs when using CleverHans with TensorFlow 2.x.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def test_tensorflow_import():
    """Test that TensorFlow imports correctly."""
    print("Testing TensorFlow import...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        return tf
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return None

def test_compatibility_layer():
    """Test that our compatibility layer works."""
    print("\nTesting compatibility layer...")
    try:
        from tensorflow_compatibility import setup_tensorflow_compatibility, diagnose_environment
        print("✅ Compatibility layer imported successfully")
        
        # Run setup
        setup_tensorflow_compatibility()
        print("✅ Compatibility setup completed")
        
        # Run diagnostics
        print("\n--- Environment Diagnostics ---")
        diagnose_environment()
        
        return True
    except Exception as e:
        print(f"❌ Compatibility layer failed: {e}")
        return False

def test_graphkeys_access():
    """Test that tf.GraphKeys is accessible after compatibility setup."""
    print("\nTesting GraphKeys access...")
    try:
        import tensorflow as tf
        
        # This should work now with our compatibility layer
        keys = tf.GraphKeys
        print(f"✅ tf.GraphKeys accessible: {type(keys)}")
        
        # Test some common GraphKeys that CleverHans uses
        common_keys = ['GLOBAL_VARIABLES', 'TRAINABLE_VARIABLES', 'UPDATE_OPS']
        for key in common_keys:
            if hasattr(keys, key):
                print(f"✅ tf.GraphKeys.{key} available")
            else:
                print(f"⚠️  tf.GraphKeys.{key} not found")
        
        return True
    except AttributeError as e:
        print(f"❌ GraphKeys access failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_cleverhans_import():
    """Test that CleverHans imports without GraphKeys errors."""
    print("\nTesting CleverHans import...")
    try:
        import cleverhans
        print(f"✅ CleverHans {cleverhans.__version__} imported successfully")
        
        # Test importing attacks module (this often triggers GraphKeys usage)
        from cleverhans import attacks
        print("✅ CleverHans attacks module imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ CleverHans import failed: {e}")
        print(f"   Error details: {type(e).__name__}: {e}")
        return False

def test_full_integration():
    """Test the full integration as used in quick_eval_cifar.py."""
    print("\nTesting full integration...")
    try:
        # Simulate the import pattern from quick_eval_cifar.py
        from tensorflow_compatibility import setup_tensorflow_compatibility
        setup_tensorflow_compatibility()
        
        import cleverhans
        from cleverhans import attacks
        import tensorflow as tf
        
        # Test that we can access GraphKeys through TensorFlow
        _ = tf.GraphKeys.GLOBAL_VARIABLES
        
        print("✅ Full integration test passed")
        return True
    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        return False

def main():
    """Run all compatibility tests."""
    print("🚀 Running TensorFlow GraphKeys Compatibility Tests")
    print("=" * 50)
    
    tests = [
        ("TensorFlow Import", test_tensorflow_import),
        ("Compatibility Layer", test_compatibility_layer),
        ("GraphKeys Access", test_graphkeys_access),
        ("CleverHans Import", test_cleverhans_import),
        ("Full Integration", test_full_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Issue #46 is resolved.")
        print("🔧 The TensorFlow GraphKeys compatibility fix is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("🔍 You may need to install missing dependencies or check your environment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
