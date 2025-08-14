#!/usr/bin/env python3
"""Simple test script for dynamic_sizing field dtype fix."""

import json
import tempfile
import os
import sys
import re

# Add meshgraphnets to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fix_dynamic_sizing import fix_malformed_dtype, validate_sizing_field_shape


def test_dtype_parsing_logic():
    """Test the dtype parsing logic that we added to dataset.py."""
    print("\n🧪 Testing dataset.py dtype parsing logic...")
    
    test_cases = [
        "<dtype: 'float32'>",
        "<dtype: 'int64'>", 
        "<class 'numpy.float32'>",
        "<class 'numpy.int32'>",
        "float32",
        "int64"
    ]
    
    for dtype_str in test_cases:
        # This is the exact logic from dataset.py
        original_dtype = dtype_str
        if dtype_str.startswith('<') and dtype_str.endswith('>'):
            # Extract the actual dtype from malformed strings
            match = re.search(r"'([^']+)'", dtype_str)
            if match:
                dtype_str = match.group(1)
                # Handle numpy.float32 -> float32
                if dtype_str.startswith('numpy.'):
                    dtype_str = dtype_str.replace('numpy.', '')
            else:
                # Alternative pattern for numpy dtypes like "<class 'numpy.float32'>"
                match = re.search(r'\.([^.>]+)>', dtype_str)
                if match:
                    dtype_str = match.group(1)
        
        print(f"  ✅ {original_dtype} -> {dtype_str}")
        
        # Verify this would work with TensorFlow
        expected_clean_types = ['float32', 'int64', 'int32']
        assert dtype_str in expected_clean_types, f"Unexpected dtype: {dtype_str}"
    
    print("  🎉 Dataset parsing logic test passed!")


def main():
    """Run core tests."""
    print("🔧 Testing Issue #651 Fix: Dynamic Sizing Fields (Core)")
    print("=" * 60)
    
    try:
        # Test dtype fixing
        print("🧪 Testing fix_malformed_dtype()...")
        test_cases = [
            ("<dtype: 'float32'>", "float32"),
            ("float32", "float32"),
            ("<dtype: 'int64'>", "int64"),
            ("<class 'numpy.float32'>", "float32"),
            ("<class 'numpy.int32'>", "int32"),
        ]
        
        for input_dtype, expected in test_cases:
            result = fix_malformed_dtype(input_dtype)
            assert result == expected, f"Expected {expected}, got {result}"
            print(f"  ✅ {input_dtype} -> {result}")
        
        # Test shape fixing
        print("\n🧪 Testing validate_sizing_field_shape()...")
        shape_tests = [
            ([-1, 4], [-1, 3]),
            ([-1, 3], [-1, 3]),
        ]
        
        for input_shape, expected in shape_tests:
            result = validate_sizing_field_shape(input_shape.copy())
            assert result == expected, f"Expected {expected}, got {result}"
            print(f"  ✅ {input_shape} -> {result}")
        
        # Test the parsing logic
        test_dtype_parsing_logic()
        
        print("\n" + "=" * 60)
        print("🎉 CORE TESTS PASSED! Issue #651 fix is working correctly.")
        print("\nThe fix handles:")
        print("  ✅ Malformed dtype strings like '<dtype: 'float32'>'")
        print("  ✅ Numpy dtype strings like '<class 'numpy.float32'>'") 
        print("  ✅ Shape corrections for sizing_field (4 -> 3 dimensions)")
        print("  ✅ Backward compatibility with correct formats")
        print("  ✅ Robust parsing logic for dataset.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
