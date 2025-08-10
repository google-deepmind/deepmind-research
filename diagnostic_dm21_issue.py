#!/usr/bin/env python3
"""
Diagnostic script for DM21 functionals issue #589
This script investigates why all DM21 functionals are defaulting to DM21M.

Author: GSoC 2026 Contributor
Issue: https://github.com/google-deepmind/deepmind-research/issues/589
"""

import os
import sys
import tempfile
import traceback

# Add the DM21 module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 
                               'density_functional_approximation_dm21'))

try:
    import density_functional_approximation_dm21 as dm21
    from density_functional_approximation_dm21 import neural_numint
    from pyscf import gto, dft
    import numpy as np
    print("✓ Successfully imported DM21 modules")
except ImportError as e:
    print(f"✗ Failed to import DM21 modules: {e}")
    print("This might indicate missing dependencies. Please install:")
    print("pip install pyscf tensorflow==1.15 tensorflow-hub attrs")
    sys.exit(1)

def test_functional_names():
    """Test that functional names are correctly mapped."""
    print("\n=== Testing Functional Name Mapping ===")
    
    functionals = [
        dm21.Functional.DM21,
        dm21.Functional.DM21m, 
        dm21.Functional.DM21mc,
        dm21.Functional.DM21mu
    ]
    
    for func in functionals:
        print(f"Functional {func} -> name: '{func.name}'")
        checkpoint_path = os.path.join(
            os.path.dirname(neural_numint.__file__), 
            'checkpoints', 
            func.name
        )
        print(f"  Expected checkpoint path: {checkpoint_path}")
        print(f"  Path exists: {os.path.exists(checkpoint_path)}")
        
        if os.path.exists(checkpoint_path):
            files = os.listdir(checkpoint_path)
            print(f"  Files in checkpoint: {files}")
        print()

def test_model_loading():
    """Test if models can be loaded correctly."""
    print("\n=== Testing Model Loading ===")
    
    # Create a simple test molecule
    mol = gto.Mole()
    mol.atom = 'H 0.0 0.0 0.0'
    mol.basis = 'sto-3g'
    mol.spin = 1
    mol.verbose = 0  # Suppress PySCF output
    mol.build()
    
    functionals_to_test = [
        ('DM21', dm21.Functional.DM21),
        ('DM21m', dm21.Functional.DM21m),
        ('DM21mc', dm21.Functional.DM21mc), 
        ('DM21mu', dm21.Functional.DM21mu)
    ]
    
    for name, func in functionals_to_test:
        print(f"\nTesting {name}:")
        try:
            # Try to create NeuralNumInt instance
            ni = dm21.NeuralNumInt(func)
            print(f"  ✓ Successfully created NeuralNumInt for {name}")
            print(f"  Model path: {ni._model_path}")
            print(f"  Functional name: {ni._functional_name}")
            
            # Try to create a DFT calculation
            mf = dft.UKS(mol)
            mf._numint = ni
            mf.conv_tol = 1e-6
            mf.verbose = 0
            
            # Just test that it initializes without error
            print(f"  ✓ Successfully initialized DFT calculation with {name}")
            
        except Exception as e:
            print(f"  ✗ Failed to create/use {name}: {e}")
            traceback.print_exc()

def test_tensorflow_sessions():
    """Test if TensorFlow sessions are being reused incorrectly."""
    print("\n=== Testing TensorFlow Session Management ===")
    
    try:
        # Create multiple NeuralNumInt instances
        ni_dm21 = dm21.NeuralNumInt(dm21.Functional.DM21)
        ni_dm21m = dm21.NeuralNumInt(dm21.Functional.DM21m)
        
        print(f"DM21 session: {id(ni_dm21._session)}")
        print(f"DM21m session: {id(ni_dm21m._session)}")
        print(f"DM21 graph: {id(ni_dm21._graph)}")
        print(f"DM21m graph: {id(ni_dm21m._graph)}")
        
        # Check if they point to different models
        print(f"DM21 model path: {ni_dm21._model_path}")
        print(f"DM21m model path: {ni_dm21m._model_path}")
        
        if ni_dm21._model_path == ni_dm21m._model_path:
            print("  ✗ ERROR: Both functionals using same model path!")
        else:
            print("  ✓ Functionals using different model paths")
            
    except Exception as e:
        print(f"  ✗ Error in session testing: {e}")

def test_case_sensitivity():
    """Test if there are case sensitivity issues."""
    print("\n=== Testing Case Sensitivity ===")
    
    base_dir = os.path.join(
        os.path.dirname(neural_numint.__file__), 
        'checkpoints'
    )
    
    if os.path.exists(base_dir):
        actual_dirs = os.listdir(base_dir)
        print(f"Actual checkpoint directories: {actual_dirs}")
        
        expected_dirs = ['DM21', 'DM21m', 'DM21mc', 'DM21mu']
        for expected in expected_dirs:
            if expected in actual_dirs:
                print(f"  ✓ {expected} directory exists")
            else:
                print(f"  ✗ {expected} directory missing")
                # Check for case variations
                lower_match = [d for d in actual_dirs if d.lower() == expected.lower()]
                if lower_match:
                    print(f"    Found case variation: {lower_match}")

def main():
    print("DM21 Functionals Diagnostic Tool")
    print("=" * 50)
    print("This tool diagnoses Issue #589: DM21 functionals defaulting to DM21M")
    
    test_functional_names()
    test_case_sensitivity()
    test_model_loading()
    test_tensorflow_sessions()
    
    print("\n" + "=" * 50)
    print("Diagnostic complete. Review the output above for potential issues.")
    print("\nCommon causes for this issue:")
    print("1. TensorFlow session reuse between different functionals")
    print("2. Incorrect checkpoint path mapping")
    print("3. Case sensitivity issues in filesystem")
    print("4. Model loading errors falling back to default")
    print("5. Global variables not being reset between model loads")

if __name__ == "__main__":
    main()
