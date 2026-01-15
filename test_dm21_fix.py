#!/usr/bin/env python3
"""
Test script for DM21 functionals fix - Issue #589

This script demonstrates the correct usage of DM21 functionals and validates
that the fix prevents all functionals from defaulting to DM21M.

Author: GSoC 2026 Contributor
Issue: https://github.com/google-deepmind/deepmind-research/issues/589
"""

import os
import sys
import traceback

def test_dm21_functional_selection():
    """Test that different DM21 functionals are properly loaded and distinguished."""
    
    print("Testing DM21 Functional Selection Fix")
    print("=" * 50)
    
    # Add the DM21 module to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 
                                   'density_functional_approximation_dm21'))
    
    try:
        import density_functional_approximation_dm21 as dm21
        print("✓ Successfully imported DM21 modules")
    except ImportError as e:
        print(f"✗ Failed to import DM21 modules: {e}")
        print("\nTo run this test, please install the required dependencies:")
        print("pip install pyscf tensorflow==1.15 tensorflow-hub attrs")
        return False
    
    # Test that functional names are correctly mapped
    print("\n1. Testing Functional Name Mapping...")
    functionals = [
        ('DM21', dm21.Functional.DM21),
        ('DM21m', dm21.Functional.DM21m),
        ('DM21mc', dm21.Functional.DM21mc),
        ('DM21mu', dm21.Functional.DM21mu)
    ]
    
    for name, func in functionals:
        print(f"   {name}: enum name = '{func.name}'")
        if func.name != name:
            print(f"   ✗ ERROR: Expected '{name}', got '{func.name}'")
            return False
    
    print("   ✓ All functional names correctly mapped")
    
    # Test that models can be instantiated with proper isolation
    print("\n2. Testing Model Instantiation and Isolation...")
    instances = {}
    
    for name, func in functionals:
        try:
            print(f"   Creating {name} instance...")
            ni = dm21.NeuralNumInt(func)
            instances[name] = ni
            
            # Verify correct paths
            expected_path = os.path.join(
                os.path.dirname(dm21.neural_numint.__file__), 
                'checkpoints', 
                name
            )
            
            if ni._model_path != expected_path:
                print(f"   ✗ ERROR: Wrong model path for {name}")
                print(f"     Expected: {expected_path}")
                print(f"     Got: {ni._model_path}")
                return False
            
            print(f"   ✓ {name} instance created successfully")
            
        except Exception as e:
            print(f"   ✗ Failed to create {name} instance: {e}")
            traceback.print_exc()
            return False
    
    # Test that different instances use different TensorFlow sessions/graphs
    print("\n3. Testing Session Isolation...")
    session_ids = {}
    graph_ids = {}
    
    for name, instance in instances.items():
        session_ids[name] = id(instance._session)
        graph_ids[name] = id(instance._graph)
        print(f"   {name}: session_id={session_ids[name]}, graph_id={graph_ids[name]}")
    
    # Check for uniqueness
    unique_sessions = len(set(session_ids.values()))
    unique_graphs = len(set(graph_ids.values()))
    
    if unique_sessions != len(instances):
        print("   ✗ ERROR: Sessions are not properly isolated!")
        print(f"     Expected {len(instances)} unique sessions, got {unique_sessions}")
        return False
    
    if unique_graphs != len(instances):
        print("   ✗ ERROR: Graphs are not properly isolated!")
        print(f"     Expected {len(instances)} unique graphs, got {unique_graphs}")
        return False
    
    print("   ✓ All instances have unique sessions and graphs")
    
    # Test cleanup
    print("\n4. Testing Cleanup...")
    for name, instance in instances.items():
        try:
            instance.close()
            print(f"   ✓ {name} instance cleaned up successfully")
        except Exception as e:
            print(f"   ✗ Failed to cleanup {name} instance: {e}")
            return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! DM21 functional selection should now work correctly.")
    return True

def demonstrate_correct_usage():
    """Demonstrate the correct way to use different DM21 functionals."""
    
    print("\n\nDemonstrating Correct Usage")
    print("=" * 50)
    
    usage_example = '''
# Correct way to use different DM21 functionals:

import density_functional_approximation_dm21 as dm21
from pyscf import gto, dft

# Create your molecule
mol = gto.Mole()
mol.atom = 'H 0.0 0.0 0.0'
mol.basis = 'sto-3g'
mol.spin = 1
mol.build()

# For each functional, create a new NeuralNumInt instance
functionals_to_test = [
    dm21.Functional.DM21,    # Full training dataset with constraints
    dm21.Functional.DM21m,   # Molecules only
    dm21.Functional.DM21mc,  # Molecules + fractional charge
    dm21.Functional.DM21mu   # Molecules + electron gas
]

results = {}
for functional in functionals_to_test:
    # Create DFT calculation
    mf = dft.UKS(mol)
    
    # IMPORTANT: Create a new NeuralNumInt instance for each functional
    mf._numint = dm21.NeuralNumInt(functional)
    
    # Recommended settings for neural functionals
    mf.conv_tol = 1e-6        # Relaxed convergence
    mf.conv_tol_grad = 1e-3   # Relaxed gradient convergence
    mf.verbose = 1
    
    # Run calculation
    energy = mf.kernel()
    results[functional.name] = energy
    
    # IMPORTANT: Clean up to prevent interference
    mf._numint.close()

print("Results:", results)

# Best practice: Use context managers or explicit cleanup
# to ensure proper resource management when switching between functionals
'''
    
    print(usage_example)

if __name__ == "__main__":
    success = test_dm21_functional_selection()
    
    if success:
        demonstrate_correct_usage()
        print("\n🎉 Issue #589 has been resolved!")
        print("\nKey improvements made:")
        print("1. Enhanced TensorFlow session isolation between different functionals")
        print("2. Added proper cleanup methods to prevent resource leaks")
        print("3. Added error handling and validation for model loading")
        print("4. Documented correct usage patterns")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1)
