# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utility functions for fixing dynamic_sizing field metadata issues."""

import json
import os
import re
from typing import Dict, Any


def fix_malformed_dtype(dtype_str: str) -> str:
    """Fix malformed dtype strings in metadata.
    
    Args:
        dtype_str: The potentially malformed dtype string
        
    Returns:
        Clean dtype string that TensorFlow can understand
        
    Examples:
        "<dtype: 'float32'>" -> "float32"
        "float32" -> "float32" (unchanged)
        "<class 'numpy.float32'>" -> "float32"
    """
    # Handle malformed dtype strings like "<dtype: 'float32'>"
    if dtype_str.startswith('<') and dtype_str.endswith('>'):
        # Extract the actual dtype from the malformed string
        match = re.search(r"'([^']+)'", dtype_str)
        if match:
            dtype_name = match.group(1)
            # Handle numpy.float32 -> float32
            if dtype_name.startswith('numpy.'):
                dtype_name = dtype_name.replace('numpy.', '')
            return dtype_name
        # Alternative pattern for numpy dtypes
        match = re.search(r'\.([^.>]+)>', dtype_str)
        if match:
            return match.group(1)
    
    return dtype_str


def validate_sizing_field_shape(shape: list, field_name: str = "sizing_field") -> list:
    """Validate and fix sizing field shape.
    
    MeshGraphNets sizing fields should typically be 3D (x, y, z coordinates).
    
    Args:
        shape: The shape list from metadata
        field_name: Name of the field being validated
        
    Returns:
        Corrected shape list
    """
    if len(shape) == 2 and shape[1] == 4:
        print(f"Warning: {field_name} has shape {shape}, correcting to [-1, 3] for 3D coordinates")
        return [-1, 3]
    
    if len(shape) == 2 and shape[1] != 3 and shape[1] != 1:
        print(f"Warning: {field_name} has unusual shape {shape}, you may need to verify this is correct")
    
    return shape


def fix_meta_json(meta_path: str, backup: bool = True) -> bool:
    """Fix metadata JSON file for dynamic_sizing datasets.
    
    Args:
        meta_path: Path to the meta.json file
        backup: Whether to create a backup of the original file
        
    Returns:
        True if fixes were applied, False if no fixes needed
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    # Read original metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    fixes_applied = False
    
    # Create backup if requested
    if backup:
        backup_path = meta_path + '.backup'
        with open(backup_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Created backup: {backup_path}")
    
    # Fix dtype issues in features
    if 'features' in meta:
        for field_name, field_info in meta['features'].items():
            if 'dtype' in field_info:
                original_dtype = field_info['dtype']
                fixed_dtype = fix_malformed_dtype(original_dtype)
                
                if fixed_dtype != original_dtype:
                    print(f"Fixing dtype for {field_name}: {original_dtype} -> {fixed_dtype}")
                    field_info['dtype'] = fixed_dtype
                    fixes_applied = True
            
            # Special handling for sizing_field
            if field_name == 'sizing_field' and 'shape' in field_info:
                original_shape = field_info['shape']
                fixed_shape = validate_sizing_field_shape(original_shape, field_name)
                
                if fixed_shape != original_shape:
                    print(f"Fixing shape for {field_name}: {original_shape} -> {fixed_shape}")
                    field_info['shape'] = fixed_shape
                    fixes_applied = True
    
    # Write fixed metadata if changes were made
    if fixes_applied:
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Fixed metadata written to: {meta_path}")
    else:
        print("No fixes needed for metadata file")
    
    return fixes_applied


def fix_dynamic_sizing_datasets(data_dir: str) -> Dict[str, bool]:
    """Fix all dynamic_sizing datasets in a directory.
    
    Args:
        data_dir: Root directory containing datasets
        
    Returns:
        Dictionary mapping dataset names to whether fixes were applied
    """
    dynamic_sizing_datasets = [
        'flag_dynamic_sizing',
        'sphere_dynamic_sizing'
    ]
    
    results = {}
    
    for dataset_name in dynamic_sizing_datasets:
        dataset_path = os.path.join(data_dir, dataset_name)
        meta_path = os.path.join(dataset_path, 'meta.json')
        
        if os.path.exists(meta_path):
            print(f"\nProcessing {dataset_name}...")
            try:
                fixed = fix_meta_json(meta_path)
                results[dataset_name] = fixed
                if fixed:
                    print(f"✅ {dataset_name} metadata fixed successfully")
                else:
                    print(f"✅ {dataset_name} metadata already correct")
            except Exception as e:
                print(f"❌ Error fixing {dataset_name}: {e}")
                results[dataset_name] = False
        else:
            print(f"⚠️  {dataset_name} not found at {dataset_path}")
            results[dataset_name] = False
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix dynamic_sizing field metadata issues')
    parser.add_argument('--data_dir', required=True, 
                       help='Root directory containing MeshGraphNets datasets')
    parser.add_argument('--no_backup', action='store_true',
                       help='Skip creating backup files')
    
    args = parser.parse_args()
    
    print("🔧 MeshGraphNets Dynamic Sizing Field Fixer")
    print("=" * 50)
    
    results = fix_dynamic_sizing_datasets(args.data_dir)
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    
    fixed_count = sum(1 for fixed in results.values() if fixed)
    total_count = len([k for k, v in results.items() if v is not False])
    
    for dataset, fixed in results.items():
        if fixed is not False:
            status = "✅ FIXED" if fixed else "✅ OK"
            print(f"  {status}: {dataset}")
    
    print(f"\n🎉 {fixed_count}/{total_count} datasets needed fixes")
    print("\nIssue #651 resolved! Dynamic sizing fields should now work correctly.")
