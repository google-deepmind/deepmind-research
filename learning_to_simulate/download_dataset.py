#!/usr/bin/env python3
"""
Cross-platform dataset downloader for Learning to Simulate project.

This script provides a robust, cross-platform alternative to the bash script
for downloading datasets. It includes progress bars, resume capability, and
better error handling.

Usage:
    python download_dataset.py <dataset_name> <output_dir>
    
Example:
    python download_dataset.py WaterDrop ./datasets
    python download_dataset.py Sand ./datasets
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url, filepath, resume=True):
    """Download a file with progress bar and resume capability."""
    headers = {}
    mode = 'wb'
    
    # Check if file exists and get its size for resume
    if resume and filepath.exists():
        existing_size = filepath.stat().st_size
        headers['Range'] = f'bytes={existing_size}-'
        mode = 'ab'
        print(f"Resuming download from byte {existing_size}")
    
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        if 'content-range' in response.headers:
            # If resuming, adjust total size
            total_size = int(response.headers['content-range'].split('/')[-1])
        
        # Create progress bar
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=filepath.name
        )
        
        # If resuming, set initial position
        if mode == 'ab' and filepath.exists():
            progress_bar.update(filepath.stat().st_size)
        
        # Download file
        with open(filepath, mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        print(f"✅ Successfully downloaded {filepath.name}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {filepath.name}: {e}")
        return False


def download_dataset(dataset_name, output_dir):
    """Download all files for a specific dataset."""
    base_url = f"https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{dataset_name}/"
    files = ['metadata.json', 'train.tfrecord', 'valid.tfrecord', 'test.tfrecord']
    
    # Create output directory
    dataset_dir = Path(output_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Downloading dataset '{dataset_name}' to {dataset_dir}")
    
    success_count = 0
    for filename in files:
        url = base_url + filename
        filepath = dataset_dir / filename
        
        print(f"\n📥 Downloading {filename}...")
        if download_file(url, filepath):
            success_count += 1
        else:
            print(f"⚠️ Failed to download {filename}")
    
    print(f"\n🎉 Download complete! {success_count}/{len(files)} files downloaded successfully.")
    
    if success_count == len(files):
        print(f"✅ All files downloaded to: {dataset_dir}")
        return True
    else:
        print(f"⚠️ Some downloads failed. Check your internet connection and try again.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Learning to Simulate project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  - WaterDrop
  - Sand  
  - Goop
  - MultiMaterial
  - RandomFloor
  - Continuous
  - WaterDropSample
  - SandSample
  - GoopSample
  - MultiMaterialSample
  - RandomFloorSample
  - ContinuousSample

Examples:
  python download_dataset.py WaterDrop ./datasets
  python download_dataset.py Sand /tmp/learning_to_simulate
        """
    )
    
    parser.add_argument('dataset_name', help='Name of the dataset to download')
    parser.add_argument('output_dir', help='Output directory to save the dataset')
    
    args = parser.parse_args()
    
    # Validate dataset name
    valid_datasets = [
        'WaterDrop', 'Sand', 'Goop', 'MultiMaterial', 'RandomFloor', 'Continuous',
        'WaterDropSample', 'SandSample', 'GoopSample', 'MultiMaterialSample', 
        'RandomFloorSample', 'ContinuousSample'
    ]
    
    if args.dataset_name not in valid_datasets:
        print(f"⚠️ Warning: '{args.dataset_name}' is not in the list of known datasets.")
        print(f"Valid datasets: {', '.join(valid_datasets)}")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Download dataset
    success = download_dataset(args.dataset_name, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
