#!/usr/bin/env python3
"""
MeshGraphNet Dataset Download Tool

This script provides automated download functionality for MeshGraphNet datasets
from Google Cloud Storage. It addresses Issue #596 where the original download
script was failing due to URL construction issues.

Usage:
    python download_meshgraphnet_datasets.py --dataset flag_simple --output_dir ./data
    python download_meshgraphnet_datasets.py --dataset cylinder_flow --output_dir /tmp
    python download_meshgraphnet_datasets.py --list-datasets

Available datasets:
    - flag_simple: Simple flag simulation dataset
    - cylinder_flow: Cylinder flow CFD dataset  
    - deforming_plate: Deforming plate simulation
    - sphere_simple: Simple sphere simulation

Fixes Issue #596: MeshGraphNet Dataset Link is giving 404 not found error
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
from typing import List, Optional
from pathlib import Path


class MeshGraphNetDownloader:
    """Download MeshGraphNet datasets from Google Cloud Storage."""
    
    BASE_URL = "https://storage.googleapis.com/dm-meshgraphnets"
    
    AVAILABLE_DATASETS = {
        "airfoil": "Airfoil simulation dataset (CFD around airfoils)",
        "flag_simple": "Simple flag simulation dataset (cloth dynamics)",
        "cylinder_flow": "Cylinder flow CFD dataset (fluid dynamics)", 
        "deforming_plate": "Deforming plate simulation dataset",
        "sphere_simple": "Simple sphere simulation dataset"
    }
    
    REQUIRED_FILES = ["meta.json", "train.tfrecord", "valid.tfrecord", "test.tfrecord"]
    
    def __init__(self, output_dir: str = "/tmp"):
        """Initialize downloader with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_progress_hook(self, block_num: int, block_size: int, total_size: int):
        """Progress callback for download."""
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100.0, (downloaded / total_size) * 100.0)
            bar_length = 50
            filled_length = int(bar_length * percent // 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Convert bytes to human readable format
            def humanize_bytes(bytes_val):
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.1f}{unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.1f}TB"
            
            downloaded_str = humanize_bytes(downloaded)
            total_str = humanize_bytes(total_size)
            
            print(f"\r    Progress: [{bar}] {percent:.1f}% ({downloaded_str}/{total_str})", 
                  end='', flush=True)
    
    def download_file(self, url: str, output_path: Path) -> bool:
        """Download a single file with progress tracking."""
        try:
            print(f"\n  📁 Downloading: {output_path.name}")
            print(f"     URL: {url}")
            
            urllib.request.urlretrieve(url, output_path, self.download_progress_hook)
            print()  # New line after progress bar
            
            # Verify file was downloaded and has content
            if output_path.exists() and output_path.stat().st_size > 0:
                file_size = self.humanize_bytes(output_path.stat().st_size)
                print(f"  ✅ Successfully downloaded: {output_path.name} ({file_size})")
                return True
            else:
                print(f"  ❌ Download failed: File is empty or doesn't exist")
                return False
                
        except urllib.error.HTTPError as e:
            print(f"\n  ❌ HTTP Error {e.code}: {e.reason}")
            print(f"     URL: {url}")
            if e.code == 404:
                print(f"     The file may not exist or dataset name may be incorrect.")
            return False
        except urllib.error.URLError as e:
            print(f"\n  ❌ URL Error: {e.reason}")
            return False
        except Exception as e:
            print(f"\n  ❌ Unexpected error: {str(e)}")
            return False
    
    @staticmethod
    def humanize_bytes(bytes_val: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f}TB"
    
    def download_dataset(self, dataset_name: str) -> bool:
        """Download a complete dataset."""
        if dataset_name not in self.AVAILABLE_DATASETS:
            print(f"❌ Error: Dataset '{dataset_name}' not found.")
            print("Available datasets:")
            self.list_datasets()
            return False
        
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting download of dataset: {dataset_name}")
        print(f"📂 Output directory: {dataset_dir}")
        print(f"📋 Description: {self.AVAILABLE_DATASETS[dataset_name]}")
        
        all_success = True
        total_files = len(self.REQUIRED_FILES)
        
        for i, filename in enumerate(self.REQUIRED_FILES, 1):
            print(f"\n📥 Downloading file {i}/{total_files}: {filename}")
            
            # Construct URL properly to avoid double slashes
            url = f"{self.BASE_URL}/{dataset_name}/{filename}"
            output_path = dataset_dir / filename
            
            success = self.download_file(url, output_path)
            if not success:
                all_success = False
                # Don't break - try to download other files
        
        if all_success:
            print(f"\n🎉 Successfully downloaded dataset '{dataset_name}'!")
            print(f"📁 Files saved to: {dataset_dir}")
            
            # Show dataset summary
            print(f"\n📊 Dataset Summary:")
            total_size = 0
            for filename in self.REQUIRED_FILES:
                file_path = dataset_dir / filename
                if file_path.exists():
                    size = file_path.stat().st_size
                    total_size += size
                    print(f"  • {filename}: {self.humanize_bytes(size)}")
            
            print(f"  Total size: {self.humanize_bytes(total_size)}")
            
        else:
            print(f"\n⚠️  Some files failed to download for dataset '{dataset_name}'")
            print("Please check the error messages above and try again.")
        
        return all_success
    
    def list_datasets(self):
        """List all available datasets."""
        print("📋 Available MeshGraphNet datasets:")
        for name, description in self.AVAILABLE_DATASETS.items():
            print(f"  • {name}: {description}")
    
    def verify_dataset(self, dataset_name: str) -> bool:
        """Verify that all required files exist for a dataset."""
        dataset_dir = self.output_dir / dataset_name
        
        if not dataset_dir.exists():
            print(f"❌ Dataset directory does not exist: {dataset_dir}")
            return False
        
        missing_files = []
        for filename in self.REQUIRED_FILES:
            file_path = dataset_dir / filename
            if not file_path.exists() or file_path.stat().st_size == 0:
                missing_files.append(filename)
        
        if missing_files:
            print(f"❌ Missing or empty files in {dataset_name}:")
            for filename in missing_files:
                print(f"  • {filename}")
            return False
        else:
            print(f"✅ All required files present for dataset: {dataset_name}")
            return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download MeshGraphNet datasets from Google Cloud Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download flag_simple dataset to current directory:
    python download_meshgraphnet_datasets.py --dataset flag_simple

  Download cylinder_flow dataset to specific directory:
    python download_meshgraphnet_datasets.py --dataset cylinder_flow --output_dir ./data

  List all available datasets:
    python download_meshgraphnet_datasets.py --list-datasets

  Verify downloaded dataset:
    python download_meshgraphnet_datasets.py --verify flag_simple

This tool fixes Issue #596: MeshGraphNet Dataset Link giving 404 not found error.
        """
    )
    
    parser.add_argument(
        "--dataset", 
        type=str,
        help="Name of the dataset to download"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/tmp",
        help="Output directory for downloaded datasets (default: /tmp)"
    )
    
    parser.add_argument(
        "--list-datasets",
        action="store_true", 
        help="List all available datasets"
    )
    
    parser.add_argument(
        "--verify",
        type=str,
        help="Verify that a dataset has all required files"
    )
    
    args = parser.parse_args()
    
    # Create downloader instance
    downloader = MeshGraphNetDownloader(args.output_dir)
    
    if args.list_datasets:
        downloader.list_datasets()
        return 0
    
    if args.verify:
        success = downloader.verify_dataset(args.verify)
        return 0 if success else 1
    
    if args.dataset:
        success = downloader.download_dataset(args.dataset)
        return 0 if success else 1
    
    # If no arguments provided, show help
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
