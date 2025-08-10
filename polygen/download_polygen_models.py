#!/usr/bin/env python3
"""
PolyGen Pre-trained Models Downloader

This script downloads the pre-trained PolyGen models (face_model.tar.gz and 
vertex_model.tar.gz) from Google Cloud Storage and extracts them to the 
specified directory.

Usage:
    python download_polygen_models.py [--output_dir /path/to/models]
"""

import os
import sys
import argparse
import tarfile
import requests
from pathlib import Path
from typing import Optional


def download_file_with_progress(url: str, output_path: Path) -> bool:
    """Download a file with progress indication."""
    
    try:
        print(f"📥 Downloading {output_path.name}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  Progress: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
        
        print()  # New line after progress
        print(f"✅ Downloaded {output_path.name} ({downloaded:,} bytes)")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {output_path.name}: {e}")
        return False


def extract_tar_gz(tar_path: Path, extract_to: Path) -> bool:
    """Extract a tar.gz file."""
    
    try:
        print(f"📦 Extracting {tar_path.name}...")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        
        print(f"✅ Extracted {tar_path.name} to {extract_to}")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting {tar_path.name}: {e}")
        return False


def download_polygen_models(output_dir: str = "/tmp") -> bool:
    """Download and extract PolyGen pre-trained models."""
    
    base_url = "https://storage.googleapis.com/deepmind-research-polygen"
    models = {
        "vertex_model.tar.gz": "vertex_model",
        "face_model.tar.gz": "face_model"
    }
    
    output_path = Path(output_dir)
    success_count = 0
    
    print("🚀 Starting PolyGen models download...")
    print(f"📁 Output directory: {output_path.absolute()}")
    
    for model_file, extract_dir in models.items():
        url = f"{base_url}/{model_file}"
        
        # Create model-specific directory
        model_dir = output_path / extract_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the tar.gz file
        tar_path = model_dir / model_file
        
        if tar_path.exists():
            print(f"⏭️ {model_file} already exists, skipping download")
        else:
            if not download_file_with_progress(url, tar_path):
                continue
        
        # Extract the file
        if extract_tar_gz(tar_path, model_dir):
            # Clean up the tar.gz file after extraction
            tar_path.unlink()
            print(f"🗑️ Cleaned up {model_file}")
            success_count += 1
    
    if success_count == len(models):
        print("\n🎉 All models downloaded and extracted successfully!")
        print(f"📍 Models location:")
        for model_file, extract_dir in models.items():
            model_path = output_path / extract_dir
            print(f"  - {extract_dir}: {model_path.absolute()}")
        return True
    else:
        print(f"\n⚠️ {success_count}/{len(models)} models downloaded successfully")
        return False


def verify_models(output_dir: str) -> bool:
    """Verify that the models were downloaded and extracted correctly."""
    
    output_path = Path(output_dir)
    
    expected_files = {
        "vertex_model": ["checkpoint", "model.data-00000-of-00001", "model.index", "model.meta"],
        "face_model": ["checkpoint", "model.data-00000-of-00001", "model.index", "model.meta"]
    }
    
    print("\n🔍 Verifying downloaded models...")
    
    all_good = True
    for model_name, required_files in expected_files.items():
        model_dir = output_path / model_name
        
        if not model_dir.exists():
            print(f"❌ {model_name} directory not found")
            all_good = False
            continue
            
        missing_files = []
        for file_name in required_files:
            file_path = model_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ {model_name} missing files: {missing_files}")
            all_good = False
        else:
            print(f"✅ {model_name} verified")
    
    return all_good


def main():
    """Main command line interface."""
    
    parser = argparse.ArgumentParser(
        description="Download PolyGen pre-trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to /tmp (default)
  python download_polygen_models.py
  
  # Download to custom directory
  python download_polygen_models.py --output_dir ./models
  
  # Just verify existing models
  python download_polygen_models.py --verify_only --output_dir ./models
        """
    )
    
    parser.add_argument('--output_dir', type=str, default='/tmp',
                       help='Directory to download and extract models (default: /tmp)')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify existing models without downloading')
    
    args = parser.parse_args()
    
    if args.verify_only:
        if verify_models(args.output_dir):
            print("✅ All models verified successfully!")
            return 0
        else:
            print("❌ Model verification failed")
            return 1
    
    # Download models
    if download_polygen_models(args.output_dir):
        # Verify after download
        if verify_models(args.output_dir):
            print("✅ Download and verification completed successfully!")
            return 0
        else:
            print("⚠️ Download completed but verification failed")
            return 1
    else:
        print("❌ Download failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
