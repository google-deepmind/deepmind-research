#!/usr/bin/env python3
"""
WikiGraphs Dataset Download Tool

This script provides automated download functionality for WikiGraphs datasets,
including WikiText-103 and Freebase graph data. It addresses Issue #575 where
the original download script was failing due to broken S3 links.

Usage:
    python download_wikigraphs_datasets.py --all --output_dir /tmp/data
    python download_wikigraphs_datasets.py --wikitext --output_dir ./data
    python download_wikigraphs_datasets.py --freebase --output_dir ./data
    python download_wikigraphs_datasets.py --verify /tmp/data

Fixes Issue #575: file not found error '/tmp/data/wikitext-103/wiki.train.tokens'
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
import zipfile
import tarfile
import tempfile
from typing import Optional, List
from pathlib import Path


class WikiGraphsDownloader:
    """Download WikiGraphs datasets including WikiText-103 and Freebase data."""
    
    # Fixed URLs - using working alternatives to broken S3 links
    WIKITEXT_URLS = {
        "wikitext-103": "https://wikitext.smerity.com/wikitext-103-v1.zip",
        "wikitext-103-raw": "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
    }
    
    # Freebase processed graph data URLs  
    FREEBASE_URLS = {
        "max256": "https://docs.google.com/uc?export=download&id=1uuSS2o72dUCJrcLff6NBiLJuTgSU-uRo",
        "max512": "https://docs.google.com/uc?export=download&id=1nOfUq3RUoPEWNZa2QHXl2q-1gA5F6kYh", 
        "max1024": "https://docs.google.com/uc?export=download&id=1uuJwkocJXG1UcQ-RCH3JU96VsDvi7UD2"
    }
    
    def __init__(self, output_dir: str = "/tmp/data"):
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
            downloaded_str = self.humanize_bytes(downloaded)
            total_str = self.humanize_bytes(total_size)
            
            print(f"\r    Progress: [{bar}] {percent:.1f}% ({downloaded_str}/{total_str})", 
                  end='', flush=True)
    
    @staticmethod
    def humanize_bytes(bytes_val: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f}TB"
    
    def download_file(self, url: str, output_path: Path) -> bool:
        """Download a single file with progress tracking."""
        try:
            print(f"\n  📁 Downloading: {output_path.name}")
            print(f"     URL: {url}")
            
            urllib.request.urlretrieve(url, output_path, self.download_progress_hook)
            print()  # New line after progress bar
            
            if output_path.exists() and output_path.stat().st_size > 0:
                file_size = self.humanize_bytes(output_path.stat().st_size)
                print(f"  ✅ Successfully downloaded: {output_path.name} ({file_size})")
                return True
            else:
                print(f"  ❌ Download failed: File is empty or doesn't exist")
                return False
                
        except urllib.error.HTTPError as e:
            print(f"\n  ❌ HTTP Error {e.code}: {e.reason}")
            if e.code == 404:
                print(f"     The file may no longer be available at this URL.")
            return False
        except urllib.error.URLError as e:
            print(f"\n  ❌ URL Error: {e.reason}")
            return False
        except Exception as e:
            print(f"\n  ❌ Unexpected error: {str(e)}")
            return False
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """Extract a ZIP file."""
        try:
            print(f"  📦 Extracting: {zip_path.name}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"  ✅ Successfully extracted: {zip_path.name}")
            return True
        except Exception as e:
            print(f"  ❌ Extraction failed: {str(e)}")
            return False
    
    def extract_tar(self, tar_path: Path, extract_to: Path) -> bool:
        """Extract a TAR file."""
        try:
            print(f"  📦 Extracting: {tar_path.name}")
            with tarfile.open(tar_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
            print(f"  ✅ Successfully extracted: {tar_path.name}")
            return True
        except Exception as e:
            print(f"  ❌ Extraction failed: {str(e)}")
            return False
    
    def download_wikitext(self) -> bool:
        """Download WikiText-103 datasets."""
        print("🚀 Downloading WikiText-103 datasets...")
        
        all_success = True
        
        for dataset_name, url in self.WIKITEXT_URLS.items():
            print(f"\n📥 Processing {dataset_name}...")
            
            target_dir = self.output_dir / dataset_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Download ZIP file
            zip_filename = f"{dataset_name}-v1.zip"
            zip_path = target_dir / zip_filename
            
            success = self.download_file(url, zip_path)
            if not success:
                all_success = False
                continue
            
            # Extract ZIP file
            success = self.extract_zip(zip_path, target_dir)
            if not success:
                all_success = False
                continue
            
            # Move extracted contents to target directory
            extracted_dir = target_dir / dataset_name
            if extracted_dir.exists():
                print(f"  📁 Moving extracted files...")
                for item in extracted_dir.iterdir():
                    item.replace(target_dir / item.name)
                extracted_dir.rmdir()
            
            # Clean up ZIP file
            zip_path.unlink()
            print(f"  🧹 Cleaned up: {zip_filename}")
        
        return all_success
    
    def download_freebase(self) -> bool:
        """Download Freebase graph datasets."""
        print("🚀 Downloading Freebase graph datasets...")
        
        all_success = True
        
        # Create packaged directory for temporary files
        packaged_dir = self.output_dir / "packaged"
        packaged_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download all TAR files
            for version, url in self.FREEBASE_URLS.items():
                print(f"\n📥 Processing Freebase {version}...")
                
                tar_filename = f"{version}.tar"
                tar_path = packaged_dir / tar_filename
                
                success = self.download_file(url, tar_path)
                if not success:
                    all_success = False
                    continue
            
            # Extract all TAR files
            for version in self.FREEBASE_URLS.keys():
                tar_path = packaged_dir / f"{version}.tar"
                if not tar_path.exists():
                    continue
                    
                print(f"\n📦 Extracting Freebase {version}...")
                output_dir = self.output_dir / "freebase" / version
                output_dir.mkdir(parents=True, exist_ok=True)
                
                success = self.extract_tar(tar_path, output_dir)
                if not success:
                    all_success = False
        
        finally:
            # Clean up packaged directory
            if packaged_dir.exists():
                print(f"\n🧹 Cleaning up temporary files...")
                for file in packaged_dir.iterdir():
                    file.unlink()
                packaged_dir.rmdir()
        
        return all_success
    
    def verify_wikitext(self) -> bool:
        """Verify WikiText-103 dataset files."""
        print("🔍 Verifying WikiText-103 datasets...")
        
        required_files = {
            "wikitext-103": ["wiki.train.tokens", "wiki.valid.tokens", "wiki.test.tokens"],
            "wikitext-103-raw": ["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"]
        }
        
        all_present = True
        
        for dataset, files in required_files.items():
            dataset_dir = self.output_dir / dataset
            print(f"\n📁 Checking {dataset}:")
            
            if not dataset_dir.exists():
                print(f"  ❌ Dataset directory missing: {dataset_dir}")
                all_present = False
                continue
            
            for filename in files:
                file_path = dataset_dir / filename
                if file_path.exists() and file_path.stat().st_size > 0:
                    size = self.humanize_bytes(file_path.stat().st_size)
                    print(f"  ✅ {filename}: {size}")
                else:
                    print(f"  ❌ Missing or empty: {filename}")
                    all_present = False
        
        return all_present
    
    def verify_freebase(self) -> bool:
        """Verify Freebase dataset files."""
        print("🔍 Verifying Freebase datasets...")
        
        freebase_dir = self.output_dir / "freebase"
        if not freebase_dir.exists():
            print(f"  ❌ Freebase directory missing: {freebase_dir}")
            return False
        
        all_present = True
        
        for version in self.FREEBASE_URLS.keys():
            version_dir = freebase_dir / version
            print(f"\n📁 Checking freebase/{version}:")
            
            if not version_dir.exists():
                print(f"  ❌ Version directory missing: {version_dir}")
                all_present = False
                continue
            
            # Check for common files
            files = list(version_dir.glob("*.gz"))
            if files:
                total_size = sum(f.stat().st_size for f in files)
                print(f"  ✅ Found {len(files)} files ({self.humanize_bytes(total_size)})")
            else:
                print(f"  ❌ No data files found")
                all_present = False
        
        return all_present


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download WikiGraphs datasets (WikiText-103 + Freebase)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download all datasets to /tmp/data:
    python download_wikigraphs_datasets.py --all

  Download only WikiText-103 to custom directory:
    python download_wikigraphs_datasets.py --wikitext --output_dir ./data

  Download only Freebase graphs:
    python download_wikigraphs_datasets.py --freebase --output_dir ./data

  Verify downloaded datasets:
    python download_wikigraphs_datasets.py --verify /tmp/data

This tool fixes Issue #575: file not found error '/tmp/data/wikitext-103/wiki.train.tokens'
by using working download URLs instead of broken S3 links.
        """
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Download both WikiText-103 and Freebase datasets"
    )
    
    parser.add_argument(
        "--wikitext",
        action="store_true", 
        help="Download only WikiText-103 datasets"
    )
    
    parser.add_argument(
        "--freebase",
        action="store_true",
        help="Download only Freebase graph datasets"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/tmp/data",
        help="Output directory for downloaded datasets (default: /tmp/data)"
    )
    
    parser.add_argument(
        "--verify",
        type=str,
        help="Verify datasets in the specified directory"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        downloader = WikiGraphsDownloader(args.verify)
        wikitext_ok = downloader.verify_wikitext()
        freebase_ok = downloader.verify_freebase()
        
        if wikitext_ok and freebase_ok:
            print("\n🎉 All datasets verified successfully!")
            return 0
        else:
            print("\n⚠️  Some datasets are missing or incomplete.")
            return 1
    
    downloader = WikiGraphsDownloader(args.output_dir)
    
    success = True
    
    if args.all or args.wikitext:
        success &= downloader.download_wikitext()
    
    if args.all or args.freebase:
        success &= downloader.download_freebase()
    
    if not (args.all or args.wikitext or args.freebase):
        parser.print_help()
        return 1
    
    if success:
        print(f"\n🎉 Download completed successfully!")
        print(f"📁 Files saved to: {downloader.output_dir}")
        
        # Automatically verify downloads
        print(f"\n🔍 Verifying downloads...")
        wikitext_ok = True
        freebase_ok = True
        
        if args.all or args.wikitext:
            wikitext_ok = downloader.verify_wikitext()
        if args.all or args.freebase:
            freebase_ok = downloader.verify_freebase()
        
        if wikitext_ok and freebase_ok:
            print("\n✅ All files verified successfully!")
        
    else:
        print(f"\n⚠️  Some downloads failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
