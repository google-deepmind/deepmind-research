#!/usr/bin/env python3
"""
WikiText-103 Processed Dataset Quick Setup

A lightweight, dependency-minimal solution for Issue #40: 
"Will it be convenient to publish the processed WikiText103 data set"

This script downloads and organizes WikiText-103 data without requiring
heavy ML dependencies, making it accessible for quick dataset setup.

Usage:
    python setup_wikitext103_dataset.py
    python setup_wikitext103_dataset.py --output_dir ./data
    python setup_wikitext103_dataset.py --validate_only ./data
"""

import argparse
import os
import sys
import json
import csv
import collections
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SimpleWikiText103Setup:
    """Lightweight WikiText-103 dataset setup without heavy dependencies."""
    
    # Working URLs for WikiText-103 data
    WIKITEXT_URLS = {
        "wikitext-103": "https://wikitext.smerity.com/wikitext-103-v1.zip",
        "wikitext-103-raw": "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
    }
    
    def __init__(self, output_dir: str = "/tmp/data"):
        """Initialize the setup tool."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset paths
        self.wikitext_dir = self.output_dir / "wikitext-103"
        self.wikitext_raw_dir = self.output_dir / "wikitext-103-raw"
        self.vocab_file = self.output_dir / "wikitext-vocab.csv"
        self.info_file = self.output_dir / "dataset_info.json"
        
    def download_progress_hook(self, block_num: int, block_size: int, total_size: int):
        """Simple progress callback."""
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100.0, (downloaded / total_size) * 100.0)
            bar_length = 30
            filled_length = int(bar_length * percent // 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            
            print(f"\r    [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", 
                  end='', flush=True)
    
    def download_file(self, url: str, output_path: Path) -> bool:
        """Download a file with progress tracking."""
        try:
            print(f"📥 Downloading: {output_path.name}")
            urllib.request.urlretrieve(url, output_path, self.download_progress_hook)
            print()  # New line after progress bar
            
            if output_path.exists() and output_path.stat().st_size > 0:
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✅ Downloaded: {output_path.name} ({size_mb:.1f} MB)")
                return True
            else:
                print(f"❌ Download failed: File is empty")
                return False
                
        except Exception as e:
            print(f"❌ Download error: {str(e)}")
            return False
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """Extract a ZIP file."""
        try:
            print(f"📦 Extracting: {zip_path.name}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"✅ Extracted: {zip_path.name}")
            return True
        except Exception as e:
            print(f"❌ Extraction failed: {str(e)}")
            return False
    
    def download_datasets(self) -> bool:
        """Download WikiText-103 datasets."""
        print("🚀 Setting up WikiText-103 datasets...")
        
        all_success = True
        
        for dataset_name, url in self.WIKITEXT_URLS.items():
            print(f"\n📁 Processing {dataset_name}...")
            
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
            
            # Move extracted contents
            extracted_dir = target_dir / dataset_name
            if extracted_dir.exists():
                print(f"📁 Organizing files...")
                for item in extracted_dir.iterdir():
                    item.replace(target_dir / item.name)
                extracted_dir.rmdir()
            
            # Clean up ZIP file
            zip_path.unlink()
            print(f"🧹 Cleaned up: {zip_filename}")
        
        return all_success
    
    def create_simple_vocabulary(self, threshold: int = 3) -> bool:
        """Create vocabulary from tokenized data."""
        print(f"\n📝 Creating vocabulary (threshold={threshold})...")
        
        train_file = self.wikitext_dir / "wiki.train.tokens"
        if not train_file.exists():
            print(f"❌ Training file not found: {train_file}")
            return False
        
        try:
            # Count tokens
            vocab = collections.defaultdict(int)
            total_tokens = 0
            total_articles = 0
            
            print("📊 Processing training data...")
            with open(train_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple article splitting (articles separated by = Title = pattern)
            import re
            title_pattern = re.compile(r'\n = ([^=].*) = \n')
            articles = title_pattern.split(content)[1:]  # Skip first empty part
            
            for i in range(0, len(articles), 2):  # Every other element is article content
                if i + 1 < len(articles):
                    article_text = articles[i + 1]
                    total_articles += 1
                    
                    tokens = article_text.split()
                    for token in tokens:
                        if token.strip():
                            vocab[token] += 1
                            total_tokens += 1
                    
                    if total_articles % 1000 == 0:
                        print(f"    Processed {total_articles:,} articles...")
            
            # Filter and sort vocabulary
            filtered_vocab = [(token, count) for token, count in vocab.items() 
                            if count >= threshold]
            filtered_vocab.sort(key=lambda x: -x[1])
            
            # Save vocabulary
            with open(self.vocab_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(filtered_vocab)
            
            print(f"✅ Vocabulary created:")
            print(f"   📊 Unique tokens: {len(vocab):,}")
            print(f"   📊 Filtered size: {len(filtered_vocab):,}")
            print(f"   📊 Total tokens: {total_tokens:,}")
            print(f"   📊 Articles: {total_articles:,}")
            print(f"   💾 Saved to: {self.vocab_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Vocabulary creation failed: {str(e)}")
            return False
    
    def validate_setup(self) -> bool:
        """Validate the dataset setup."""
        print("\n🔍 Validating dataset setup...")
        
        all_valid = True
        validation_results = {}
        
        # Check tokenized files
        tokenized_files = ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']
        validation_results['tokenized'] = self._check_files(self.wikitext_dir, tokenized_files)
        
        # Check raw files
        raw_files = ['wiki.train.raw', 'wiki.valid.raw', 'wiki.test.raw']
        validation_results['raw'] = self._check_files(self.wikitext_raw_dir, raw_files)
        
        # Check vocabulary
        if self.vocab_file.exists():
            vocab_size = self._count_lines(self.vocab_file)
            validation_results['vocabulary'] = {
                'exists': True,
                'size': vocab_size
            }
            print(f"✅ Vocabulary: {vocab_size:,} tokens")
        else:
            validation_results['vocabulary'] = {'exists': False}
            print(f"❌ Vocabulary file missing")
            all_valid = False
        
        # Quick statistics
        if validation_results['tokenized']['all_exist']:
            stats = self._get_quick_stats()
            validation_results['statistics'] = stats
            
            print(f"\n📊 Quick Statistics:")
            for subset, stat in stats.items():
                print(f"   {subset.title()}: {stat['size_mb']:.1f} MB")
        
        return all_valid
    
    def _check_files(self, directory: Path, filenames: List[str]) -> Dict:
        """Check if files exist and get their sizes."""
        result = {'all_exist': True, 'files': {}}
        
        print(f"📁 Checking {directory.name}:")
        for filename in filenames:
            file_path = directory / filename
            if file_path.exists() and file_path.stat().st_size > 0:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                result['files'][filename] = {'exists': True, 'size_mb': round(size_mb, 2)}
                print(f"   ✅ {filename}: {size_mb:.1f} MB")
            else:
                result['files'][filename] = {'exists': False, 'size_mb': 0}
                result['all_exist'] = False
                print(f"   ❌ Missing: {filename}")
        
        return result
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def _get_quick_stats(self) -> Dict:
        """Get quick statistics for each subset."""
        stats = {}
        
        for subset in ['train', 'valid', 'test']:
            file_path = self.wikitext_dir / f"wiki.{subset}.tokens"
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                stats[subset] = {'size_mb': round(size_mb, 2)}
        
        return stats
    
    def create_info_file(self):
        """Create dataset information file."""
        info = {
            'dataset_name': 'WikiText-103 Processed Dataset',
            'description': 'Processed WikiText-103 dataset for Issue #40',
            'created_by': 'WikiText-103 Setup Tool',
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'paths': {
                'tokenized_data': str(self.wikitext_dir.relative_to(self.output_dir)),
                'raw_data': str(self.wikitext_raw_dir.relative_to(self.output_dir)),
                'vocabulary': str(self.vocab_file.relative_to(self.output_dir)) if self.vocab_file.exists() else None
            },
            'usage': {
                'description': 'Use with WikiGraphs or any NLP framework',
                'tokenized_files': ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens'],
                'raw_files': ['wiki.train.raw', 'wiki.valid.raw', 'wiki.test.raw'],
                'vocabulary_format': 'CSV with (token, frequency) pairs'
            },
            'citation': {
                'paper': 'Merity et al. (2016). Pointer Sentinel Mixture Models. arXiv:1609.07843',
                'source': 'https://wikitext.smerity.com/'
            }
        }
        
        with open(self.info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        
        print(f"📄 Created info file: {self.info_file}")
    
    def create_simple_examples(self):
        """Create simple usage examples."""
        examples_dir = self.output_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Simple data loading example
        example_code = f'''#!/usr/bin/env python3
"""
Simple WikiText-103 Data Loading Example

This example shows basic loading and iteration without heavy dependencies.
"""

import csv
from pathlib import Path

def load_vocabulary():
    """Load vocabulary from CSV file."""
    vocab = {{}}
    vocab_file = Path('{self.vocab_file}')
    
    if vocab_file.exists():
        with open(vocab_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, (token, freq) in enumerate(reader):
                vocab[token] = {{'id': i, 'freq': int(freq)}}
    
    return vocab

def load_articles(subset='valid'):
    """Load articles from WikiText-103 subset."""
    data_file = Path('{self.wikitext_dir}') / f'wiki.{{subset}}.tokens'
    
    if not data_file.exists():
        print(f"File not found: {{data_file}}")
        return []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split articles by title pattern
    import re
    title_pattern = re.compile(r'\\n = ([^=].*) = \\n')
    parts = title_pattern.split(content)
    
    articles = []
    for i in range(1, len(parts), 2):  # Get titles and content
        if i + 1 < len(parts):
            title = parts[i].strip()
            text = parts[i + 1].strip()
            articles.append({{'title': title, 'text': text}})
    
    return articles

def main():
    """Example usage."""
    print("WikiText-103 Simple Loading Example")
    print("=" * 40)
    
    # Load vocabulary
    vocab = load_vocabulary()
    print(f"Vocabulary size: {{len(vocab):,}} tokens")
    
    # Load validation articles
    articles = load_articles('valid')
    print(f"Validation articles: {{len(articles):,}}")
    
    # Show first article
    if articles:
        article = articles[0]
        print(f"\\nFirst article:")
        print(f"  Title: {{article['title']}}")
        print(f"  Text preview: {{article['text'][:100]}}...")
        print(f"  Text length: {{len(article['text']):,}} characters")

if __name__ == "__main__":
    main()
'''
        
        example_file = examples_dir / "simple_loading.py"
        with open(example_file, 'w', encoding='utf-8') as f:
            f.write(example_code)
        
        print(f"📁 Created examples: {examples_dir}")
    
    def print_summary(self):
        """Print setup summary."""
        print("\n" + "="*50)
        print("📋 WIKITEX-103 DATASET SETUP COMPLETE")
        print("="*50)
        
        print(f"📁 Dataset location: {self.output_dir}")
        
        # Quick overview
        datasets = []
        if self.wikitext_dir.exists():
            datasets.append("✅ Tokenized WikiText-103")
        if self.wikitext_raw_dir.exists():
            datasets.append("✅ Raw WikiText-103")
        if self.vocab_file.exists():
            datasets.append("✅ Vocabulary file")
        
        print("📦 Available datasets:")
        for dataset in datasets:
            print(f"   {dataset}")
        
        print("\n🎯 Usage:")
        print("   1. Use tokenized files for NLP model training")
        print("   2. Use raw files for text analysis")
        print("   3. Use vocabulary for tokenization")
        
        print(f"\n🔗 Issue #40: ✅ SOLVED - Processed WikiText-103 ready!")
        print("="*50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Simple WikiText-103 dataset setup (Issue #40 solution)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Setup complete dataset (recommended):
    python setup_wikitext103_dataset.py

  Custom output directory:
    python setup_wikitext103_dataset.py --output_dir ./my_data

  Only validate existing data:
    python setup_wikitext103_dataset.py --validate_only ./existing_data

  Create vocabulary only:
    python setup_wikitext103_dataset.py --vocab_only --data_dir ./data

This addresses Issue #40: "Will it be convenient to publish the processed WikiText103 data set"
        """
    )
    
    parser.add_argument("--output_dir", type=str, default="/tmp/data",
                       help="Output directory (default: /tmp/data)")
    parser.add_argument("--validate_only", type=str,
                       help="Only validate existing dataset in specified directory")
    parser.add_argument("--vocab_only", action="store_true",
                       help="Only create vocabulary")
    parser.add_argument("--data_dir", type=str,
                       help="Data directory for vocab_only mode")
    parser.add_argument("--vocab_threshold", type=int, default=3,
                       help="Vocabulary frequency threshold (default: 3)")
    
    args = parser.parse_args()
    
    if args.validate_only:
        setup = SimpleWikiText103Setup(args.validate_only)
        success = setup.validate_setup()
        return 0 if success else 1
    
    if args.vocab_only:
        data_dir = args.data_dir if args.data_dir else args.output_dir
        setup = SimpleWikiText103Setup(data_dir)
        success = setup.create_simple_vocabulary(args.vocab_threshold)
        return 0 if success else 1
    
    # Full setup
    setup = SimpleWikiText103Setup(args.output_dir)
    
    print("🚀 WikiText-103 Dataset Setup")
    print("   Issue #40: Convenient processed WikiText-103 dataset")
    print(f"   Output directory: {args.output_dir}")
    print()
    
    try:
        # Download datasets
        success = setup.download_datasets()
        if not success:
            print("❌ Download failed")
            return 1
        
        # Create vocabulary
        success = setup.create_simple_vocabulary(args.vocab_threshold)
        if not success:
            print("⚠️  Vocabulary creation failed, but datasets are available")
        
        # Validate
        setup.validate_setup()
        
        # Create info and examples
        setup.create_info_file()
        setup.create_simple_examples()
        
        # Summary
        setup.print_summary()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
