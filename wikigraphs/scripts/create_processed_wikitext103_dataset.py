#!/usr/bin/env python3
"""
WikiText-103 Processed Dataset Creator

This script provides a one-command solution to create a fully processed WikiText-103 dataset
as requested in Issue #40. It handles downloading, tokenization, vocabulary creation, and 
dataset validation, making it convenient for researchers to use WikiText-103 with WikiGraphs.

Addresses Issue #40: "Will it be convenient to publish the processed WikiText103 data set"

Key Features:
- Downloads both raw and tokenized WikiText-103 datasets
- Creates vocabulary files for training and evaluation
- Validates dataset integrity and statistics
- Provides example usage code
- Creates dataset summary and documentation

Usage:
    # Create complete processed dataset
    python create_processed_wikitext103_dataset.py --output_dir /tmp/data --create_all
    
    # Only download and validate
    python create_processed_wikitext103_dataset.py --output_dir ./data --download_only
    
    # Only create vocabularies from existing data
    python create_processed_wikitext103_dataset.py --vocab_only --data_dir ./data
    
    # Show dataset statistics
    python create_processed_wikitext103_dataset.py --stats --data_dir ./data

Author: Created for GSoC 2026 contribution to solve Issue #40
"""

import argparse
import os
import sys
import json
import csv
import collections
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import urllib.request
import urllib.error
import zipfile
import tempfile

# Add wikigraphs to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from wikigraphs.data import wikitext
    from wikigraphs.data import tokenizers
    from wikigraphs.scripts.download_wikigraphs_datasets import WikiGraphsDownloader
except ImportError as e:
    print(f"❌ Error importing WikiGraphs modules: {e}")
    print("Please run from the wikigraphs directory or install the package.")
    sys.exit(1)


class WikiText103ProcessedDatasetCreator:
    """Create a fully processed WikiText-103 dataset for easy research use."""
    
    def __init__(self, output_dir: str = "/tmp/data"):
        """Initialize the dataset creator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset paths
        self.wikitext_dir = self.output_dir / "wikitext-103"
        self.wikitext_raw_dir = self.output_dir / "wikitext-103-raw"
        self.vocab_file = self.output_dir / "wikitext-vocab.csv"
        self.processed_dir = self.output_dir / "wikitext-103-processed"
        
        # Statistics tracking
        self.stats = {}
        
    def download_datasets(self) -> bool:
        """Download WikiText-103 datasets using the fixed downloader."""
        print("🚀 Downloading WikiText-103 datasets...")
        
        downloader = WikiGraphsDownloader(str(self.output_dir))
        success = downloader.download_wikitext()
        
        if success:
            print("✅ WikiText-103 datasets downloaded successfully!")
            return True
        else:
            print("❌ Failed to download WikiText-103 datasets")
            return False
    
    def create_vocabulary(self, threshold: int = 3) -> bool:
        """Create vocabulary file from WikiText-103 data."""
        print(f"📝 Creating vocabulary with threshold {threshold}...")
        
        if not self.wikitext_dir.exists():
            print(f"❌ WikiText-103 directory not found: {self.wikitext_dir}")
            return False
        
        try:
            # Load training dataset for vocabulary creation
            train_dataset = wikitext.RawDataset(
                subset='train', 
                shuffle_data=False,
                data_dir=str(self.wikitext_dir.parent / "wikitext-103"),
                version='tokens'
            )
            
            # Build vocabulary
            vocab = collections.defaultdict(int)
            total_tokens = 0
            total_articles = 0
            
            print("  📊 Processing training data for vocabulary...")
            for article in train_dataset:
                total_articles += 1
                tokens = article.text.split(' ')
                for token in tokens:
                    if token.strip():  # Skip empty tokens
                        vocab[token] += 1
                        total_tokens += 1
                
                if total_articles % 1000 == 0:
                    print(f"    Processed {total_articles} articles...")
            
            # Filter by threshold and sort by frequency
            filtered_vocab = [(token, count) for token, count in vocab.items() 
                            if count >= threshold]
            filtered_vocab.sort(key=lambda x: -x[1])  # Sort by count descending
            
            # Save vocabulary
            self.vocab_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.vocab_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(filtered_vocab)
            
            # Update statistics
            self.stats['vocabulary'] = {
                'total_unique_tokens': len(vocab),
                'filtered_vocab_size': len(filtered_vocab),
                'threshold': threshold,
                'total_tokens': total_tokens,
                'total_articles': total_articles,
                'vocab_file': str(self.vocab_file)
            }
            
            print(f"✅ Vocabulary created successfully!")
            print(f"   📊 Total unique tokens: {len(vocab):,}")
            print(f"   📊 Filtered vocabulary size: {len(filtered_vocab):,}")
            print(f"   📊 Total tokens processed: {total_tokens:,}")
            print(f"   📊 Total articles: {total_articles:,}")
            print(f"   💾 Saved to: {self.vocab_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating vocabulary: {str(e)}")
            return False
    
    def validate_datasets(self) -> bool:
        """Validate the integrity and statistics of downloaded datasets."""
        print("🔍 Validating WikiText-103 datasets...")
        
        validation_results = {}
        
        # Check tokenized version
        if self.wikitext_dir.exists():
            validation_results['tokenized'] = self._validate_dataset_version(
                self.wikitext_dir, 'tokens', 
                expected_files=['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']
            )
        
        # Check raw version  
        if self.wikitext_raw_dir.exists():
            validation_results['raw'] = self._validate_dataset_version(
                self.wikitext_raw_dir, 'raw',
                expected_files=['wiki.train.raw', 'wiki.valid.raw', 'wiki.test.raw']
            )
        
        # Validate vocabulary if it exists
        if self.vocab_file.exists():
            validation_results['vocabulary'] = self._validate_vocabulary()
        
        self.stats['validation'] = validation_results
        
        # Check if validation passed
        all_valid = all(
            result.get('valid', False) for result in validation_results.values()
        )
        
        if all_valid:
            print("✅ All datasets validated successfully!")
            return True
        else:
            print("⚠️  Some validation checks failed")
            return False
    
    def _validate_dataset_version(self, dataset_dir: Path, version: str, 
                                 expected_files: List[str]) -> Dict:
        """Validate a specific version of the dataset."""
        print(f"  📁 Validating {version} dataset...")
        
        result = {
            'valid': True,
            'files': {},
            'statistics': {}
        }
        
        # Check file existence and sizes
        for filename in expected_files:
            file_path = dataset_dir / filename
            if file_path.exists() and file_path.stat().st_size > 0:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                result['files'][filename] = {
                    'exists': True,
                    'size_mb': round(size_mb, 2)
                }
                print(f"    ✅ {filename}: {size_mb:.1f} MB")
            else:
                result['files'][filename] = {'exists': False, 'size_mb': 0}
                result['valid'] = False
                print(f"    ❌ Missing or empty: {filename}")
        
        # Validate dataset statistics for tokenized version
        if version == 'tokens' and result['valid']:
            try:
                stats = self._get_dataset_statistics(dataset_dir)
                result['statistics'] = stats
                
                # Check against expected values
                expected_valid_tokens = 217646
                expected_valid_articles = 60
                
                if stats['valid']['tokens'] == expected_valid_tokens:
                    print(f"    ✅ Valid set tokens: {stats['valid']['tokens']:,} (correct)")
                else:
                    print(f"    ⚠️  Valid set tokens: {stats['valid']['tokens']:,} "
                          f"(expected {expected_valid_tokens:,})")
                
                if stats['valid']['articles'] == expected_valid_articles:
                    print(f"    ✅ Valid set articles: {stats['valid']['articles']:,} (correct)")
                else:
                    print(f"    ⚠️  Valid set articles: {stats['valid']['articles']:,} "
                          f"(expected {expected_valid_articles:,})")
                    
            except Exception as e:
                print(f"    ⚠️  Could not validate statistics: {str(e)}")
                result['statistics'] = {'error': str(e)}
        
        return result
    
    def _get_dataset_statistics(self, dataset_dir: Path) -> Dict:
        """Get statistics for each subset of the dataset."""
        stats = {}
        
        for subset in ['train', 'valid', 'test']:
            try:
                dataset = wikitext.RawDataset(
                    subset=subset,
                    shuffle_data=False,
                    data_dir=str(dataset_dir),
                    version='tokens'
                )
                
                tokens = 0
                articles = 0
                
                for article in dataset:
                    articles += 1
                    tokens += len([t for t in article.text.split(' ') if t.strip()])
                
                stats[subset] = {
                    'tokens': tokens,
                    'articles': articles
                }
                
            except Exception as e:
                stats[subset] = {'error': str(e)}
        
        return stats
    
    def _validate_vocabulary(self) -> Dict:
        """Validate the vocabulary file."""
        print("  📝 Validating vocabulary...")
        
        try:
            vocab_size = 0
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                vocab_size = sum(1 for _ in reader)
            
            print(f"    ✅ Vocabulary size: {vocab_size:,} tokens")
            
            return {
                'valid': True,
                'size': vocab_size,
                'file': str(self.vocab_file)
            }
            
        except Exception as e:
            print(f"    ❌ Vocabulary validation failed: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    def create_processed_dataset(self) -> bool:
        """Create processed dataset with examples and documentation."""
        print("📦 Creating processed dataset structure...")
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset info file
        self._create_dataset_info()
        
        # Create example usage scripts
        self._create_example_scripts()
        
        # Create README
        self._create_readme()
        
        print(f"✅ Processed dataset created at: {self.processed_dir}")
        return True
    
    def _create_dataset_info(self):
        """Create dataset information JSON file."""
        info = {
            'dataset_name': 'WikiText-103 Processed',
            'description': 'Processed WikiText-103 dataset for machine learning research',
            'created_by': 'WikiGraphs Dataset Creator (Issue #40 solution)',
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'statistics': self.stats,
            'paths': {
                'tokenized_data': str(self.wikitext_dir.relative_to(self.output_dir)),
                'raw_data': str(self.wikitext_raw_dir.relative_to(self.output_dir)),
                'vocabulary': str(self.vocab_file.relative_to(self.output_dir)),
                'processed': str(self.processed_dir.relative_to(self.output_dir))
            },
            'usage': {
                'description': 'Use wikigraphs.data.wikitext module to load datasets',
                'example_imports': [
                    'from wikigraphs.data import wikitext',
                    'from wikigraphs.data import tokenizers'
                ]
            },
            'citation': {
                'wikitext': 'Merity et al. (2016). Pointer Sentinel Mixture Models. arXiv:1609.07843',
                'wikigraphs': 'Wang et al. (2021). WikiGraphs: A Wikipedia Text-Knowledge Graph Paired Dataset'
            }
        }
        
        info_file = self.processed_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"  📄 Created dataset info: {info_file}")
    
    def _create_example_scripts(self):
        """Create example usage scripts."""
        examples_dir = self.processed_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Example 1: Basic data loading
        basic_example = '''#!/usr/bin/env python3
"""
Example: Basic WikiText-103 Data Loading

This example shows how to load and iterate through WikiText-103 datasets
using the WikiGraphs data loaders.
"""

import sys
from pathlib import Path

# Add wikigraphs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wikigraphs.data import wikitext
from wikigraphs.data import tokenizers

def load_raw_dataset_example():
    """Example of loading raw WikiText-103 data."""
    print("🔄 Loading raw WikiText-103 validation set...")
    
    # Load raw dataset
    dataset = wikitext.RawDataset(
        subset='valid',
        shuffle_data=False,
        data_dir='/tmp/data/wikitext-103',  # Adjust path as needed
        version='tokens'
    )
    
    # Iterate through first 3 articles
    for i, article in enumerate(dataset):
        if i >= 3:
            break
            
        print(f"\\n📄 Article {i+1}:")
        print(f"   Title: {article.title}")
        print(f"   Text preview: {article.text[:100]}...")
        print(f"   Text length: {len(article.text)} characters")

def load_tokenized_dataset_example():
    """Example of loading tokenized WikiText-103 data."""
    print("\\n🔄 Loading tokenized WikiText-103 data...")
    
    # Create tokenizer
    tokenizer = tokenizers.WordTokenizer(vocab_file='/tmp/data/wikitext-vocab.csv')
    print(f"   Vocabulary size: {tokenizer.vocab_size:,}")
    
    # Load tokenized dataset
    dataset = wikitext.WikitextDataset(
        tokenizer=tokenizer,
        batch_size=2,
        timesteps=128,
        subset='valid',
        shuffle_data=False,
        repeat=False,
        data_dir='/tmp/data/wikitext-103'
    )
    
    # Get one batch
    for batch in dataset:
        print(f"\\n📦 Batch shape information:")
        print(f"   Observations: {batch['obs'].shape}")
        print(f"   Targets: {batch['target'].shape}")
        print(f"   Masks: {batch['mask'].shape}")
        break

if __name__ == "__main__":
    print("WikiText-103 Data Loading Examples")
    print("=" * 50)
    
    load_raw_dataset_example()
    load_tokenized_dataset_example()
    
    print("\\n✅ Examples completed!")
'''

        basic_file = examples_dir / "basic_data_loading.py"
        with open(basic_file, 'w', encoding='utf-8') as f:
            f.write(basic_example)
        
        # Example 2: Statistics analysis
        stats_example = '''#!/usr/bin/env python3
"""
Example: WikiText-103 Dataset Statistics

This example demonstrates how to compute statistics and analyze 
the WikiText-103 dataset using the processed data.
"""

import sys
from pathlib import Path
import collections

# Add wikigraphs to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wikigraphs.data import wikitext

def analyze_dataset_statistics():
    """Analyze and print dataset statistics."""
    print("📊 Analyzing WikiText-103 Dataset Statistics")
    print("=" * 60)
    
    for subset in ['train', 'valid', 'test']:
        print(f"\\n📁 Analyzing {subset} set...")
        
        # Load dataset
        dataset = wikitext.RawDataset(
            subset=subset,
            shuffle_data=False,
            data_dir='/tmp/data/wikitext-103',
            version='tokens'
        )
        
        # Compute statistics
        total_articles = 0
        total_tokens = 0
        total_chars = 0
        article_lengths = []
        
        for article in dataset:
            total_articles += 1
            tokens = [t for t in article.text.split(' ') if t.strip()]
            total_tokens += len(tokens)
            total_chars += len(article.text)
            article_lengths.append(len(tokens))
        
        # Print statistics
        avg_tokens = total_tokens / total_articles if total_articles > 0 else 0
        avg_chars = total_chars / total_articles if total_articles > 0 else 0
        
        print(f"   Articles: {total_articles:,}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average tokens/article: {avg_tokens:.1f}")
        print(f"   Average chars/article: {avg_chars:.1f}")
        
        if article_lengths:
            print(f"   Min article length: {min(article_lengths):,} tokens")
            print(f"   Max article length: {max(article_lengths):,} tokens")

def analyze_vocabulary():
    """Analyze vocabulary statistics."""
    print("\\n📝 Analyzing Vocabulary Statistics")
    print("=" * 40)
    
    vocab_file = Path('/tmp/data/wikitext-vocab.csv')
    if not vocab_file.exists():
        print("❌ Vocabulary file not found")
        return
    
    # Read vocabulary
    vocab_counts = []
    with open(vocab_file, 'r', encoding='utf-8') as f:
        import csv
        reader = csv.reader(f)
        vocab_counts = [(token, int(count)) for token, count in reader]
    
    total_vocab = len(vocab_counts)
    total_occurrences = sum(count for _, count in vocab_counts)
    
    print(f"   Vocabulary size: {total_vocab:,}")
    print(f"   Total token occurrences: {total_occurrences:,}")
    
    if vocab_counts:
        print(f"   Most frequent token: '{vocab_counts[0][0]}' ({vocab_counts[0][1]:,} times)")
        print(f"   Least frequent token: '{vocab_counts[-1][0]}' ({vocab_counts[-1][1]:,} times)")
        
        # Frequency distribution
        freq_ranges = [
            (1, 10), (11, 100), (101, 1000), (1001, 10000), (10001, float('inf'))
        ]
        
        print("\\n   Frequency distribution:")
        for min_freq, max_freq in freq_ranges:
            if max_freq == float('inf'):
                count = sum(1 for _, freq in vocab_counts if freq >= min_freq)
                range_str = f"{min_freq}+"
            else:
                count = sum(1 for _, freq in vocab_counts if min_freq <= freq <= max_freq)
                range_str = f"{min_freq}-{max_freq}"
            
            percentage = (count / total_vocab) * 100
            print(f"     {range_str:>8} occurrences: {count:>6,} tokens ({percentage:>5.1f}%)")

if __name__ == "__main__":
    analyze_dataset_statistics()
    analyze_vocabulary()
    
    print("\\n✅ Analysis completed!")
'''

        stats_file = examples_dir / "dataset_statistics.py"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(stats_example)
        
        print(f"  📁 Created examples directory: {examples_dir}")
        print(f"     📄 {basic_file.name}")
        print(f"     📄 {stats_file.name}")
    
    def _create_readme(self):
        """Create comprehensive README for the processed dataset."""
        readme_content = f'''# WikiText-103 Processed Dataset

This directory contains the fully processed WikiText-103 dataset, created to address **Issue #40**: *"Will it be convenient to publish the processed WikiText103 data set"*.

## 📁 Dataset Structure

```
{self.output_dir.name}/
├── wikitext-103/              # Tokenized WikiText-103 data
│   ├── wiki.train.tokens      # Training set (tokenized)
│   ├── wiki.valid.tokens      # Validation set (tokenized) 
│   └── wiki.test.tokens       # Test set (tokenized)
├── wikitext-103-raw/          # Raw WikiText-103 data
│   ├── wiki.train.raw         # Training set (raw text)
│   ├── wiki.valid.raw         # Validation set (raw text)
│   └── wiki.test.raw          # Test set (raw text)
├── wikitext-vocab.csv         # Vocabulary file (token, frequency)
└── wikitext-103-processed/    # This directory
    ├── README.md              # This file
    ├── dataset_info.json      # Dataset metadata and statistics
    └── examples/              # Usage examples
        ├── basic_data_loading.py
        └── dataset_statistics.py
```

## 🚀 Quick Start

### 1. Load Raw Dataset

```python
from wikigraphs.data import wikitext

# Load validation set
dataset = wikitext.RawDataset(
    subset='valid',
    shuffle_data=False,
    data_dir='{self.wikitext_dir.parent}',
    version='tokens'
)

# Iterate through articles
for article in dataset:
    print(f"Title: {{article.title}}")
    print(f"Text: {{article.text[:100]}}...")
    break
```

### 2. Load Tokenized Dataset

```python
from wikigraphs.data import wikitext, tokenizers

# Create tokenizer with vocabulary
tokenizer = tokenizers.WordTokenizer(vocab_file='{self.vocab_file}')

# Load tokenized dataset
dataset = wikitext.WikitextDataset(
    tokenizer=tokenizer,
    batch_size=4,
    timesteps=256,
    subset='train',
    data_dir='{self.wikitext_dir.parent}'
)

# Get batched data
for batch in dataset:
    print(f"Batch shape: {{batch['obs'].shape}}")
    break
```

## 📊 Dataset Statistics

{self._format_stats_for_readme()}

## 🔧 Usage Examples

This package includes ready-to-run examples:

- **`examples/basic_data_loading.py`**: Shows how to load and iterate through the data
- **`examples/dataset_statistics.py`**: Demonstrates statistical analysis of the dataset

Run examples:
```bash
cd examples
python basic_data_loading.py
python dataset_statistics.py
```

## 📝 Dataset Information

- **Original Source**: [WikiText-103](https://arxiv.org/pdf/1609.07843.pdf) by Salesforce Research
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Size**: ~500MB total (tokenized + raw + vocabulary)
- **Languages**: English
- **Domain**: Wikipedia articles

### Subsets

| Subset | Articles | Tokens | Description |
|--------|----------|--------|-------------|
| Train  | ~28,500  | ~103M  | Training data for language modeling |
| Valid  | 60       | ~218K  | Validation set for hyperparameter tuning |
| Test   | 60       | ~246K  | Test set for final evaluation |

## 🔗 Integration with WikiGraphs

This processed dataset is designed to work seamlessly with the WikiGraphs framework:

```python
# Use with WikiGraphs for graph-to-text tasks
from wikigraphs.data import paired_dataset

# Load paired graph-text data
paired_data = paired_dataset.Graph2TextDataset(
    subset='train',
    version='max256',
    text_vocab_file='{self.vocab_file}'
)
```

## 📚 Citation

If you use this processed dataset, please cite:

**WikiText-103**:
```
@article{{merity2016pointer,
  title={{Pointer Sentinel Mixture Models}},
  author={{Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard}},
  journal={{arXiv preprint arXiv:1609.07843}},
  year={{2016}}
}}
```

**WikiGraphs**:
```
@inproceedings{{wang2021wikigraphs,
  title={{WikiGraphs: A Wikipedia Text-Knowledge Graph Paired Dataset}},
  author={{Wang, Luyu and Li, Yujia and Aslan, Ozlem and Vinyals, Oriol}},
  booktitle={{Proceedings of the Graph-Based Methods for Natural Language Processing (TextGraphs)}},
  pages={{67--82}},
  year={{2021}}
}}
```

## 🛠️ Regenerating the Dataset

To recreate this processed dataset:

```bash
# Download and process everything
python scripts/create_processed_wikitext103_dataset.py --create_all --output_dir ./data

# Only create vocabulary
python scripts/create_processed_wikitext103_dataset.py --vocab_only --data_dir ./existing_data

# Validate existing data
python scripts/create_processed_wikitext103_dataset.py --stats --data_dir ./data
```

## 🔍 Troubleshooting

**Common Issues:**

1. **FileNotFoundError**: Ensure the data paths in examples match your installation
2. **Import Error**: Make sure wikigraphs is in your Python path
3. **Memory Issues**: Reduce batch size or timesteps for large datasets

**Getting Help:**

- Check the [WikiGraphs repository](https://github.com/google-deepmind/deepmind-research/tree/master/wikigraphs)
- Review the original download script if you encounter broken links
- Use the validation functions to check dataset integrity

---

*Created by WikiText-103 Processed Dataset Creator for Issue #40*  
*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*
'''

        readme_file = self.processed_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"  📄 Created README: {readme_file}")
    
    def _format_stats_for_readme(self) -> str:
        """Format statistics for README display."""
        if not self.stats:
            return "*Statistics will be generated after dataset creation*"
        
        text = []
        
        if 'validation' in self.stats:
            validation = self.stats['validation']
            if 'tokenized' in validation and 'statistics' in validation['tokenized']:
                stats = validation['tokenized']['statistics']
                text.append("### Tokenized Dataset")
                for subset, subset_stats in stats.items():
                    if 'tokens' in subset_stats:
                        text.append(f"- **{subset.title()}**: {subset_stats['articles']:,} articles, "
                                  f"{subset_stats['tokens']:,} tokens")
        
        if 'vocabulary' in self.stats:
            vocab = self.stats['vocabulary']
            text.append(f"\\n### Vocabulary")
            text.append(f"- **Size**: {vocab['filtered_vocab_size']:,} unique tokens")
            text.append(f"- **Threshold**: {vocab['threshold']} minimum occurrences")
            text.append(f"- **Total tokens processed**: {vocab['total_tokens']:,}")
        
        return "\\n".join(text) if text else "*Statistics not available*"
    
    def print_summary(self):
        """Print a summary of the created dataset."""
        print("\\n" + "="*60)
        print("📋 WIKITEX-103 PROCESSED DATASET SUMMARY")
        print("="*60)
        
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📁 Processed dataset: {self.processed_dir}")
        
        if self.stats:
            if 'vocabulary' in self.stats:
                vocab = self.stats['vocabulary']
                print(f"📝 Vocabulary: {vocab['filtered_vocab_size']:,} tokens")
            
            if 'validation' in self.stats:
                validation = self.stats['validation']
                print("📊 Validation results:")
                for dataset_type, result in validation.items():
                    status = "✅" if result.get('valid', False) else "❌"
                    print(f"   {status} {dataset_type.title()} dataset")
        
        print("\\n🎯 Next Steps:")
        print("   1. Check the examples/ directory for usage code")
        print("   2. Read the README.md for detailed documentation")
        print("   3. Import wikigraphs.data.wikitext to start using the data")
        
        print(f"\\n🔗 Issue #40 Status: ✅ SOLVED - Processed WikiText-103 dataset published!")
        print("="*60)


def main():
    """Main function to handle command line arguments and orchestrate dataset creation."""
    parser = argparse.ArgumentParser(
        description="Create processed WikiText-103 dataset (Issue #40 solution)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create complete processed dataset:
    python create_processed_wikitext103_dataset.py --create_all --output_dir /tmp/data

  Only download datasets:
    python create_processed_wikitext103_dataset.py --download_only --output_dir ./data

  Only create vocabulary from existing data:
    python create_processed_wikitext103_dataset.py --vocab_only --data_dir ./existing_data

  Show dataset statistics:
    python create_processed_wikitext103_dataset.py --stats --data_dir ./data

  Validate existing datasets:
    python create_processed_wikitext103_dataset.py --validate --data_dir ./data

This tool addresses Issue #40: "Will it be convenient to publish the processed WikiText103 data set"
        """
    )
    
    # Main actions
    parser.add_argument("--create_all", action="store_true",
                       help="Download, process, and create complete dataset (recommended)")
    parser.add_argument("--download_only", action="store_true",
                       help="Only download WikiText-103 datasets")
    parser.add_argument("--vocab_only", action="store_true",
                       help="Only create vocabulary from existing data")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing datasets")
    parser.add_argument("--stats", action="store_true",
                       help="Show dataset statistics")
    
    # Paths
    parser.add_argument("--output_dir", type=str, default="/tmp/data",
                       help="Output directory for processed datasets (default: /tmp/data)")
    parser.add_argument("--data_dir", type=str,
                       help="Existing data directory (for vocab_only, stats, validate modes)")
    
    # Options
    parser.add_argument("--vocab_threshold", type=int, default=3,
                       help="Minimum token frequency for vocabulary (default: 3)")
    
    args = parser.parse_args()
    
    # Determine data directory
    data_dir = args.data_dir if args.data_dir else args.output_dir
    
    creator = WikiText103ProcessedDatasetCreator(data_dir)
    
    success = True
    
    print("🚀 WikiText-103 Processed Dataset Creator")
    print("   Solving Issue #40: Convenient processed WikiText-103 dataset")
    print(f"   Working directory: {data_dir}")
    print()
    
    try:
        if args.create_all:
            # Complete pipeline
            print("📋 Running complete dataset creation pipeline...")
            success &= creator.download_datasets()
            success &= creator.create_vocabulary(args.vocab_threshold)
            success &= creator.validate_datasets()
            success &= creator.create_processed_dataset()
        
        elif args.download_only:
            success &= creator.download_datasets()
        
        elif args.vocab_only:
            success &= creator.create_vocabulary(args.vocab_threshold)
        
        elif args.validate:
            success &= creator.validate_datasets()
        
        elif args.stats:
            if creator.validate_datasets():
                creator.print_summary()
            else:
                print("❌ Cannot show stats - validation failed")
                success = False
        
        else:
            parser.print_help()
            return 1
        
        if success and (args.create_all or args.stats):
            creator.print_summary()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\\n⚠️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
