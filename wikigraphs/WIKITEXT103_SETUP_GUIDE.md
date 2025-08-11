# WikiText-103 Processed Dataset Setup Guide

**Solving Issue #40**: *"Will it be convenient to publish the processed WikiText103 data set"*

This guide provides **two solutions** for setting up processed WikiText-103 datasets with different dependency requirements.

## 🚀 Quick Start (Recommended)

### Option 1: Lightweight Setup (No Dependencies)

```bash
# Complete setup - downloads, processes, validates everything
python scripts/setup_wikitext103_dataset.py

# Custom output directory
python scripts/setup_wikitext103_dataset.py --output_dir ./my_data

# Only validate existing data
python scripts/setup_wikitext103_dataset.py --validate_only ./existing_data
```

**Requirements**: Only Python 3.6+ (no additional packages needed)

### Option 2: Full-Featured Setup (Requires WikiGraphs)

```bash
# Complete automated pipeline
python scripts/create_processed_wikitext103_dataset.py --create_all

# Custom options
python scripts/create_processed_wikitext103_dataset.py --create_all --output_dir ./data --vocab_threshold 5
```

**Requirements**: WikiGraphs dependencies (JAX, Haiku, etc.)

## 📋 What Gets Created

Both solutions create the same dataset structure:

```
/tmp/data/  (or your chosen directory)
├── wikitext-103/              # Tokenized WikiText-103
│   ├── wiki.train.tokens      # ~500MB, 28K+ articles, 103M+ tokens
│   ├── wiki.valid.tokens      # ~1MB, 60 articles, 218K tokens
│   └── wiki.test.tokens       # ~1MB, 60 articles, 246K tokens
├── wikitext-103-raw/          # Raw text WikiText-103
│   ├── wiki.train.raw         # ~500MB, raw text format
│   ├── wiki.valid.raw         # ~1MB, raw text format  
│   └── wiki.test.raw          # ~1MB, raw text format
├── wikitext-vocab.csv         # Vocabulary: 267K+ tokens (token,frequency)
├── dataset_info.json         # Dataset metadata and statistics
└── examples/                  # Usage examples
    ├── simple_loading.py      # Basic data loading
    └── dataset_statistics.py  # Statistical analysis (full version only)
```

## 🎯 Choose Your Solution

### Use **Lightweight Setup** (`setup_wikitext103_dataset.py`) if:
- ✅ You want minimal dependencies  
- ✅ You need quick dataset access
- ✅ You're not using WikiGraphs framework
- ✅ You want to integrate with other NLP libraries

### Use **Full Setup** (`create_processed_wikitext103_dataset.py`) if:
- ✅ You're using WikiGraphs for research
- ✅ You want comprehensive validation and statistics
- ✅ You need advanced preprocessing features
- ✅ You want detailed documentation and examples

## 📊 Dataset Statistics

| Subset | Articles | Tokens | File Size | Description |
|--------|----------|--------|-----------|-------------|
| Train  | ~28,500  | ~103M  | ~500MB    | Training data |
| Valid  | 60       | ~218K  | ~1MB      | Validation set |
| Test   | 60       | ~246K  | ~1MB      | Test set |

**Total**: ~1GB download, ~267K vocabulary tokens

## 🔧 Usage Examples

### Basic Data Loading (Works with both solutions)

```python
import csv
from pathlib import Path

# Load vocabulary
vocab = {}
with open('/tmp/data/wikitext-vocab.csv', 'r') as f:
    reader = csv.reader(f)
    for i, (token, freq) in enumerate(reader):
        vocab[token] = {'id': i, 'freq': int(freq)}

print(f"Vocabulary size: {len(vocab):,}")

# Load articles
with open('/tmp/data/wikitext-103/wiki.valid.tokens', 'r') as f:
    content = f.read()

import re
title_pattern = re.compile(r'\n = ([^=].*) = \n')
parts = title_pattern.split(content)

articles = []
for i in range(1, len(parts), 2):
    if i + 1 < len(parts):
        title = parts[i].strip()
        text = parts[i + 1].strip()
        articles.append({'title': title, 'text': text})

print(f"Articles: {len(articles)}")
print(f"First article: {articles[0]['title']}")
```

### Integration with WikiGraphs (Full setup)

```python
from wikigraphs.data import wikitext, tokenizers

# Create tokenizer with vocabulary
tokenizer = tokenizers.WordTokenizer(vocab_file='/tmp/data/wikitext-vocab.csv')

# Load dataset
dataset = wikitext.WikitextDataset(
    tokenizer=tokenizer,
    batch_size=4,
    timesteps=256,
    subset='train',
    data_dir='/tmp/data/wikitext-103'
)

# Get batched data
for batch in dataset:
    print(f"Batch shape: {batch['obs'].shape}")
    break
```

## 🔍 Validation & Troubleshooting

### Validate Dataset

```bash
# Lightweight validation
python scripts/setup_wikitext103_dataset.py --validate_only /tmp/data

# Full validation with statistics
python scripts/create_processed_wikitext103_dataset.py --stats --data_dir /tmp/data
```

### Common Issues

1. **Download failures**: 
   - Check internet connection
   - Ensure sufficient disk space (~1GB)
   - Try different output directory

2. **File not found errors**:
   - Verify paths in code examples
   - Check that download completed successfully
   - Use `--validate_only` to check files

3. **Import errors (full setup)**:
   - Install WikiGraphs dependencies: `pip install jax haiku optax`
   - Run from wikigraphs directory
   - Use lightweight setup instead

## 🎉 Benefits

### **Before Issue #40**:
- Manual download of multiple files
- Complex setup procedures  
- No standardized vocabulary
- Inconsistent file organization
- No validation or documentation

### **After Our Solution**:
- ✅ **One-command setup**: Everything automated
- ✅ **Multiple options**: Lightweight vs full-featured
- ✅ **Robust validation**: Integrity checks and statistics
- ✅ **Ready-to-use examples**: Copy-paste code snippets
- ✅ **Cross-platform**: Works on Windows, Linux, macOS
- ✅ **Flexible integration**: Works with any NLP framework

## 📝 Technical Details

### Lightweight Setup Features:
- Zero external dependencies (only Python stdlib)
- Simple vocabulary creation from tokenized data
- Basic validation and statistics
- File organization and cleanup
- Progress tracking during downloads

### Full Setup Features:
- Integration with WikiGraphs data loaders
- Comprehensive dataset validation
- Advanced vocabulary creation with configurable thresholds
- Detailed statistical analysis
- Rich documentation generation
- Multiple operation modes

## 🔗 Integration

The processed dataset works with:
- **WikiGraphs**: `from wikigraphs.data import wikitext`
- **Hugging Face**: Convert to datasets format
- **PyTorch**: Use with DataLoader
- **TensorFlow**: Convert to tf.data format
- **Raw processing**: Direct file access

## 📚 Citation

```bibtex
@article{merity2016pointer,
  title={Pointer Sentinel Mixture Models},
  author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  journal={arXiv preprint arXiv:1609.07843},
  year={2016}
}
```

## 🎯 Impact

**Issue #40 Status**: ✅ **SOLVED**

This solution transforms WikiText-103 setup from a complex multi-step process to a simple one-command operation, making processed WikiText-103 datasets easily accessible to the research community.

---

*Created for GSoC 2026 contribution to solve Issue #40*
