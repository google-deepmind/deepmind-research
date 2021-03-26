# Enformer

This package provides an implementation of the Enformer model and examples on
running the model.

If this source code or accompanying files are helpful for your research please
cite the following publication:

"Effective gene expression prediction from sequence by integrating long-range
interactions"

Žiga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R. Ledsam,
Agnieszka Grabska-Barwinska, Kyle R. Taylor, Yannis Assael, John Jumper,
Pushmeet Kohli, David R. Kelley

## Setup

Requirements:

*   dm-sonnet (2.0.0)
*   kipoiseq (0.5.2)
*   numpy (1.19.5)
*   pandas (1.2.3)
*   tensoflow (2.4.1)
*   tensorflow-hub (0.11.0)

See `requirements.txt`.

To run the unit test:

```shell
python3.8 -m venv enformer_venv
source enformer_venv/bin/activate
pip install -r requirements.txt
python -m enformer_test
```

## Pre-computed variant effect predictions

We precomputed variant effect scores for all frequent variants (MAF>0.5%, in any
population) present in the 1000 genomes project. Variant scores in HDF5 file
format per chromosome for HG19 reference genome can be found
[here](https://console.cloud.google.com/storage/browser/dm-enformer/variant-scores/1000-genomes/enformer).
The HDF5 file has the same format as the output of
[this](https://github.com/calico/basenji/blob/738321c85f8925ae6ac318a6cd4901a42ea6bc3f/bin/basenji_sad.py#L264)
script and contains the following arrays:

*   snp \[num_snps](string) - snp id
*   chr \[num_snps](string) - chromosome name
*   pos \[num_snps](uint32) - position (1-based)
*   ref \[num_snps](string) - reference allele
*   alt \[num_snps](string) - alternative allele
*   target_ids \[num_targets](string) - target ids
*   target_labels \[num_targets](string) - target names
*   SAD \[num_snps, num_targets](float16) - SNP Activity Difference (SAD)
    scores - main variant effect score computed as `model(alternate_sequence) -
    model(reference_sequence)`.
*   SAR \[num_snps, num_targets](float16) - Same as SAD, by computing
    `np.log2(1 + model(alternate_sequence)) - np.log2(1 +
    model(reference_sequence))`

Furthermore, we provide the top 20 principal components of variant-effect scores
in the [PC20 folder](https://console.cloud.google.com/storage/browser/dm-enformer/variant-scores/1000-genomes/enformer/PC20)
stored as a tabix-indexed TSV file per chromosome (HG19 reference
genome). The format of these files has the following columns:

*   #CHROM - chromosome (chr1)
*   POS - variant position (1-based)
*   ID - dbSNP identifier
*   REF - reference allele (e.g. A)
*   ALT - alternate allele (e.g. T)
*   PC{i} - i-th principal component of the variant effect prediction.

All model predictions are licensed under
[CC-BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

## Running Inference

The simplest way to perform inference is to load the model via tfhub.dev (TODO:
LINK). The input sequence length is 393,216 with the prediction corresponding to
128 base pair windows of the center 114,688 base pairs. The input sequence is
one hot encoded using the order of indices being 'ACGT' with N values being all
zeros.

```python
import tensorflow as tf
import tensorflow_hub as hub

enformer = hub.Module('https://tfhub.dev/deepmind/enformer/1')

SEQ_LENGTH = 393_216

# Numpy array [batch_size, SEQ_LENGTH, 4] one hot encoded in order 'ACGT'. The
# `one_hot_encode` function is available in `enformer.py` and outputs can be
# stacked to form a batch.
inputs = tf.zeros((1, SEQ_LENGTH, 4), dtype=tf.float32)
predictions = enformer.predict_on_batch(inputs)
predictions['human'].shape  # [batch_size, 896, 5313]
predictions[mouse].shape  # [batch_size, 896, 1643]
```

## Outputs

For each 128 bp window, predictions are made for every track. The mapping from
track idx to track name is found in the corresponding file in the basenji
[dataset](https://github.com/calico/basenji/tree/master/manuscripts/cross2020)
folder (targets_{organism}.txt file).

As an example, to load track annotations for the human targets:

```python
import pandas as pd
targets_txt = 'https://raw.githubusercontent.com/calico/basenji/0.5/manuscripts/cross2020/targets_human.txt'
df_targets = pd.read_csv(targets_txt, sep='\t')
df_targets.shape  # (5313, 8) With rows match output shape above.
```

## Training Code

The model is implemented using [Sonnet](https://github.com/deepmind/sonnet). The
full sonnet module is defined in `enformer.py` called Enformer. See
[enformer-training.ipynb](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-training.ipynb).
on how to train the model on Basenji2 data.

## Colab

Further usage and training examples are given in the following colab notebooks:

### `enformer-usage.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-usage.ipynb).

This shows how to:

*   **Make predictions** with pre-trained Enformer and reproduce Fig. 1d
*   **Compute contribution scores** and reproduce parts of Fig. 2a
*   **Predict the effect of genetic variants** and reproduce parts of Fig. 3g
*   Score multiple variants in a VCF

### `enformer-training.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-training.ipynb).

This colab shows how to:

* Setup training data by directly accessing the Basenji2 data on GCS
* Train the model for a few steps on both human and mouse genomes
* Evaluate the model on human and mouse genomes

## Disclaimer

This is not an official Google product.

