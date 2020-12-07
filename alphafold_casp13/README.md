# AlphaFold

This package provides an implementation of the contact prediction network,
associated model weights and CASP13 dataset as published in Nature.

Any publication that discloses findings arising from using this source code must
cite *Improved protein structure prediction using potentials from deep learning*
by Andrew W. Senior, Richard Evans, John Jumper, James Kirkpatrick, Laurent
Sifre, Tim Green, Chongli Qin, Augustin Žídek, Alexander W. R. Nelson, Alex
Bridgland, Hugo Penedones, Stig Petersen, Karen Simonyan, Steve Crossan,
Pushmeet Kohli, David T. Jones, David Silver, Koray Kavukcuoglu, Demis Hassabis.

The paper abstract can be found on Nature's site
[10.1038/s41586-019-1923-7](https://www.nature.com/articles/s41586-019-1923-7)
and the full text can be accessed directly at https://rdcu.be/b0mtx.

## Setup

**This code can't be used to predict structure of an arbitrary protein sequence.
It can be used to predict structure only on the CASP13 dataset (links below).**
The feature generation code is tightly coupled to our internal infrastructure as
well as external tools, hence we are unable to open-source it. We give guide as
to the features used for those accustomed to computing them below. See also
[issue #18](https://github.com/deepmind/deepmind-research/issues/28) for more
details.

This code works on Linux, we don't support other operating systems.

### Dependencies

*   Python 3.6+.
*   [Abseil 0.8.0](https://github.com/abseil/abseil-py)
*   [Numpy 1.16](https://numpy.org)
*   [Six 1.12](https://pypi.org/project/six/)
*   [Setuptools 41.0.0](https://setuptools.readthedocs.io/en/latest/)
*   [Sonnet 1.35](https://github.com/deepmind/sonnet)
*   [TensorFlow 1.14](https://tensorflow.org). Not compatible with TensorFlow
    2.0+.
*   [TensorFlow Probability 0.7.0](https://www.tensorflow.org/probability)

You can set up Python virtual environment (you might need to install the
`python3-venv` package first) with all needed dependencies inside the forked
`deepmind_research` repository using:

```shell
python3 -m venv alphafold_venv
source alphafold_venv/bin/activate
pip install wheel
pip install -r alphafold_casp13/requirements.txt
```

Alternatively, you can just use the `run_eval.sh` script provided which will run
these commands for you. See the section on running the system below for more
details.

## Data

While the code is licensed under the Apache 2.0 License, the AlphaFold weights
and data are made available for non-commercial use only under the terms of the
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
license. You can find details at:
https://creativecommons.org/licenses/by-nc/4.0/legalcode

You can download the data from:

*   http://bit.ly/alphafold-casp13-data-license: The data license file.
*   http://bit.ly/alphafold-casp13-data: The dataset to reproduce AlphaFold's
    CASP13 results.
*   http://bit.ly/alphafold-casp13-weights: The model checkpoints.

### Input data

The dataset to reproduce AlphaFold's CASP13 results can be downloaded from
http://bit.ly/alphafold-casp13-data. The dataset is in a single zip file called
`casp13_data.zip` which has about **43.5 GB**.

The zip file contains 1 directory for each CASP13 target and a `LICENSE.txt`
file. Each target directory contains the following files:

1.  `TARGET.tfrec` file. This is a
    [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) file
    with serialized tf.train.Example protocol buffers that contain the features
    needed to run the model.
1.  `contacts/TARGET.pickle` file(s) with the predicted distogram.
1.  `contacts/TARGET.rr` file(s) with the contact map derived from the predicted
    distogram. The RR format is described on the
    [CASP website](http://predictioncenter.org/casp13/index.cgi?page=format#RR).

Note that for **T0999** the target was manually split based on hits in HHSearch
into 5 sub-targets, hence there are 5 distograms
(`contacts/T0999s{1,2,3,4,5}.pickle`) and 5 RR files
(`contacts/T0999s{1,2,3,4,5}.rr`).

The `contacts/` folder is not needed to run the model, these files are included
only for convenience so that you don't need to run the inference for CASP13
targets to get the contact map.

### Model checkpoints

The model checkpoints can be downloaded from
http://bit.ly/alphafold-casp13-weights. The model checkpoints are in a zip file
called `alphafold_casp13_weights.zip` which has about **210 MB**.

The zip file contains:

1.  A directory `873731`. This contains the weights for the distogram model.
1.  A directory `916425`. This contains the weights for the background distogram
    model.
1.  A directory `941521`. This contains the weights for the torsion model.
1.  `LICENSE.txt`. The model checkpoints have a non-commercial license which is
    defined in this file.

Each directory with model weights contains a number of different model
configurations. Each model has a config file and associated weights. There is
only one torsion model. Each model directory also contains a stats file that is
used for feature normalization specific to that model.

## Distogram prediction

### Running the system

You can use the `run_eval.sh` script to run the entire Distogram prediction
system. There are a few steps you need to start with:

1.  Download the input data as described above. Unpack the data in the directory
    with the code.
1.  Download the model checkpoints as described above. Unpack the data.
1.  In `run_eval.sh` set the following:
    *   `DISTOGRAM_MODEL` to the path to the directory with the distogram model.
    *   `BACKGROUND_MODEL` to the path to the directory with the background
        model.
    *   `TORSION_MODEL` to the path to the directory with the torsion model.
    *   `TARGET` to the name of the target.
    *   `TARGET_PATH` to the path to the directory with the target input data.
    *   `OUTPUT_DIR` is by default set to a new directory with a timestamp
        within your home directory.

Then run `alphafold_casp13/run_eval.sh` from the `deepmind_research` parent
directory (you will get errors if you try running `run_eval.sh` directly from
the `alphafold_casp13` directory).

The contact prediction works in the following way:

1.  4 replicas (by *replica* we mean a configuration file describing the network
    architecture and a snapshot with the network weights), each with slightly
    different model configuration, are launched to predict the distogram.
1.  4 replicas, each with slightly different model configuration are launched to
    predict the background distogram.
1.  1 replica is launched to predict the torsions.
1.  The predictions from the different replicas are averaged together using
    `ensemble_contact_maps.py`.
1.  The predictions for the 64 × 64, 128 × 128 and 256 × 256 distogram crops are
    pasted together using `paste_contact_maps.py`.

When running `run_eval.sh` the output has the following directory structure:

*   **distogram/**: Contains 4 subfolders, one for each replica. Each of these
    contain the predicted ASA, secondary structure and a pickle file with the
    distogram for each crop (see below for more details). It also contains an
    `ensemble` directory with the ensembled distograms.
*   **background_distogram/**: Contains 4 subfolders, one for each replica. Each
    of these contain a pickle file with the background distogram for each crop.
    It also contains an `ensemble` directory with the ensembled background
    distograms.
*   **torsion/**: Contains 1 subfolder as there was only a single replica. This
    folder contains contains the predicted ASA, secondary structure, backbone
    torsions and a pickle file with the distogram for each crop. It also
    contains an `ensemble` directory, which contains a copy of the predicted
    output as there is only a single replica in this case.
*   **pasted/**: Contains distograms obtained from the ensembled distograms by
    pasting. An RR contact map file is computed from this pasted distogram.
    **This is the final distogram that was used in the subsequent AlphaFold
    folding pipeline in CASP13.**

### Distogram output format

The distogram is a Python pickle file with a dictionary containing the following
fields:

*   `min_range`: The minimum range in Angstroms to consider in distograms.
*   `max_range`: The range in Angstroms to consider in distograms, see
    `num_bins` below for clarification. The upper end of the distogram is
    `min_range + max_range`.
*   `num_bins`: The number of bins in the distance histogram being predicted. We
    divide the interval from `min_range` to `min_range + max_range` into this
    many bins. The distograms were trained so that distances lower than
    `min_range` were counted in the lowest bin and distances higher than
    `min_range + max_range` were added to the final bin. The `num_bins - 1`
    boundaries between bins are thus `np.linspace(0, max_range, num_bins + 1,
    endpoint=True)[1:-1] + min_range`.
*   `sequence`: The target sequence of amino acids of length `L`.
*   `target`: The name of the target.
*   `domain`: The name of the target including the domain name.
*   `probs`: The distogram as a Numpy array of shape `[L, L, num_bins]`.

## Data splits

We used a version of [PDB](https://www.rcsb.org/) downloaded on 2018-03-15. The
train/test split can be found in the `train_domains.txt` and `test_domains.txt`
files in this repository. The split is based on the
[CATH 2018-03-16](https://www.cathdb.info/) database.

## Features

There is currently no plan to open source the feature generation code as it is
tightly coupled to our internal infrastructure as well as external tools which
we cannot open source.

Some features are needed only as placeholders to construct the model. These can
be set to all zeros when running the inference. Such features are marked in the
table below as not needed and you can just fill them with zeros when running
inference.

The table below provides an overview of the features we used to make it possible
to reconstruct our feature generation code. Some features that require more
thorough explanation are explained in the section below the table. Note that
`NR` stands for number of residues, i.e. the length of the amino acid sequence:

| Name                              | Needed | TF DType | Shape           | Description                                                                                                                  |
|-----------------------------------|:------:|----------|-----------------|------------------------------------------------------------------------------------------------------------------------------|
| `aatype`                          | ✔️      | float32  | `(NR, 21)`      | One hot encoding of amino acid types. The mapping is `ARNDCQEGHILKMFPSTWYVX -> range(21)`. See below.                        |
| `alpha_mask`                      | ❌     | int64    | `(NR, 1)`       | Mask for `alpha_positions`.                                                                                                  |
| `alpha_positions`                 | ❌     | float32  | `(NR, 3)`       | `(x, y, z)` Carbon Alpha coordinates.                                                                                        |
| `beta_mask`                       | ❌     | int64    | `(NR, 1)`       | Mask for `beta_positions`.                                                                                                   |
| `beta_positions`                  | ❌     | float32  | `(NR, 3)`       | `(x, y, z)` Carbon Beta coordinates.                                                                                         |
| `between_segment_residues`        | ❌     | int64    | `(NR, 1)`       | The number of between segment residues (BSR) at the next position. E.g. `ABCXXD` (`XX` is BSR) would be `[0,0,2,0]`.         |
| `chain_name`                      | ❌     | string   | `(1)`           | The chain name. E.g. 'A', 'B', ...                                                                                           |
| `deletion_probability`            | ✔️      | float32  | `(NR, 1)`       | The fraction of sequences that had a deletion at this position. See below.                                                   |
| `domain_name`                     | ❌     | string   | `(1)`           | The domain name.                                                                                                             |
| `gap_matrix`                      | ✔️      | float32  | `(NR, NR, 1)`   | Covariation signal from the gapped states, this gives an indication of the variance induced due to gapped states. See below. |
| `hhblits_profile`                 | ❌     | float32  | `(NR, 22)`      | A profile (probability distribution over amino acid types) computed using HHBlits MSA. Encoding: 20 amino acids + 'X' + '-'. |
| `hmm_profile`                     | ✔️      | float32  | `(NR, 30)`      | The HHBlits HHM profile (from the `-ohhm` HHBlits output file). Asterisks in the output are replaced by 0.0. See below.      |
| `key`                             | ❌     | string   | `(1)`           | The unique id of the protein.                                                                                                |
| `mutual_information`              | ❌      | float32  | `(NR, NR, 1)`   | The average product corrected mutual information. See https://doi.org/10.1093/bioinformatics/btm604.                         |
| `non_gapped_profile`              | ✔️      | float32  | `(NR, 21)`      | A profile from amino acids only (discounting gaps). See below.                                                               |
| `num_alignments`                  | ✔️      | int64    | `(NR, 1)`       | The number of HHBlits multiple sequence alignments. Has to be repeated `NR` times. See below.                                |
| `num_effective_alignments`        | ❌     | float32  | `(1)`           | The number of effective alignments (neff at 62 % sequence similarity).                                                       |
| `phi_angles`                      | ❌     | float32  | `(NR, 1)`       | The phi angles.                                                                                                              |
| `phi_mask`                        | ❌     | int64    | `(NR, 1)`       | Mask for `phi_angles`.                                                                                                       |
| `profile`                         | ❌     | float32  | `(NR, 21)`      | A profile (probability distribution over amino acid types) computed using PSI-BLAST. Equivalent to the output of ChkParse.   |
| `profile_with_prior`              | ✔️      | float32  | `(NR, 22)`      | A profile computed using HHBlits which takes into account priors and Blosum matrix. See equation 5 in https://doi.org/10.1093/nar/25.17.3389.         |
| `profile_with_prior_without_gaps` | ✔️      | float32  | `(NR, 21)`      | Same as `profile_with_prior` but without gaps included.                                                                      |
| `pseudo_bias`                     | ✔️      | float32  | `(NR, 22)`      | The bias computed in the MSA pseudolikelihood computation.                                                                   |
| `pseudo_frob`                     | ✔️      | float32  | `(NR, NR, 1)`   | Frobenius norm of `pseudolikelihood` (gaps not included). Similar to the output of CCMPred.                                  |
| `pseudolikelihood`                | ✔️      | float32  | `(NR, NR, 484)` | The weights computed in the MSA pseudolikelihood computation.                                                                |
| `psi_angles`                      | ❌     | float32  | `(NR, 1)`       | The psi angles.                                                                                                              |
| `psi_mask`                        | ❌     | int64    | `(NR, 1)`       | Mask for `psi_angles`.                                                                                                       |
| `residue_index`                   | ✔️      | int64    | `(NR, 1)`       | Index of each residue giong from 0 to `NR - 1`. See below.                                                                   |
| `resolution`                      | ❌     | float32  | `(1)`           | The protein structure resolution.                                                                                            |
| `reweighted_profile`              | ✔️      | float32  | `(NR, 22)`      | Profile where sequences are reweighted to weight rarer sequences higher. See below.                                          |
| `sec_structure`                   | ❌     | int64    | `(NR, 8)`       | Secondary structure generated by DSSP and one-hot encoded by the mapping `-HETSGBI -> range(8)`.                             |
| `sec_structure_mask`              | ❌     | int64    | `(NR, 1)`       | Mask for `sec_structure_mask`.                                                                                               |
| `seq_length`                      | ✔️      | int64    | `(NR, 1)`       | The length of the amino acid sequence. Has to be repeated `NR` times. See below.                                             |
| `sequence`                        | ✔️      | string   | `(1)`           | The amino acid sequence (1-letter amino acid encoding). See below.                                                           |
| `solv_surf`                       | ❌     | float32  | `(NR, 1)`       | Relative solvent accessible area computed using DSSP and then normalized by amino acid maximum accessibility.                |
| `solv_surf_mask`                  | ❌     | int64    | `(NR, 1)`       | Mask for `solv_surf`.                                                                                                        |
| `superfamily`                     | ❌     | string   | `(1)`           | The superfamily CATH code.                                                                                                   |

### More details on needed features

#### `aatype`

One hot encoding of amino acid types. The following code converts an amino acid
string into the one-hot encoding:

```python
def sequence_to_onehot(sequence):
  """Maps the given sequence into a one-hot encoded matrix."""
  mapping = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYVX')}
  num_entries = max(mapping.values()) + 1
  one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

  for aa_index, aa_type in enumerate(sequence):
    aa_id = mapping[aa_type]
    one_hot_arr[aa_index, aa_id] = 1

  return one_hot_arr
```

#### `deletion_probability`

The fraction of sequences that had an insert state (denoted by a lowercase
letter in the A3M format) at this position. We used the following code to
compute it from the HHBlits MSA in the A3M format:

```python
deletion_matrix = []
for msa_sequence in hhblits_a3m_sequences:
  deletion_vec = []
  deletion_count = 0
  for j in msa_sequence:
    if j.islower():
      deletion_count += 1
    else:
      deletion_vec.append(deletion_count)
      deletion_count = 0
  deletion_matrix.append(deletion_vec)

deletion_matrix = np.array(deletion_matrix)
deletion_matrix[deletion_matrix != 0] = 1.0
deletion_probability = deletion_matrix.sum(axis=0) / len(deletion_matrix)
```

#### `gap_matrix`

Covariation signal from the gapped states, this gives an indication of the
variance induced due to gapped states. Example:

```
MSA = A A C D B D F J G B M A
      - - C D B D F J G B M A
      A A C D B - - J G B M A
gap_count = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]]
gap_matrix = np.matmul(gap_count.T, gap_count)
```

#### `hmm_profile`

The HHBlits HHM profile (from the `-ohhm` HHBlits output file). Asterisks in the
output are replaced by 0.0. The following code parses the HHM file:

```python
def extract_hmm_profile(hhm_file, sequence, asterisks_replace=0.0):
  """Extracts information from the hmm file and replaces asterisks."""
  profile_part = hhm_file.split('#')[-1]
  profile_part = profile_part.split('\n')
  whole_profile = [i.split() for i in profile_part]
  # This part strips away the header and the footer.
  whole_profile = whole_profile[5:-2]
  gap_profile = np.zeros((len(sequence), 10))
  aa_profile = np.zeros((len(sequence), 20))
  count_aa = 0
  count_gap = 0
  for line_values in whole_profile:
    if len(line_values) == 23:
      # The first and the last values in line_values are metadata, skip them.
      for j, t in enumerate(line_values[2:-1]):
        aa_profile[count_aa, j] = (
            2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
      count_aa += 1
    elif len(line_values) == 10:
      for j, t in enumerate(line_values):
        gap_profile[count_gap, j] = (
            2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
      count_gap += 1
    elif not line_values:
      pass
    else:
      raise ValueError('Wrong length of line %s hhm file. Expected 0, 10 or 23'
                       'got %d'%(line_values, len(line_values)))
  hmm_profile = np.hstack([aa_profile, gap_profile])
  assert len(hmm_profile) == len(sequence)
  return hmm_profile
```

#### `non_gapped_profile`

A profile from amino acids only (discounting gaps).

```python
def non_gapped_profile(amino_acids):
  """Computes a profile from only amino acids and discounting gaps."""
  profile = np.zeros(21)
  for aa in amino_acids:
    if aa != 21:  # Ignore gaps.
      profile[aa] += 1.
  return profile / np.sum(profile)
```

#### `num_alignments`

The number of HHBlits multiple sequence alignments. Has to be repeated `NR`
times. For example, if there are 10 alignments for a sequence of length 8, then
`num_alignments = [[10], [10], [10], [10], [10], [10], [10], [10]]`.

#### `pseudo_frob`
This feature collapses the 484 channels of pseudolikelihood into one by taking
the Frobenius norm of the 484 channels and then subtracting the Average Product
Correction of the computed Frobenius norm. The Frobenius norm does not take into
account the 22nd gap state.

#### `pseudolikelihood`

Parameters of a Potts Model coupling the amino acid types of particular residues
estimated by pseudolikelihood. See https://doi.org/10.1103/PhysRevE.87.012707
for more details.

#### `residue_index`

Index of each residue giong from 0 to `NR - 1`. For example, the sequence `AACR`
has `residue_index = [[0], [1], [2], [3]]`.

#### `reweighted_profile`

Profile where sequences are reweighted to weight rarer sequences higher. The
sequence weights are calculated like this:

```python
def sequence_weights(sequence_matrix):
  """Compute sequence reweighting to weight rarer sequences higher."""
  num_rows, num_res = sequence_matrix.shape
  cutoff = 0.62 * num_res
  weights = np.ones(num_rows, dtype=np.float32)
  for i in range(num_rows):
    for j in range(i + 1, num_rows):
      similarity = (sequence_matrix[i] == sequence_matrix[j]).sum()
      if similarity > cutoff:
        weights[i] += 1
        weights[j] += 1
  return 1.0 / weights
```

#### `seq_length`

The length of the amino acid sequence. Has to be repeated `NR` times. For
example, the sequence `AACR` would have  `seq_length = [[4], [4], [4], [4]]`.

#### `sequence`

The amino acid sequence (1-letter amino acid encoding). For example, a protein
with Alanine, Lysine, Arginine has `sequence = 'AKR'`.


# Disclaimer

This is not an official Google product.
