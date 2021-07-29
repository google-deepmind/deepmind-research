# Perceiver and Perceiver IO

Perceiver [1] is a general architecture that works on many kinds of data,
including images, video, audio, 3D point clouds, language and symbolic inputs,
multimodal combinations, etc.
Perceivers can handle new types of data with only minimal modifications.
Perceivers process inputs using domain-agnostic Transformer-style attention.
Unlike Transformers, Perceivers first map inputs to a small latent space where
processing is cheap and doesn't depend on the input size.
This makes it possible to build very deep networks
even when using large inputs like images or videos.

Perceiver IO [2] is a generalization of Perceiver to handle arbitrary *outputs*
in addition to arbitrary inputs.
The original Perceiver only produced a single classification label.
In addition to classification labels,
Perceiver IO can produce (for example) language, optical flow,
and multimodal videos with audio.
This is done using the same building blocks as the original Perceiver.
The computational complexity of Perceiver IO is linear in the input and output
size and the bulk of the processing occurs in the latent space,
allowing us to process inputs and outputs that are much larger
than can be handled by standard Transformers.
This means, for example, Perceiver IO can do BERT-style masked language modeling
directly using *bytes* instead of tokenized inputs.

This directory contains our implementation of Perceiver IO
(encompassing the original Perceiver as a special case).
The `perceiver.py` file contains our implementation of Perceiver IO,
and `io_processors.py` contains domain-specific input and output processors
for the experiments we ran.
We provide example colabs in the `colabs` directory to demonstrate
how our models can be used and show the qualitative performance of Perceiver IO.

## Usage

First, install dependencies following these instructions:

1. Create a virtual env: `python3 -m venv ~/.venv/perceiver`
2. Switch to the virtual env: `source ~/.venv/perceiver/bin/activate`
3. Follow instructions for installing JAX on your platform:
   https://github.com/google/jax#installation
4. Install other dependencies: `pip install -f requirements.txt`

After install dependencies, you can open the notebooks in the `colabs` directory
using Jupyter or Colab.

### Colabs
We provide the following colabs:

* colabs/optical_flow.ipynb: Colab for running a pre-trained optical flow
  Perceiver model and visualizing the output flow (Section 4.2 in [2]).
* colabs/video_autoencoding.ipynb: Colab for running a pre-trained
  video autoencoding Perceiver model and visualizing video reconstructions
  (Section 4.3 in [2]).

## References

[1] Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals,
Joao Carreira.
*Perceiver: General Perception with Iterative Attention*. ICML 2021.

[2] TODO: Add citation after paper is published on ArXiv.
