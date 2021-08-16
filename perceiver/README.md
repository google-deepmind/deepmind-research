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
how our models can be used and show the qualitative performance of Perceiver IO
on a diverse collection of tasks.

## Usage

First, install dependencies following these instructions:

1. Create a virtual env: `python3 -m venv ~/.venv/perceiver`
2. Switch to the virtual env: `source ~/.venv/perceiver/bin/activate`
3. Follow instructions for installing JAX on your platform:
   https://github.com/google/jax#installation
4. Install other dependencies: `pip install -f requirements.txt`

After installing dependencies, you can open the notebooks in the `colabs` directory
using Jupyter or Colab, and you can run our example training script.
Our colabs and training script assume that you are running from the
`deepmind_research` directory.

### Colabs
We provide the following colabs:

* [colabs/masked_language_modelling.ipynb](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/perceiver/colabs/masked_language_modelling.ipynb):
  Colab for running a pre-trained
  Perceiver IO masked-language model (Section 4.1 in [2]).
* [colabs/optical_flow.ipynb](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/perceiver/colabs/optical_flow.ipynb):
  Colab for running a pre-trained optical flow
  Perceiver IO model and visualizing the output flow (Section 4.2 in [2]).
* [colabs/video_autoencoding.ipynb](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/perceiver/colabs/video_autoencoding.ipynb):
  Colab for running a pre-trained
  video autoencoding Perceiver IO model and visualizing video reconstructions
  (Section 4.3 in [2]).
* [colabs/imagenet_classification.ipynb](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/perceiver/colabs/imagenet_classification.ipynb):
  Colab for running three pre-trained
  ImageNet classification Perceiver IO models (Section 4.5 in [2]).

### Training scripts
We also provide an example training script to train a Perceiver IO model for
ImageNet classification.
The provided hyperparameters are the settings used to train Perceiver IO
with 2D Fourier position encodings, as described in
section 4.5 and supplemental section H of the paper [2].

To run the script locally and train a miniature Perceiver model,
run: `perceiver/train/launch_local.sh`.
The script would need to be adapted to run on a distributed training setup
in order to train a full-scale model with the full batch size.

## Attributions and Disclaimers

The file `perceiver/train/autoaugment.py` originates from the `tensorflow/tpu`
repository (https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/efficientnet/autoaugment.py),
copyright (c) The Tensorflow Authors.

Sintel data is provided by and available from Sintel.org (https://durian.blender.org/),
copyright (c) Blender Foundation/www.sintel.org.

The sample image in `imagenet_classification.ipynb` is obtained from
Getty Images under license (https://www.gettyimages.co.uk/eula#RF).

Video content may include clips provided as part of the THUMOS Challenge datasets,
which may be accessed at http://crcv.ucf.edu/THUMOS14/download.html,
copyrights held by the creators.

All data and parameters included with Perceiver are made available
under the terms of the CC BY 4.0 license,
available at https://creativecommons.org/licenses/by/4.0/legalcode.

This is not an officially supported Google product.

## References

[1] Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals,
João Carreira.
*Perceiver: General Perception with Iterative Attention*. ICML 2021.
https://arxiv.org/abs/2103.03206

[2] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch,
Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock,
Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman,
Oriol Vinyals, João Carreira.
*Perceiver IO: A General Architecture for Structured Inputs & Outputs*.
arXiv, 2021.
https://arxiv.org/abs/2107.14795
