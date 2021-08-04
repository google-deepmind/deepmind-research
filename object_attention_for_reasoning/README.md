Implementation of the ALOE model
["Attention over learned object embeddings enables complex visual reasoning"](https://arxiv.org/abs/2012.08508)
[1].

This package includes source code for the ALOE transformer model,
pre-trained model parameters for the CLEVRER task,
and MONet [2] latent variables for all videos in the training
and validation sets. It does not include the model training code.
See Section 2 of [1] for details.

[1] David Ding, Felix Hill, Adam Santoro, Malcolm Reynolds, Matt Botvinick.
*Attention over learned object embeddings enables complex visual reasoning*.
arXiv preprint arXiv:2012.08508, 2020.

[2] Chris P. Burgess, Loic Matthey, Nick Watters, Rishabh Kabra, Irina Higgins,
Matt Botvinick, and Alexander Lerchner
*MONet: Unsupervised scene decomposition and representation*.
arXiv preprint arXiv:1901.11390, 2019.


# Instructions

Note: This code depends on Tensorflow 1 and Sonnet 1. Tensorflow 1 is only
available on PYPI for Python 3.7 and earlier.

To run this code, execute the following commands from the `deepmind_research/`
directory:

```shell
# Download checkpoints and MONet latents
wget https://storage.googleapis.com/object-attention-for-reasoning/checkpoints_and_latents.zip
unzip checkpoints_and_latents.zip
python3.7 -m venv object_based_attention_venv
source object_based_attention_venv/bin/activate
pip install --upgrade setuptools wheel
pip install -r requirements.txt
python -m object_attention_for_reasoning.run_model
```
If the code runs correctly, you should see the model's predicted answer to two
CLEVRER questions (a descriptive one and a multiple choice one), and both
answers should be correct.

If you find the provided code useful, please cite this paper:
```
@article{aloe2020,
  title={Attention over learned object embeddings enables complex visual reasoning},
  author={David Ding and Felix Hill and Adam Santoro and Malcolm Reynolds and Matt Botvinick},
  journal={arXiv preprint arXiv:2012.08508},
  year={2020}
}
```
