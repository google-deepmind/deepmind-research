# Transporter - Unsupervised Learning of Object Keypoints for Perception and Control

This directory contains a [Sonnet](https://sonnet.dev) implementation of
the Transporter architecture and a notebook explaining how the model can be used
for keypoint inference. To launch the notebook in Google colab, [click here](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/transporter/transporter_example.ipynb).


The Transporter is a neural network architecture for discovering concise
geometric object representations in terms of keypoints or image-space
coordinates. Our method learns from raw video frames in a fully unsupervised
manner, by transporting learnt image features between video frames using a
keypoint bottleneck. The discovered keypoints track objects and object parts
across long time-horizons more accurately than recent similar methods.

For details, see our
paper [Unsupervised Learning of Object Keypoints for Perception and Control](https://arxiv.org/abs/1906.11883).

If you use the code here please cite this paper.
> Tejas Kulkarni, Ankush Gupta, Catalin Ionescu, Sebastian Borgeaud, Malcolm Reynolds, Andrew Zisserman, Volodymyr Mnih.  *Unsupervised Learning of Object Keypoints for Perception and Control*.  NeurIPS 2019.  [\[arXiv\]](https://arxiv.org/abs/1906.11883).

## Contributors
* Tejas Kulkarni <tkulkarni@google.com>
* Ankush Gupta <ankushgupta@google.com>
* Catalin Ionescu
* Sebastian Borgeaud
* Malcolm Reynolds
* Andrew Zisserman
* Volodymyr Mnih


## Disclaimer
This is not an official Google product.

