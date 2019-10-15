# Transporter - Unsupervised Learning of Object Keypoints for Perception and Control

This directory contains a [Sonnet](https://sonnet.dev) implementation of
the Transporter architecture.

The Transporter is a neural network architecture for discovering concise
geometric object representations in terms of keypoints or image-space
coordinates. Our method learns from raw video frames in a fully unsupervised
manner, by transporting learnt image features between video frames using a
keypoint bottleneck. The discovered keypoints track objects and object parts
across long time-horizons more accurately than recent similar methods.

For details, see our
paper [Unsupervised Learning of Object Keypoints for Perception and Control](https://arxiv.org/abs/1906.11883).

## Contributors
* Tejas Kulkarni <tkulkarni@google.com>
* Ankush Gupta <ankushgupta@google.com>
* Catalin Ionescu
* Sebastian Borgeaud
* Malcolm Reynolds
* Andrew Zisserman
* Volodymyr Mnih


