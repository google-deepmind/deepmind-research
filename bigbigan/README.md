# Pretrained BigBiGAN models

We have released pretrained BigBiGAN models used for unsupervised image
generation and representation learning, as described in our July 2019
tech report,
["Large Scale Adversarial Representation Learning"](https://arxiv.org/abs/1907.02544) [1].

The pretrained models are available for use via [TF Hub](https://tfhub.dev/s?publisher=deepmind&q=bigbigan).
The release includes two BigBiGAN models with different encoder architectures:

* Small encoder (ResNet-50): [https://tfhub.dev/deepmind/bigbigan-resnet50/1](https://tfhub.dev/deepmind/bigbigan-resnet50/1)
* Large encoder (RevNet-50 x4): [https://tfhub.dev/deepmind/bigbigan-revnet50x4/1](https://tfhub.dev/deepmind/bigbigan-revnet50x4/1)

See the TF Hub pages linked above for documentation and example usage of each module.

## Demo (Colab)

A Google Colab-based demo with example usage of the model functionality and sample visualization is available [here](//colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/bigbigan_with_tf_hub.ipynb).

## Example use
The snippet below demonstrates the use of the released TF Hub modules for
image generation/reconstruction and encoder feature computation.
(The [Colab demo](//colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/bigbigan_with_tf_hub.ipynb)
includes more extensive documentation and visualizations.)

```python
# Load BigBiGAN module.
module = hub.Module('https://tfhub.dev/deepmind/bigbigan-resnet50/1')  # small encoder
# module = hub.Module('https://tfhub.dev/deepmind/bigbigan-revnet50x4/1')  # large encoder

# Sample a batch of 8 random latent vectors (z) from the Gaussian prior. Then
# call the generator on the latent samples to generate a batch of images with
# shape [8, 128, 128, 3] and range [-1, 1].
z = tf.random.normal([8, 120])  # latent samples
gen_samples = module(z, signature='generate')

# Given a batch of 256x256 RGB images in range [-1, 1], call the encoder to
# compute predicted latents z and other features (e.g. for use in downstream
# recognition tasks).
images = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
features = module(images, signature='encode', as_dict=True)

# Get the predicted latent sample `z_sample` from the dict of features.
# Other available features include `avepool_feat` and `bn_crelu_feat`, used in
# the representation learning results.
z_sample = features['z_sample']  # shape [?, 120]

# Compute reconstructions of the input `images` by passing the encoder's output
# `z_sample` back through the generator. Note that raw generator outputs are
# half the resolution of encoder inputs (128x128). To get upsampled generator
# outputs matching the encoder input resolution (256x256), instead use:
#     recons = module(z_sample, signature='generate', as_dict=True)['upsampled']
recons = module(z_sample, signature='generate')  # shape [?, 128, 128, 3]
```

## References

[1] Jeff Donahue and Karen Simonyan.
[Large Scale Adversarial Representation Learning](https://arxiv.org/abs/1907.02544).
*arxiv:1907.02544*, 2019.
