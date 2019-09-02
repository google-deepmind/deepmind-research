# Deep Compressed Sensing

This is the example code for the following ICML 2019 paper.  If you use the code
here please cite this paper.

> Yan Wu, Mihaela Rosca, Timothy Lillicrap
  *Deep Compressed Sensing*.  ICML 2019.  [\[arXiv\]](https://arxiv.org/abs/1905.06723).


## Running the code

The code contains:

  * the implementation of the compressed sensing algorithm (`cs.py`).
  * the implementation of the GAN algorithm (`gan.py`).
  * a main file (`main_cs.py`) to reproduce the Compressed Sensing results in
  the paper.
  * a main file (`main_gan.py`) to reproduce the GAN results in the paper
  (the improvement over the SN-GAN baseline via latent optimization).
