# Colabs

## Dendritic Gated Networks

`dendritic_gated_network.ipynb` implements a Dendritic Gated Network (DGN) solving a regression (using quadratic loss) or a binary classification problem (using Bernoulli log loss).

See our paper titled ["A rapid and efficient learning rule for biological neural circuits"](https://www.biorxiv.org/content/10.1101/2021.03.10.434756v1) for details of the DGN model.

### Instructions for running the `dendritic_gated_network.ipynb` colab/notebook.

We suggest running the [dendritic_gated_network.ipynb](https://github.com/deepmind/deepmind-research/blob/master/gated_linear_networks/colabs/dendritic_gated_network.ipynb) notebook using Google Colaboratory (or Colab). All the dependencies are included by default in Colab cloud runtimes (last tested on the 8th of March, 2021). See https://research.google.com/colaboratory/faq.html for web browser requirements. The notebook runs for about a minute using the free tier runtimes.

The code also runs in JupyterLab/JupyterNotebook (tested on Version 1.02).

1. Visit https://colab.research.google.com/
2. Sign in with your Google account.
3. Click on "File" and select "Open notebook".

4. Then you can
 * either open the notebook directly from GitHub:
     * Click on the GitHub tab
     * Paste https://github.com/deepmind/deepmind-research/blob/master/gated_linear_networks/colabs/dendritic_gated_network.ipynb into the URL section and click the search button. If the notebook does not open automatically, then select the correct notebook from the list provided.
 * or upload the provided notebook manually:
     * Click on the Upload tab
     * Choose or drag dendritic_gated_network.ipynb
5. Click Connect (top right corner) to connect to a run time
6. Click on the "Runtime" tab and select "Run all" to run all the cells.

### Expected outputs
We provide the expected outputs below.

Classification (do_classification = True):

```
epoch: 0, test loss: 0.693 (train: 0.693), test accuracy: 0.412 (train: 0.363)
epoch: 1, test loss: 0.099 (train: 0.196), test accuracy: 0.974 (train: 0.963)
epoch: 2, test loss: 0.095 (train: 0.079), test accuracy: 0.974 (train: 0.978)
epoch: 3, test loss: 0.099 (train: 0.070), test accuracy: 0.974 (train: 0.982)
```
Regression (do_classification = False):

```
epoch: 0, test loss: 0.419 (train: 0.500)
epoch: 1, test loss: 0.388 (train: 0.486)
epoch: 2, test loss: 0.354 (train: 0.439)
epoch: 3, test loss: 0.328 (train: 0.400)
epoch: 4, test loss: 0.310 (train: 0.369)
epoch: 5, test loss: 0.297 (train: 0.344)
epoch: 6, test loss: 0.287 (train: 0.324)
epoch: 7, test loss: 0.281 (train: 0.308)
epoch: 8, test loss: 0.277 (train: 0.296)
epoch: 9, test loss: 0.275 (train: 0.285)
epoch: 10, test loss: 0.273 (train: 0.277)
```
