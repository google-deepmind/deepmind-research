# Continual Unsupervised Representation Learning (CURL)

This repository contains code to accompany the NeurIPS 2019 submission on
Continual Unsupervised Representation Learning (CURL).

The experiments in the paper can be reproduced by running one of the three
different training scripts:


`train_sup.py`: to run the supervised continual learning benchmark

`train_unsup.py`: to run the unsupervised i.i.d learning benchmark

`train_main.py`: to run all other experiments in the paper (with details in the
file on what to change)

In each of these cases, the cluster accuracy / purity and k-NN error are logged
to the terminal, and other quantities can be accessed from training.py
(e.g. the confusion matrix can be found in `results['test_confusion']`).

We recommend running these scripts in a Python
[virtual environment](https://docs.python.org/3/tutorial/venv.html):

(Assuming python3-dev is installed in your system)

```console
python3 -m venv .curl_venv
source .curl_venv/bin/activate
pip install wheel
pip install -r requirements.txt

PYTHONPATH=`pwd`/..:$PYTHONPATH python3 train_main.py --dataset='mnist'

Run `deactivate` to exit the virtual environment.
```
