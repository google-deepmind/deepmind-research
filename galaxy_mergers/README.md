# A Deep Learning Approach for Characterizing Major Galaxy Mergers

This repository contains evaluation code and checkpoints to reproduce
figures in https://arxiv.org/abs/2102.05182.

The main evaluation module is `main.py`. It uses the provided checkpoint path
and dataset path to run evaluation.


## Setup

To set up a Python virtual environment with the required dependencies, run:

```shell
python3 -m venv galaxy_mergers_env
source galaxy_mergers_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### License

While the code is licensed under the Apache 2.0 License, the checkpoints weights
are made available for non-commercial use only under the terms of the
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
license. You can find details at:
https://creativecommons.org/licenses/by-nc/4.0/legalcode.


### Citing our work

If you use this work, consider citing our paper:

```bibtex
@article{koppula2021deep,
  title={A Deep Learning Approach for Characterizing Major Galaxy Mergers},
  author={Koppula, Skanda and Bapst, Victor and Huertas-Company, Marc and Blackwell, Sam and Grabska-Barwinska, Agnieszka and Dieleman,   Sander and Huber, Andrea and Antropova, Natasha and Binkowski, Mikolaj and Openshaw, Hannah and others},
  journal={Workshop for Machine Learning and the Physical Sciences @ NeurIPS 2020},
  year={2021}
}
```
