# PAC Bayes with Backprop (PBB) Quadratic Bound.

Implements PBB quadratic bound referred to as method 'pb_quad' in the paper
"PAC-Bayes with Backprop" (https://arxiv.org/abs/1908.07380). PBB refers to
family of methods to train probabilistic neural networks by minimizing
PAC-Bayes bounds. A script is also provided to run 'pb_quad' on MNIST which
reproduces the results in the PBB paper.

Code to train PAC Bayes Quadratic Bound model on MNIST dataset consists of:

* 'pb_quad.py' -> Library which implements the bound and has utility functions.
* 'pb_quad_trainer.py' -> Script to run on MNIST dataset.
* 'pb_quad_trainer_test.py' -> Smoke test script.

## Running PBB Quadratic bound objective on MNIST

### Setup
```
python3 -m venv pbb_env
source pbb_env/bin/activate
pip install -r pbb_env/requirements.txt
```

### Run Code
```
python3 -m pb_quad_trainer.py
```
## Citation
```

@misc{rivasplata2019pacbayes,
    title={PAC-Bayes with Backprop},
    author={Omar Rivasplata and Vikram M Tankasali and Csaba Szepesvari},
    year={2019},
    eprint={1908.07380},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
