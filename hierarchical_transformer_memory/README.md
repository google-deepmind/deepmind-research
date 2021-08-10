# Towards mental time travel: A hierarchical memory for RL agents

This provides an implementation of two components of the paper "Towards mental
time travel: A hierarchical memory for reinforcement learning agents." The
article can be found on arXiv at [https://arxiv.org/abs/2105.14039](https://arxiv.org/abs/2105.14039)

Specifically, this repository contains:

1) A JAX/Haiku implementation of hierarchical transformer attention over memory.
2) An implementation of the Ballet environment used in the paper.

We have also released the Rapid Word Learning tasks from the paper, but to
simplify dependencies they are located in the `dm_fast_mapping` repository:
[deepmind/dm_fast_mapping](https://github.com/deepmind/dm_fast_mapping) see the
[documentation](https://github.com/deepmind/dm_fast_mapping/blob/master/docs/index.md)
for that repository for further details about using those tasks.

## Setup

For easy installation, run:

```shell
python3 -m venv htm_env
source htm_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note that this installs the components needed for both the attention module and
the environment. If you only wish to use the environment, you do not need to
install JAX, Haiku, or Chex.

## Using the hierarchical attention module:

Please see `hierarchical_attention/htm_attention_test.py` for some examples of
the expected inputs for this module.

## Running the ballet environment

The ballet environment is contained in the `pycolab_ballet/` subfolder. To load
a simple ballet environment with 2 dancers and short delays, and watch a few
steps of the dances, you can do:

```
from pycolab_ballet import ballet_environment

env = ballet_environment.simple_builder(level_name='2_delay16')
timestep = env.reset()
for _ in range(5):
  action = 0
  timestep = env.step(action)
```

## Citing this work

If you use this code, please cite the associated paper:

```
@article{lampinen2021towards,
  title={Towards mental time travel:
         a hierarchical memory for reinforcement learning agents},
  author={Lampinen, Andrew Kyle and Chan, Stephanie CY and Banino, Andrea and
          Hill, Felix},
  journal={arXiv preprint arXiv:2105.14039},
  year={2021}
}
```
