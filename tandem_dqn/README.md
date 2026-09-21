# Tandem DQN

This repository provides an implementation of the Tandem DQN agent and
main experiments presented in the paper
"The Difficulty of Passive Learning in Deep Reinforcement Learning"
(Georg Ostrovski, Pablo Samuel Castro and Will Dabney, 2021).

The code is a modified fork of the [DoubleDQN](https://arxiv.org/abs/1509.06461)
agent in the [DQN Zoo](https://github.com/deepmind/dqn_zoo) agent collection.

## Quick start

This code can be run with a regular CPU setup, but will only be reasonably fast
(with the given configuration) if run with a GPU accelerator, in which case
you will need an NVIDIA GPU with recent CUDA drivers.
For installation, run

```shell
git clone git@github.com:deepmind/deepmind-research.git
virtualenv --python=python3.6 "tandem"
source tandem/bin/activate
cd deepmind_research
pip install -r tandem_dqn/requirements.txt
```

The code has only been tested with Python 3.6.14

## Running the experiments presented in the paper

To execute the vanilla Tandem DQN experiment on the Pong environment, run:

```bash
python -m "tandem_dqn.run_tandem" --environment_name=pong --seed=42
```

A number of flags can be specified to customize execution and run various of
the presented experimental variations:

* `--use_sticky_actions`
(values: `True`, `False`; default: `False`):
Use  "sticky actions" variant
([Machado et al 2017](https://arxiv.org/abs/1709.06009)) of the Atari environment.

* `--network_active`, `--network_passive`
(values: `double_q`, `qr`; default:  `double_q`):
Whether to use the DoubleDQN or QR-DQN
([Dabney et al 2017](https://arxiv.org/abs/1710.10044)) network architecture
for active and passive agents (can be set independently). Note this value
needs to be compatible with the chosen loss (see below).

* `--loss_active`, `--loss_passive`
(values: `double_q`, `double_q_v`, `double_q_p`, `double_q_pv`, `qr`,
`q_regression`; default: `double_q`)
Which loss to use for active and passive agent training. The losses are:
  * `double_q`: regular Double-Q-Learning loss.
  * `double_q_v`: Double-Q-Learning with bootstrap target _values_ provided
  by the respective _other_ agent's target network.
  * `double_q_p`: Double-Q-Learning with bootstrap target _policy_ (argmax)
  provided by the respective _other_ agent's online network.
  * `double_q_pv`: Double-Q-Learning with both boostrap target values and
  policy provided by the respective _other_ agent's target and online networks.
  * `qr`: Quantile Regression Q-Learning loss
  ([Dabney et al 2017](https://arxiv.org/abs/1710.10044)).
  * `q_regression`: Supervised regression loss towards the respective other
  agent's online network's output values.

* `--optimizer_active`, `--optimizer_passive`
(values: `adam`, `rmsprop`; default: `rmsprop` for both):
Which optimization algorithm to use for the active and passive network training
(can be set independently).

* `--exploration_epsilon_end_value`
(values in `[0, 1]`; default: `0.01`):
Value of epsilon parameter in active agent's epsilong-greedy policy after an
initial decay phase.



