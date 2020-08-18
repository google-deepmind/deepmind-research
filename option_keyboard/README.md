# The Option Keyboard: Combining Skills in Reinforcement Learning

This directory contains an implementation of the Option Keyboard framework.

From the [abstract](http://papers.nips.cc/paper/9463-the-option-keyboard-combining-skills-in-reinforcement-learning):

> The ability to combine known skills to create new ones may be crucial in the
solution of complex reinforcement learning problems that unfold over extended
periods. We argue that a robust way of combining skills is to define and manipulate
them in the space of pseudo-rewards (or “cumulants”). Based on this premise, we
propose a framework for combining skills using the formalism of options. We show
that every deterministic option can be unambiguously represented as a cumulant
defined in an extended domain. Building on this insight and on previous results
on transfer learning, we show how to approximate options whose cumulants are
linear combinations of the cumulants of known options. This means that, once we
have learned options associated with a set of cumulants, we can instantaneously
synthesise options induced by any linear combination of them, without any learning
involved. We describe how this framework provides a hierarchical interface to the
environment whose abstract actions correspond to combinations of basic skills.
We demonstrate the practical benefits of our approach in a resource management
problem and a navigation task involving a quadrupedal simulated robot.

If you use the code here please cite this paper

> Andre Barreto, Diana Borsa, Shaobo Hou, Gheorghe Comanici, Eser Aygün, Philippe Hamel, Daniel Toyama, Jonathan hunt, Shibl Mourad, David Silver, Doina Precup.  *The Option Keyboard: Combining Skills in Reinforcement Learning*.  Neurips 2019.  [\[paper\]](https://papers.nips.cc/paper/9463-the-option-keyboard-combining-skills-in-reinforcement-learning).

## Running the code

### Setup
```
python3 -m venv ok_venv
source ok_venv/bin/activate
pip install -r option_keyboard/requirements.txt
```

### Scavenger Task
All agents are trained on a simple grid-world resource collection task. There
are two types of collectible objects in the world: if the agent collects the
object that is less abundant of the two then it receives a reward of -1,
otherwise it receives a reward of +1 when it collects the object. See section
5.1 in the paper for more details.

### Train the DQN baseline
```
python3 -m option_keyboard.run_dqn
```
This trains a DQN agent on the scavenger task.

### Train the Option Keyboard and agent
```
python3 -m option_keyboard.run_ok
```
This first trains an Option Keyboard on the cumulants in the task environment.
Then it trains a DQN agent on the true task reward using high level abstract
actions provided by the keyboard.

## Disclaimer
This is not an official Google or DeepMind product.
