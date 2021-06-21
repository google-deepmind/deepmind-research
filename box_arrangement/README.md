# Predicate tasks.

This package contains tasks associated with "Behavior Priors for Efficient
Reiforcement Learning" (https://arxiv.org/abs/2010.14274), "Exploiting Hierarchy
for Learning and Transfer in KL-Regularized RL" (https://arxiv.org/abs/2010.14274)
and "Information asymmetry in KL-regularized RL"
(https://arxiv.org/abs/1905.01240).
This is research code, and has dependencies on more stable code that is
available as part of [`dm_control`], in particular upon components in
[`dm_control.locomotion`] and [`dm_control.manipulation`].

To get access to preconfigured python environments for the tasks, see the
`task_examples.py` file. To use the MuJoCo interactive viewer (from dm_control)
to load the environments, see `explore.py`.

<p float="left">
  <img src="tasks.png" height="200">
</p>

## Installation instructions

1.  Download [MuJoCo Pro](https://mujoco.org/) and extract the zip archive as
    `~/.mujoco/mujoco200_$PLATFORM` where `$PLATFORM` is one of `linux`,
    `macos`, or `win64`.

2.  Ensure that a valid MuJoCo license key file is located at
    `~/.mujoco/mjkey.txt`.

3.  Clone the `deepmind-research` repository:

    ```shell
       git clone https://github.com/deepmind/deepmind-research.git
       cd deepmind-research
    ```

4.  Create and activate a Python virtual environment:

    ```shell
       python3 -m virtualenv box_arrangement
       source box_arrangement/bin/activate
    ```

5.  Install the package:

    ```shell
       pip install ./box_arrangement
    ```

## Quickstart

To instantiate and step through the go to one of K targets task:

```python
from box_arrangement import task_examples
import numpy as np

# Build an example environment.
env = task_examples.go_to_k_targets()

# Get the `action_spec` describing the control inputs.
action_spec = env.action_spec()

# Step through the environment for one episode with random actions.
time_step = env.reset()
while not time_step.last():
  action = np.random.uniform(action_spec.minimum, action_spec.maximum,
                             size=action_spec.shape)
  time_step = env.step(action)
  print("reward = {}, discount = {}, observations = {}.".format(
      time_step.reward, time_step.discount, time_step.observation))
```

The above code snippet can also be used for other tasks by replacing
`go_to_k_targets` with one of (`move_box`, `move_box_or_gtt` and
`move_box_and_gtt`).

## Visualization

[`dm_control.viewer`] can be used to visualize and interact with the
environment. We provide the `explore.py` script specifically for this. If you
followed our installation instructions above, this can be launched for the
go to one of K targets task via:

```shell
python3 -m box_arrangement.explore --task='go_to_target'
```

## Citation

If you use the code or data in this package, please cite:

```
@misc{tirumala2020behavior,
      title={Behavior Priors for Efficient Reinforcement Learning},
      author={Dhruva Tirumala and Alexandre Galashov and Hyeonwoo Noh and Leonard Hasenclever and Razvan Pascanu and Jonathan Schwarz and Guillaume Desjardins and Wojciech Marian Czarnecki and Arun Ahuja and Yee Whye Teh and Nicolas Heess},
      year={2020},
      eprint={2010.14274},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

[`dm_control`]: https://github.com/deepmind/dm_control
[`dm_control.locomotion`]: https://github.com/deepmind/dm_control/tree/master/dm_control/locomotion
[`dm_control.manipulation`]: https://github.com/deepmind/dm_control/tree/master/dm_control/manipulation
[`dm_control.viewer`]: https://github.com/deepmind/dm_control/tree/master/dm_control/viewer
