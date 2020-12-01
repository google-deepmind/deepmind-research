# Physically Embedded Planning Environments

This repository contains the three environments introduced in
'Physically Embedded Planning Problems: New Challenges for Reinforcement
Learning'

If you use this package, please cite our accompanying [tech report]:

```
@misc{mirza2020physically,
    title={Physically Embedded Planning Problems: New Challenges for Reinforcement Learning},
    author={Mehdi Mirza and Andrew Jaegle and Jonathan J. Hunt and Arthur Guez and Saran Tunyasuvunakool and Alistair Muldal and Théophane Weber and Peter Karkus and Sébastien Racanière and Lars Buesing and Timothy Lillicrap and Nicolas Heess},
    year={2020},
    eprint={2009.05524},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

## Requirements and Installation

This repository is divided into 'mujoban' and 'board_games' folders.
Both of them are built on top of [dm_control] which requires MuJoCo. Please
follow [these] instructions to install MuJoCo.
Other dependencies can be installed
by:
```
pip3 install -r requirements.txt
```
### Board games
The game logic is based on [open_spiel]. Please install as instructed [here].
[gnugo] is required to play the game of Go against a non-random opponent. [gnugo] can be installed in Ubuntu by:
```
apt install gnugo
```

Board game scripts expect gnugo binary to be at: `/usr/games/gnugo`. Users can
change this path inside `board_games/go_logic.py`

This library has only been tested on Ubuntu.

## Example usage

The code snippets below show examples of instantiating each of the environments.

### Mujoban

```python
from dm_control import composer
from dm_control.locomotion import walkers
from physics_planning_games.mujoban.mujoban import Mujoban
from physics_planning_games.mujoban.mujoban_level import MujobanLevel
from physics_planning_games.mujoban.boxoban import boxoban_level_generator

walker = walkers.JumpingBallWithHead(add_ears=True, camera_height=0.25)
maze = MujobanLevel(boxoban_level_generator)
task = Mujoban(walker=walker,
               maze=maze,
               control_timestep=0.1,
               top_camera_height=96,
               top_camera_width=96)
env = composer.Environment(time_limit=1000, task=task)
```

### Board games

```python
from  physics_planning_games  import  board_games

environment_name = 'go_7x7'
env = board_games.load(environment_name=environment_name)
```

### Stepping through environment.

The returned environments are of type of `dm_env.Environment` and can be stepped
through as shown here with random actions:

```python
import numpy as np

timestep = env.reset()
action_spec = env.action_spec()
while True:
  action = np.stack([
      np.random.uniform(low=minimum, high=maximum)
      for minimum, maximum in zip(action_spec.minimum, action_spec.maximum)
  ])
  timestep = env.step(action)
```

### Visualization

For visualization of the environments `explore.py` loads them using the [viewer]
from [dm_control].

## More details

For more details please refer to the [tech report], [dm_control] and [dm_env].

[tech report]: https://arxiv.org/abs/2009.05524
[dm_control]: https://github.com/deepmind/dm_control
[dm_env]: https://github.com/deepmind/dm_env
[gnugo]: https://www.gnu.org/software/gnugo/
[open_spiel]: https://github.com/deepmind/open_spiel
[here]: https://github.com/deepmind/open_spiel/blob/master/docs/install.md
[these]: https://github.com/deepmind/dm_control#requirements-and-installation
[viewer]: https://github.com/deepmind/dm_control/tree/master/dm_control/viewer
