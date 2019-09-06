# Pycolab Tasks

## Playing the Pycolab Tasks

We provide a script to allow human play of the Pycolab tasks. To play, run e.g.

`python3 pycolab/human_player.py -- --game=key_to_door`

## The Pycolab Tasks

There are 2 [Pycolab](https://github.com/deepmind/pycolab) tasks presented here.
Each level is composed of 3 distinct phases. The first phase is the 'explore'
phase, where the agent should learn a piece of information or do something. For
both tasks, the 2nd phase is the 'distractor' phase, where the agent collects
apples for rewards. The 3rd phase is the 'exploit' phase, where the agent gets
rewards based on the knowledge acquired or actions performed in phase 1.

Special thanks to Hamza Merzic for writing these task scripts.

### Active Visual Match

* Phase 1: A colour square randomly placed in a two-connected room.
* Phase 2: Apples collection.
* Phase 3: Choose the colour square matched that in Phase 1 among 4 options.

### Key To Door

* Phase 1: A key randomly placed in a two-connected room.
* Phase 2: Apples collection.
* Phase 3: A small room with a door. If agent has key, it can open the door to
           get to the goal behind the door to get reward.
