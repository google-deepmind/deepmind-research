# DM Lab Tasks

## General Structure

There are 7 [DM Lab](https://github.com/deepmind/lab) tasks presented here.
Each level is composed of 3 distinct phases (except `Key To Door To Match`
which has 5 phases). The first phase is the 'explore' phase, where the agent
should learn a piece of information or do something. For all tasks, the 2nd
phase is the 'distractor' phase, where the agent collects apples for rewards.
The 3rd phase is the 'exploit' phase, where the agent gets rewards based on the
knowledge acquired or actions performed in phase 1.

## Specific Tasks

### Passive Visual Match

* Phase 1: A colour square right in front of the agent.
* Phase 2: Apples collection.
* Phase 3: Choose the colour square matched that in Phase 1 among 4 options.

### Active Visual Match

* Phase 1: A colour square randomly placed in a two-connected room.
* Phase 2: Apples collection.
* Phase 3: Choose the colour square matched that in Phase 1 among 4 options.

### Key To Door

* Phase 1: A key randomly placed in a two-connected room.
* Phase 2: Apples collection.
* Phase 3: A small room with a door. If agent has key, it can open the door to
           get to the goal behind the door to get reward.

### Key To Door Bluekey

All the same as key_to_door above but the key has a blue colour instead of
black.

### Two Negative Keys

* Phase 1: A blue and a red key placed in a small room. The agent can only
           pick up one of the key.
* Phase 2: Apples collection.
* Phase 3: A small room with a door. If agent has either key, it can open the
           door to get reward. The reward depends on which key it got in Phase 1
           All the rewards are negative in this level.

### Latent Information Acquisition

* Phase 1: Thre randomly sampled objects are randomly placed in a small room.
           When the agent touch each object, a red or green cue will appear,
           indicating the reward it is associated in this episode. No rewards
           are given in this phase.
* Phase 2: Apples collection.
* Phase 3: The same three objects in Phase 1 randomly placed again in the room.
           The agent will get positive rewards if pick up the objects with green
           cues in Phase 1, and get negative rewards for objects with red cues.

### Key To Door To Match

* Phase 1: A key is randomly placed in a room. Agent could pick it up.
* Phase 2: Apples collection.
* Phase 3: A colour square behind a door. If agent has key from Phase 1, it can
           open the door to see the colour.
* Phase 4: Apples collection.
* Phase 5: Chose the colour square matched that in Phase 3 among 4 options.
