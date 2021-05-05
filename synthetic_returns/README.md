# Code for Synthetic Returns

This repository contains code for the arXiv preprint
["Synthetic Returns for Long-Term Credit Assignment"](https://arxiv.org/abs/2102.12425)
by David Raposo, Sam Ritter, Adam Santoro, Greg Wayne, Theophane Weber, Matt
Botvinick, Hado van Hasselt, and Francis Song.

To cite this work:

```
@article{raposo2021synthetic,
  title={Rapid Task-Solving in Novel Environments},
  author={Raposo, David and Ritter, Sam and Santoro, Adam and Wayne, Greg and
  Weber, Theophane and Botvinick, Matt and van Hasselt, Hado and Song, Francis},
  journal={arXiv preprint arXiv:2102.12425},
  year={2021}
}
```

### Agent core wrapper

We implemented the Synthetic Returns module as a wrapper to a recurrent neural
network (RNN), so it should be compatible with any Deep-RL agent with an
arbitrary RNN core, whose inputs consist of batches of vectors. This could be an
LSTM as in the example below, or a more sophisticated core as long as it
implements an `hk.RNNCore`.

```python
agent_core = hk.LSTM(128)
```

To build the SR wrapper, simply pass the existing agent core to the constructor,
along with the SR configuration:

```python
sr_config = {
    "memory_size": 128,
    "capacity": 300,
    "hidden_layers": (128, 128),
    "alpha": 0.3,
    "beta": 1.0,
}
sr_agent_core = hk.ResetCore(
  SyntheticReturnsCoreWrapper(core=agent_core, **sr_config))
```

Typically, the SR wrapper should itself be wrapped in a `hk.ResetCore` in order
to reset the core state in the beginning of a new episode. This will reset not
only the episodic memory but also the original agent core that was passed to the
SR wrapper constructor.

### Learner

Consider the distributed setting, wherein a learner receives mini-batches of
trajectories of length `T` produced by the actors.

`trajectory` is a nested structure of tensors of size `[T,B,...]` (where `B` is
the batch size) containing observations, agent states, rewards and step type
indicators.

We start by producing inputs to the SR core, which consist of tuples of current
state embeddings and return targets. The current state embeddings can be
produced by a ConvNet, for example. In our experiments we used the current step
reward as target. Note that the current step reward correspond to the rewards in
the trajectory shifted by one, relative to the observations:

```python
observations = jax.tree_map(lambda x: x[:-1], trajectory.observation)
vision_output = hk.BatchApply(vision_net)(observations)
return_targets = trajectory.reward[1:]
sr_core_inputs = (vision_output, return_targets)
```

For purposes of core resetting at the beginning of a new episode, we also need
to pass an indicator of which steps correspond to the first step of an episode.

```python
should_reset = jnp.equal(
  trajectory.step_type[:-1], int(dm_env.StepType.FIRST))
core_inputs = (sr_core_inputs, should_reset)
```

We can now produce an unroll using `hk.dynamic_unroll` and passing it the SR
core, the core inputs we produced, and the initial state of the unroll, which
corresponds to the agent state in the first step of the trajectory:

```python
state = jax.tree_map(lambda t: t[0], trajectory.agent_state)
core_output, state = hk.dynamic_unroll(
  sr_agent_core, core_inputs, state)
```

The SR wrapper produces 4 output tensors: the output of the agent core, the
synthetic returns, the SR-augmented return, and the SR loss.

The synthetic returns are taken into account when computing the augmented return
and the SR loss. Therefore they are not needed anymore and can be discarded or
used for logging purposes.

The agent core outputs should be used, as usual, for producing a policy. In an
actor-critic, policy gradient set-up, like IMPALA, we would produce policy
logits and values:

```python
policy_logits = hk.BatchApply(policy_net)(core_output.output)
value = hk.BatchApply(baseline_net)(core_output.output)
```

Similarly, in a Q-learning setting we would use the agent core outputs to
produce q-values.

The SR-augmented returns should be used in place of the environment rewards for
the policy updates (e.g. when computing the policy gradient and baseline
losses):

```python
rewards = core_output.augmented_return
```

Finally, the SR loss, summed over batch and time dimensions, should be added to
the total learner loss to be minimized:

```python
total_loss += jnp.sum(core_output.sr_loss)
```
