# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for side_effects_penalty."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from six.moves import range
from side_effects_penalties import side_effects_penalty
from side_effects_penalties import training
from side_effects_penalties.side_effects_penalty import Actions


environments = ['box', 'vase', 'sushi_goal']


class SideEffectsTestCase(parameterized.TestCase):

  def _timestep_to_state(self, timestep):
    return tuple(map(tuple, np.copy(timestep.observation['board'])))

  def _env_to_action_range(self, env):
    action_spec = env.action_spec()
    action_range = list(range(action_spec.minimum, action_spec.maximum + 1))
    return action_range


class BaselineTestCase(SideEffectsTestCase):

  def _create_baseline(self, env_name):
    self._env, _ = training.get_env(env_name, True)
    self._baseline_env, _ = training.get_env(env_name, True)
    baseline_class = getattr(side_effects_penalty,
                             self.__class__.__name__[:-4])  # remove 'Test'
    self._baseline = baseline_class(
        self._env.reset(), True, self._baseline_env, self._timestep_to_state)

  def _test_trajectory(self, actions, key):
    init_state = self._timestep_to_state(self._env.reset())
    self._baseline.reset()
    current_state = init_state
    for action in actions:
      timestep = self._env.step(action)
      next_state = self._timestep_to_state(timestep)
      baseline_state = self._baseline.calculate(current_state, action,
                                                next_state)
      comparison_dict = {
          'current_state': current_state,
          'next_state': next_state,
          'init_state': init_state
      }
      self.assertEqual(baseline_state, comparison_dict[key])
      current_state = next_state
      if timestep.last():
        return


class StartBaselineTest(BaselineTestCase):

  @parameterized.parameters(*environments)
  def testInit(self, env_name):
    self._create_baseline(env_name)
    self._test_trajectory([Actions.NOOP], 'init_state')

  @parameterized.parameters(*environments)
  def testTenNoops(self, env_name):
    self._create_baseline(env_name)
    self._test_trajectory([Actions.NOOP for _ in range(10)], 'init_state')


class InactionBaselineTest(BaselineTestCase):

  box_env, _ = training.get_env('box', True)
  box_action_spec = box_env.action_spec()

  @parameterized.parameters(
      *list(range(box_action_spec.minimum, box_action_spec.maximum + 1)))
  def testStaticEnvOneAction(self, action):
    self._create_baseline('box')
    self._test_trajectory([action], 'init_state')

  def testStaticEnvRandomActions(self):
    self._create_baseline('box')
    num_steps = np.random.randint(low=1, high=20)
    action_range = self._env_to_action_range(self._env)
    actions = [np.random.choice(action_range) for _ in range(num_steps)]
    self._test_trajectory(actions, 'init_state')

  @parameterized.parameters(*environments)
  def testInactionPolicy(self, env_name):
    self._create_baseline(env_name)
    num_steps = np.random.randint(low=1, high=20)
    self._test_trajectory([Actions.NOOP for _ in range(num_steps)],
                          'next_state')


class StepwiseBaselineTest(BaselineTestCase):

  def testStaticEnvRandomActions(self):
    self._create_baseline('box')
    action_range = self._env_to_action_range(self._env)
    num_steps = np.random.randint(low=1, high=20)
    actions = [np.random.choice(action_range) for _ in range(num_steps)]
    self._test_trajectory(actions, 'current_state')

  @parameterized.parameters(*environments)
  def testInactionPolicy(self, env_name):
    self._create_baseline(env_name)
    num_steps = np.random.randint(low=1, high=20)
    self._test_trajectory([Actions.NOOP for _ in range(num_steps)],
                          'next_state')

  @parameterized.parameters(*environments)
  def testInactionRollout(self, env_name):
    self._create_baseline(env_name)
    init_state = self._timestep_to_state(self._env.reset())
    self._baseline.reset()
    action = Actions.NOOP
    state1 = init_state
    trajectory = [init_state]
    for _ in range(10):
      trajectory.append(self._timestep_to_state(self._env.step(action)))
      state2 = trajectory[-1]
      self._baseline.calculate(state1, action, state2)
      state1 = state2
    chain = self._baseline.rollout_func(init_state)
    self.assertEqual(chain, trajectory[:len(chain)])
    if len(chain) < len(trajectory):
      self.assertEqual(trajectory[len(chain) - 1], trajectory[len(chain)])

  def testStaticRollouts(self):
    self._create_baseline('box')
    action_range = self._env_to_action_range(self._env)
    num_steps = np.random.randint(low=1, high=20)
    actions = [np.random.choice(action_range) for _ in range(num_steps)]
    state1 = self._timestep_to_state(self._env.reset())
    states = [state1]
    self._baseline.reset()
    for action in actions:
      state2 = self._timestep_to_state(self._env.step(action))
      states.append(state2)
      self._baseline.calculate(state1, action, state2)
      state1 = state2
    i1, i2 = np.random.choice(len(states), 2)
    chain = self._baseline.parallel_inaction_rollouts(states[i1], states[i2])
    self.assertLen(chain, 1)
    chain1 = self._baseline.rollout_func(states[i1])
    self.assertLen(chain1, 1)
    chain2 = self._baseline.rollout_func(states[i2])
    self.assertLen(chain2, 1)

  @parameterized.parameters(('parallel', 'vase'), ('parallel', 'sushi'),
                            ('inaction', 'vase'), ('inaction', 'sushi'))
  def testConveyorRollouts(self, which_rollout, env_name):
    self._create_baseline(env_name)
    init_state = self._timestep_to_state(self._env.reset())
    self._baseline.reset()
    action = Actions.NOOP
    state1 = init_state
    init_state_next = self._timestep_to_state(self._env.step(action))
    state2 = init_state_next
    self._baseline.calculate(state1, action, state2)
    state1 = state2
    for _ in range(10):
      state2 = self._timestep_to_state(self._env.step(action))
      self._baseline.calculate(state1, action, state2)
      state1 = state2
    if which_rollout == 'parallel':
      chain = self._baseline.parallel_inaction_rollouts(init_state,
                                                        init_state_next)
    else:
      chain = self._baseline.rollout_func(init_state)
    self.assertLen(chain, 5)


class NoDeviationTest(SideEffectsTestCase):

  def _random_initial_transition(self):
    env_name = np.random.choice(environments)
    noops = np.random.choice([True, False])
    env, _ = training.get_env(env_name, noops)
    action_range = self._env_to_action_range(env)
    action = np.random.choice(action_range)
    state1 = self._timestep_to_state(env.reset())
    state2 = self._timestep_to_state(env.step(action))
    return (state1, state2)

  def testNoDeviation(self):
    deviation = side_effects_penalty.NoDeviation()
    state1, state2 = self._random_initial_transition()
    self.assertEqual(deviation.calculate(state1, state2), 0)

  def testNoDeviationUpdate(self):
    deviation = side_effects_penalty.NoDeviation()
    state1, state2 = self._random_initial_transition()
    deviation.update(state1, state2)
    self.assertEqual(deviation.calculate(state1, state2), 0)


class UnreachabilityTest(SideEffectsTestCase):

  @parameterized.named_parameters(('Discounted', 0.99), ('Undiscounted', 1.0))
  def testUnreachabilityCycle(self, gamma):
    # Reachability with no dev_fun means unreachability
    deviation = side_effects_penalty.Reachability(value_discount=gamma)
    env, _ = training.get_env('box', False)

    state0 = self._timestep_to_state(env.reset())
    state1 = self._timestep_to_state(env.step(Actions.LEFT))
    # deviation should not be calculated before calling update

    deviation.update(state0, state1)
    self.assertEqual(deviation.calculate(state0, state0), 1.0 - 1.0)
    self.assertEqual(deviation.calculate(state0, state1), 1.0 - gamma)
    self.assertEqual(deviation.calculate(state1, state0), 1.0 - 0.0)

    state2 = self._timestep_to_state(env.step(Actions.RIGHT))
    self.assertEqual(state0, state2)

    deviation.update(state1, state2)
    self.assertEqual(deviation.calculate(state0, state0), 1.0 - 1.0)
    self.assertEqual(deviation.calculate(state0, state1), 1.0 - gamma)
    self.assertEqual(deviation.calculate(state1, state0), 1.0 - gamma)
    self.assertEqual(deviation.calculate(state1, state1), 1.0 - 1.0)


if __name__ == '__main__':
  absltest.main()
