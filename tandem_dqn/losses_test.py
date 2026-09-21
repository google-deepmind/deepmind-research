# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for Tandem losses."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.config import config
import numpy as np

from tandem_dqn import agent
from tandem_dqn import losses
from tandem_dqn import networks
from tandem_dqn import replay


def make_tandem_qvals():
  return agent.TandemTuple(
      active=networks.QNetworkOutputs(3. * np.ones((3, 5), np.float32)),
      passive=networks.QNetworkOutputs(2. * np.ones((3, 5), np.float32))
  )


def make_transition():
  return replay.Transition(
      s_tm1=np.zeros(3), a_tm1=np.ones(3, np.int32), r_t=5. * np.ones(3),
      discount_t=0.9 * np.ones(3), s_t=np.zeros(3))


class DoubleQLossesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.qs = make_tandem_qvals()
    self.transition = make_transition()
    self.rng_key = jax.random.PRNGKey(42)

  @chex.all_variants()
  @parameterized.parameters(
      ('double_q',), ('double_q_v',), ('double_q_p',), ('double_q_pv',),
      ('q_regression',),
  )
  def test_active_loss_gradients(self, loss_type):
    loss_fn = losses.make_loss_fn(loss_type, active=True)
    def fn(q_tm1, q_t, q_t_target, transition, rng_key):
      return loss_fn(q_tm1, q_t, q_t_target, transition, rng_key)
    grad_fn = self.variant(jax.grad(fn, argnums=(0, 1, 2)))

    dldq_tm1, dldq_t, dldq_t_target = grad_fn(
        self.qs, self.qs, self.qs, self.transition, self.rng_key)
    # Assert that only active net gets nonzero gradients.
    self.assertGreater(np.sum(np.abs(dldq_tm1.active.q_values)), 0.)
    self.assertTrue(np.all(dldq_t.active.q_values == 0.))
    self.assertTrue(np.all(dldq_t_target.active.q_values == 0.))
    self.assertTrue(np.all(dldq_t.passive.q_values == 0.))
    self.assertTrue(np.all(dldq_tm1.passive.q_values == 0.))
    self.assertTrue(np.all(dldq_t_target.passive.q_values == 0.))

  @chex.all_variants()
  @parameterized.parameters(
      ('double_q',), ('double_q_v',), ('double_q_p',), ('double_q_pv',),
      ('q_regression',),
  )
  def test_passive_loss_gradients(self, loss_type):
    loss_fn = losses.make_loss_fn(loss_type, active=False)
    def fn(q_tm1, q_t, q_t_target, transition, rng_key):
      return loss_fn(q_tm1, q_t, q_t_target, transition, rng_key)
    grad_fn = self.variant(jax.grad(fn, argnums=(0, 1, 2)))

    dldq_tm1, dldq_t, dldq_t_target = grad_fn(
        self.qs, self.qs, self.qs, self.transition, self.rng_key)
    # Assert that only passive net gets nonzero gradients.
    self.assertGreater(np.sum(np.abs(dldq_tm1.passive.q_values)), 0.)
    self.assertTrue(np.all(dldq_t.passive.q_values == 0.))
    self.assertTrue(np.all(dldq_t_target.passive.q_values == 0.))
    self.assertTrue(np.all(dldq_t.active.q_values == 0.))
    self.assertTrue(np.all(dldq_tm1.active.q_values == 0.))
    self.assertTrue(np.all(dldq_t_target.active.q_values == 0.))


if __name__ == '__main__':
  config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
