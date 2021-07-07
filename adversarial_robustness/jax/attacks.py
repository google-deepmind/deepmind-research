# Copyright 2021 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adversarial attacks.

This file contains all the code necessary to create untargeted adversarial
attacks in JAX (within an l-infinity ball). For example, to create an untargeted
FGSM attack (with a single step), one can do the following:

```
import attacks

epsilon = 8/255  # Perturbation radius for inputs between 0 and 1.
fgsm_attack = attacks.UntargetedAttack(
    attacks.PGD(
        attacks.IteratedFGSM(epsilon),
        num_steps=1,
        initialize_fn=attacks.linf_initialize_fn(epsilon),
        project_fn=attacks.linf_project_fn(epsilon, bounds=(0., 1.))),
    loss_fn=attacks.untargeted_cross_entropy)
```

Just as elegantly, one can specify an adversarial attack on KL-divergence
to a target distribution (using 10 steps with Adam and a piecewise constant step
schedule):

```
kl_attack_with_adam = attacks.UntargetedAttack(
    attacks.PGD(
        attacks.Adam(optax.piecewise_constant_schedule(
            init_value=.1,
            boundaries_and_scales={5: .1})),
        num_steps=10,
        initialize_fn=attacks.linf_initialize_fn(epsilon),
        project_fn=attacks.linf_project_fn(epsilon, bounds=(0., 1.))),
    loss_fn=attacks.untargeted_kl_divergence)
```

The attack instances can be used later on to build adversarial examples:

```
my_model = ...  # Model. We assume that 'my_model(.)' returns logits.
clean_images, image_labels = ...  # Batch of images and associated labels.
rng = jax.random.PRNGKey(0)  # A random generator state.

adversarial_images = fgsm_attack(my_model, rng, clean_images, image_labels)
```

See `experiment.py` or `eval.py` for more examples.

This file contains the following components:
* Losses:
 * untargeted_cross_entropy: minimizes the likelihood of the label class.
 * untargeted_kl_divergence: maximizes the KL-divergence of the predictions with
   a target distribution.
 * untargeted_margin: maximizes the margin loss (distance from the highest
   non-true logits to the label class logit)
* Step optimizers:
  * SGD: Stochastic Gradient Descent.
  * IteratedFGSM: Also called BIM (see https://arxiv.org/pdf/1607.02533).
  * Adam: See https://arxiv.org/pdf/1412.6980.
* Initialization and projection functions:
  * linf_initialize_fn: Initialize function for l-infinity attacks.
  * linf_project_fn: Projection function for l-infinity attacks.
* Projected Gradient Descent (PGD):
  * PGD: Runs Projected Gradient Descent using the specified optimizer,
    initialization and projection functions for a given number of steps.
* Untargeted attack:
  * UntargetedAttack: Combines PGD and a specific loss function to find
    adversarial examples.
"""

import functools
import inspect
from typing import Callable, Optional, Tuple, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax


ModelFn = Callable[[chex.Array], chex.Array]
LossFn = Callable[[chex.Array], chex.Array]
ClassificationLossFn = Callable[[chex.Array, chex.Array], chex.Array]
OptimizeFn = Callable[[LossFn, chex.PRNGKey, chex.Array], chex.Array]
NormalizeFn = Callable[[chex.Array], chex.Array]
InitializeFn = Callable[[chex.PRNGKey, chex.Array], chex.Array]
ProjectFn = Callable[[chex.Array, chex.Array], chex.Array]


def untargeted_cross_entropy(logits: chex.Array,
                             labels: chex.Array) -> chex.Array:
  """Maximize the cross-entropy of the true class (make it less likely)."""
  num_classes = logits.shape[-1]
  log_probs = jax.nn.log_softmax(logits)
  return jnp.sum(
      hk.one_hot(labels, num_classes).astype(logits.dtype) * log_probs, axis=-1)


def untargeted_kl_divergence(logits: chex.Array,
                             label_probs: chex.Array) -> chex.Array:
  """Maximize the KL divergence between logits and label distribution."""
  # We are explicitly maximizing the cross-entropy, as this is equivalent to
  # maximizing the KL divergence (when `label_probs` does not depend
  # on the values that produce `logits`).
  log_probs = jax.nn.log_softmax(logits)
  return jnp.sum(label_probs * log_probs, axis=-1)


def untargeted_margin(logits: chex.Array,
                      labels: chex.Array) -> chex.Array:
  """Make the highest non-correct logits higher than the true class logits."""
  batch_size = logits.shape[0]
  num_classes = logits.shape[-1]
  label_logits = logits[jnp.arange(batch_size), labels]
  logit_mask = hk.one_hot(labels, num_classes).astype(logits.dtype)
  highest_logits = jnp.max(logits - 1e8 * logit_mask, axis=-1)
  return label_logits - highest_logits


class UntargetedAttack:
  """Performs an untargeted attack."""

  def __init__(self,
               optimize_fn: OptimizeFn,
               loss_fn: ClassificationLossFn = untargeted_cross_entropy):
    """Creates an untargeted attack.

    Args:
      optimize_fn: An `Optimizer` instance or any callable that takes
        a loss function and an initial input and outputs a new input that
        minimizes the loss function.
      loss_fn: `loss_fn` is a surrogate loss. Its goal should be make the true
        class less likely than any other class. Typical options for `loss_fn`
        are `untargeted_cross_entropy` or `untargeted_margin`.
    """
    self._optimize_fn = optimize_fn
    self._loss_fn = loss_fn

  def __call__(self,
               logits_fn: ModelFn,
               rng: chex.PRNGKey,
               inputs: chex.Array,
               labels: chex.Array) -> chex.Array:
    """Returns adversarial inputs."""
    def _loss_fn(x):
      return self._loss_fn(logits_fn(x), labels)
    return self._optimize_fn(_loss_fn, rng, inputs)

  # Convenience functions to detect the type of inputs required by the loss.
  def expects_labels(self):
    return 'labels' in inspect.getfullargspec(self._loss_fn).args

  def expects_probabilities(self):
    return 'label_probs' in inspect.getfullargspec(self._loss_fn).args


class StepOptimizer:
  """Makes a single gradient step that minimizes a loss function."""

  def __init__(self,
               gradient_transformation: optax.GradientTransformation):
    self._gradient_transformation = gradient_transformation

  def init(self,
           loss_fn: LossFn,
           x: chex.Array) -> optax.OptState:
    self._loss_fn = loss_fn
    return self._gradient_transformation.init(x)

  def minimize(
      self,
      x: chex.Array,
      state: optax.OptState) -> Tuple[chex.Array, chex.Array, optax.OptState]:
    """Performs a single minimization step."""
    g, loss = gradients_fn(self._loss_fn, x)
    if g is None:
      raise ValueError('loss_fn does not depend on input.')
    updates, state = self._gradient_transformation.update(g, state, x)
    return optax.apply_updates(x, updates), loss, state


class SGD(StepOptimizer):
  """Vanilla gradient descent optimizer."""

  def __init__(self,
               learning_rate_fn: Union[float, int, optax.Schedule],
               normalize_fn: Optional[NormalizeFn] = None):
    # Accept schedules, as well as scalar values.
    if isinstance(learning_rate_fn, (float, int)):
      lr = float(learning_rate_fn)
      learning_rate_fn = lambda _: lr
    # Normalization.
    def update_fn(updates, state, params=None):
      del params
      updates = jax.tree_map(normalize_fn or (lambda x: x), updates)
      return updates, state
    gradient_transformation = optax.chain(
        optax.GradientTransformation(lambda _: optax.EmptyState(), update_fn),
        optax.scale_by_schedule(learning_rate_fn),
        optax.scale(-1.))
    super(SGD, self).__init__(gradient_transformation)


class IteratedFGSM(SGD):
  """L-infinity normalized steps."""

  def __init__(self,
               learning_rate_fn: Union[float, int, optax.Schedule]):
    super(IteratedFGSM, self).__init__(learning_rate_fn, jnp.sign)


class Adam(StepOptimizer):
  """The Adam optimizer defined in https://arxiv.org/abs/1412.6980."""

  def __init__(
      self,
      learning_rate_fn: Union[float, int, optax.Schedule],
      normalize_fn: Optional[NormalizeFn] = None,
      beta1: float = .9,
      beta2: float = .999,
      epsilon: float = 1e-9):
    # Accept schedules, as well as scalar values.
    if isinstance(learning_rate_fn, (float, int)):
      lr = float(learning_rate_fn)
      learning_rate_fn = lambda _: lr
    # Normalization.
    def update_fn(updates, state, params=None):
      del params
      updates = jax.tree_map(normalize_fn or (lambda x: x), updates)
      return updates, state
    gradient_transformation = optax.chain(
        optax.GradientTransformation(lambda _: optax.EmptyState(), update_fn),
        optax.scale_by_adam(b1=beta1, b2=beta2, eps=epsilon),
        optax.scale_by_schedule(learning_rate_fn),
        optax.scale(-1.))
    super(Adam, self).__init__(gradient_transformation)


class PGD:
  """Runs Project Gradient Descent (see https://arxiv.org/pdf/1706.06083)."""

  def __init__(self,
               optimizer: StepOptimizer,
               num_steps: int,
               initialize_fn: Optional[InitializeFn] = None,
               project_fn: Optional[ProjectFn] = None):
    self._optimizer = optimizer
    if initialize_fn is None:
      initialize_fn = lambda rng, x: x
    self._initialize_fn = initialize_fn
    if project_fn is None:
      project_fn = lambda x, origin_x: x
    self._project_fn = project_fn
    self._num_steps = num_steps

  def __call__(self,
               loss_fn: LossFn,
               rng: chex.PRNGKey,
               x: chex.Array) -> chex.Array:
    def _optimize(rng, x):
      """Optimizes loss_fn when keep_best is False."""
      def body_fn(_, inputs):
        opt_state, current_x = inputs
        current_x, _, opt_state = self._optimizer.minimize(current_x, opt_state)
        current_x = self._project_fn(current_x, x)
        return opt_state, current_x
      opt_state = self._optimizer.init(loss_fn, x)
      current_x = self._project_fn(self._initialize_fn(rng, x), x)
      _, current_x = jax.lax.fori_loop(0, self._num_steps, body_fn,
                                       (opt_state, current_x))
      return current_x
    return jax.lax.stop_gradient(_optimize(rng, x))


def linf_project_fn(epsilon: float, bounds: Tuple[float, float]) -> ProjectFn:
  def project_fn(x, origin_x):
    dx = jnp.clip(x - origin_x, -epsilon, epsilon)
    return jnp.clip(origin_x + dx, bounds[0], bounds[1])
  return project_fn


def linf_initialize_fn(epsilon: float) -> InitializeFn:
  def initialize_fn(rng, x):
    return x + jax.random.uniform(rng, x.shape, minval=-epsilon,
                                  maxval=epsilon).astype(x.dtype)
  return initialize_fn


def gradients_fn(loss_fn: LossFn,
                 x: chex.Array) -> Tuple[chex.Array, chex.Array]:
  """Returns the analytical gradient as computed by `jax.grad`."""
  @functools.partial(jax.grad, has_aux=True)
  def grad_reduced_loss_fn(x):
    loss = loss_fn(x)
    return jnp.sum(loss), loss
  return grad_reduced_loss_fn(x)
