# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Optimizers and Schedulers, inspired by the PyTorch API."""
from collections import ChainMap  # pylint:disable=g-importing-member
from typing import Callable
import haiku as hk
import jax
import jax.numpy as jnp
import tree
from nfnets import utils


class Optimizer(object):
  """Optimizer base class."""

  def __init__(self, params, defaults):
    # Flag indicating if parameters have been broadcasted
    self._broadcasted = False
    # Optimizer hyperparameters; this is a dict to support using param_groups
    self._hyperparameters = {}
    # Mapping from model parameters to optimizer hyperparameters
    self._params2hyperparams = {}
    # Assign defaults
    self._hyperparameters = dict(**defaults)
    # Prepare parameter groups and mappings
    self.create_param_groups(params, defaults)
    # Join params at top-level if params is a list of groups
    if isinstance(params, list):
      flatmap = type(hk.data_structures.to_immutable_dict({}))
      if any([isinstance(group['params'], flatmap) for group in params]):
        params = hk.data_structures.merge(*[group['params']
                                            for group in params])
      else:
        params = dict(ChainMap(*[group['params'] for group in params]))
    # Prepare states
    create_buffers = lambda k, v: self.create_buffers('/'.join(k), v)
    self._states = tree.map_structure_with_path(create_buffers, params)

  def add_hyperparam_group(self, group, suffix, defaults):
    """Adds new hyperparameters to the hyperparams dict."""
    # Use default hyperparams unless overridden by group hyperparams
    group_dict = {key: key for key in defaults if key not in group}
    for key in group:
      if key != 'params':  # Reserved keyword 'params'
        group_dict[key] = '%s_%s' % (key, suffix)
        self._hyperparameters[group_dict[key]] = group[key]
    # Set up params2hyperparams
    def set_p2h(k, _):
      self._params2hyperparams['/'.join(k)] = group_dict
    tree.map_structure_with_path(set_p2h, group['params'])

  def create_param_groups(self, params, defaults):
    """Creates param-hyperparam mappings."""
    if isinstance(params, list):
      for group_index, group in enumerate(params):
        # Add group to hyperparams and get this group's full hyperparameters
        self.add_hyperparam_group(group, group_index, defaults)
    else:
      mapping = {key: key for key in self._hyperparameters}
      def set_p2h(k, _):
        self._params2hyperparams['/'.join(k)] = mapping
      tree.map_structure_with_path(set_p2h, params)

  def create_buffers(self, name, params):
    """Method to be overridden by child classes."""
    pass

  def get_opt_params(self, param_name, itr):
    """Returns hyperparams corresponding to param_name."""
    mapping = self._params2hyperparams[param_name]
    output = {}
    for key in mapping:
      hyper = self._hyperparameters[mapping[key]]
      # Handle the case where a hyper is a class, for hybrids
      if isinstance(hyper, Callable) and not isinstance(hyper, type):
        output[key] = hyper(itr)
      else:
        output[key] = hyper
    return output

  def get_hyper(self, param_name, hyper_name):
    """Get an individual hyperparam for a given param."""
    mapping = self._params2hyperparams[param_name]
    return self._hyperparameters[mapping[hyper_name]]

  def plugin(self, states):
    self._states = states

  def states(self):
    return self._states

  def broadcast(self):
    """Brodcasts all buffers and parameters."""
    self._broadcasted = True
    for name, state in self._states.items():
      self._states[name] = {key: utils.broadcast(state[key]) for key in state}

  def gather(self):
    """Gathers state (if broadcasted) for saving."""
    states = {}
    for name in self._states:
      state = self._states[name]
      states[name] = {key: state[key] if state[key] is None else state[key][0]
                      for key in state}
    return states

  def __setattr__(self, name, value):
    """Overrides the object's set-attribute function to register states, etc."""
    if '_hyperparameters' in self.__dict__ and name in self._hyperparameters:
      self._hyperparameters[name] = value
    elif '_states' in self.__dict__ and name in self._states:
      self._states[name] = value
    else:
      object.__setattr__(self, name, value)

  def __getattr__(self, name):
    """Override the object's get-attribute function to return states, etc."""
    if '_hyperparameters' in self.__dict__ and name in self._hyperparameters:
      return self._hyperparameters[name]
    elif '_states' in self.__dict__ and name in self._states:
      return self._states[name]
    else:
      object.__getattr__(self, name)

  def step(self, params, grads, states, itr=None):
    """Takes a single optimizer step.

    Args:
      params: a dict containing the parameters to be updated.
      grads: a dict containing the gradients for each parameter in params.
      states: a dict containing any optimizer buffers (momentum, etc) for
        each parameter in params.
      itr: an optional integer indicating the current step, for scheduling.
    Returns:
       The updated params and optimizer buffers.
    """
    get_hyper = lambda k, v: self.get_opt_params('/'.join(k), itr)
    hypers = tree.map_structure_with_path(get_hyper, params)
    outs = tree.map_structure_up_to(params, self.update_param,
                                    params, grads, states, hypers)
    return utils.split_tree(outs, params, 2)


class Schedule(object):
  """Hyperparameter scheduling objects."""


class CosineDecay(Schedule):
  """Cosine decay."""

  def __init__(self, min_val, max_val, num_steps):
    self.min_val = min_val
    self.max_val = max_val
    self.num_steps = num_steps

  def __call__(self, itr):
    cos = (1 + jnp.cos(jnp.pi * itr / self.num_steps))
    return 0.5 * (self.max_val - self.min_val) * cos + self.min_val


class WarmupCosineDecay(Schedule):
  """Cosine decay with linear warmup."""

  def __init__(self, start_val, min_val, max_val, num_steps, warmup_steps):
    self.start_val = start_val
    self.min_val = min_val
    self.max_val = max_val
    self.num_steps = num_steps
    self.warmup_steps = warmup_steps

  def __call__(self, itr):
    warmup_val = ((self.max_val - self.start_val) * (itr / self.warmup_steps)
                  + self.start_val)
    cos_itr = (itr - self.warmup_steps) / (self.num_steps - self.warmup_steps)
    cos = 1 + jnp.cos(jnp.pi * cos_itr)
    cos_val = 0.5 * (self.max_val - self.min_val) * cos + self.min_val
    # Select warmup_val if itr < warmup, else cosine val
    values = jnp.array([warmup_val, cos_val])
    index = jnp.sum(jnp.array(self.warmup_steps) < itr)
    return jnp.take(values, index)


class WarmupExpDecay(Schedule):
  """Exponential step decay with linear warmup."""

  def __init__(self, start_val, max_val, warmup_steps,
               decay_factor, decay_interval):
    self.start_val = start_val
    self.max_val = max_val
    self.warmup_steps = warmup_steps
    self.decay_factor = decay_factor
    self.decay_interval = decay_interval

  def __call__(self, itr):
    warmup_val = ((self.max_val - self.start_val) * (itr / self.warmup_steps)
                  + self.start_val)
    # How many decay steps have we taken?
    num_decays = jnp.floor((itr - self.warmup_steps) / self.decay_interval)
    exp_val = self.max_val * (self.decay_factor ** num_decays)
    # Select warmup_val if itr < warmup, else exp_val
    values = jnp.array([warmup_val, exp_val])
    index = jnp.sum(jnp.array(self.warmup_steps) < itr)
    return jnp.take(values, index)


class SGD(Optimizer):
  """Standard SGD with (nesterov) momentum and weight decay.

  Attributes:
    params: Either a dict mapping param names to JAX tensors, or a list where
      each member of the list is a dict containing parameters
      and hyperparameters, allowing one to specify param-specific hyperparams.
    lr: Learning rate.
    weight_decay: Weight decay parameter. Note that this is decay, not L2 reg.
    momentum: Momentum parameter
    dampening: Dampening parameter
    nesterov: Bool indicating this optimizer will use the NAG formulation.
  """
  defaults = {'weight_decay': None, 'momentum': None, 'dampening': 0,
              'nesterov': None}

  def __init__(self, params, lr, weight_decay=None,
               momentum=None, dampening=0, nesterov=None):
    super().__init__(
        params, defaults={'lr': lr, 'weight_decay': weight_decay,
                          'momentum': momentum, 'dampening': dampening,
                          'nesterov': nesterov})

  def create_buffers(self, name, param):
    """Prepares all momentum buffers for each parameter."""
    state = {'step': jnp.zeros(jax.local_device_count())}
    if self.get_hyper(name, 'momentum') is not None:
      state['momentum'] = jnp.zeros_like(param)
    return state

  def update_param(self, param, grad, state, opt_params):
    """The actual update step for this optimizer."""
    if param is None:
      return param, state
    # Apply weight decay
    if opt_params.get('weight_decay') is not None:
      grad = grad + param * opt_params['weight_decay']
    # Update momentum buffers if needed
    if 'momentum' in state:
      state['momentum'] = (opt_params['momentum'] * state['momentum']
                           + (1 - opt_params['dampening']) * grad)
      if opt_params['nesterov'] is not None:
        grad = grad + opt_params['momentum'] * state['momentum']
      else:
        grad = state['momentum']
    state['step'] += 1
    return param - opt_params['lr'] * grad, state


class Adam(Optimizer):
  """Adam optimizer, Kingma & Ba, arxiv.org/abs/1412.6980.

   Args:
      params (iterable): nested list of params to optimize
      lr (float, optional): learning rate (default: 1e-3)
      betas (Tuple[float, float], optional): coefficients used for computing
          running averages of gradient and its square (default: (0.9, 0.999))
      eps (float, optional): term added to the denominator to improve
          numerical stability (default: 1e-8)
      weight_decay (float, optional): weight decay (default: 0)
      use_adamw (bool, optional): If not None, use decoupled weight decay
          as in arxiv.org/abs/1711.05101. The paper version adds an additional
          "schedule" hyperparameter eta, which we instead just replace with the
          learning rate following the PyTorch implementation.

  Note that this implementation will not instantiate a buffer if the
  beta term for that buffer is passed in as None, thus conserving memory.
  """
  defaults = {'beta1': 0.9, 'beta2': 0.999, 'weight_decay': None, 'eps': 1e-8,
              'use_adamw': None}

  def __init__(self, params, lr, beta1=0.9, beta2=0.999,
               eps=1e-8, weight_decay=None, use_adamw=None):
    super().__init__(params=params,
                     defaults={'lr': lr, 'beta1': beta1,
                               'beta2': beta2, 'eps': eps,
                               'weight_decay': weight_decay,
                               'use_adamw': use_adamw})

  def create_buffers(self, name, param):
    """Prepare exp_avg and exp_avg_sq buffers."""
    state = {'step': jnp.zeros(jax.local_device_count())}
    if self.get_hyper(name, 'beta1') is not None:
      state['exp_avg'] = jnp.zeros_like(param)
    if self.get_hyper(name, 'beta2') is not None:
      state['exp_avg_sq'] = jnp.zeros_like(param)
    return state

  def update_param(self, param, grad, state, opt_params):
    """The actual update step for this optimizer."""
    if param is None:
      return param, state
    state['step'] = state['step'] + 1
    # Apply weight decay
    if opt_params.get('weight_decay') is not None:
      if opt_params.get('use_adamw') is not None:
        param = param * (1 - opt_params['lr'] * opt_params['weight_decay'])
      else:
        grad = grad + param * opt_params['weight_decay']
    # First moment
    if 'exp_avg' in state:
      bias_correction1 = 1 - opt_params['beta1'] ** state['step']
      state['exp_avg'] = (state['exp_avg'] * opt_params['beta1']
                          + (1 - opt_params['beta1']) * grad)
      step_size = opt_params['lr'] * state['exp_avg'] / bias_correction1
    else:
      step_size = opt_params['lr'] * grad
    # Second moment
    if 'exp_avg_sq' in state:
      bias_correction2 = 1 - opt_params['beta2'] ** state['step']
      state['exp_avg_sq'] = (state['exp_avg_sq'] * opt_params['beta2']
                             + (1 - opt_params['beta2']) * grad * grad)
      denom = jnp.sqrt(state['exp_avg_sq']) * jax.lax.rsqrt(bias_correction2)
      denom = denom + opt_params['eps']
    else:
      denom = jnp.abs(grad) + opt_params['eps']  # Add eps to avoid divide-by-0

    return param - step_size / denom, state


class RMSProp(Optimizer):
  """RMSProp optimizer, Tieleman and Hinton, ref: powerpoint slides.

    Implements RMSProp as
    rms = decay * rms{t-1} + (1-decay) * gradient ** 2
    mom = momentum * mom{t-1} + learning_rate * g_t / sqrt(rms + epsilon)
    param -= mom

    Note that the rms buffer is initialized with ones as in TF, as opposed to
    zeros as in all other implementations.

   Args:
      params (iterable): nested list of params to optimize
      lr (float): learning rate (default: 1e-3)
      decay (float): EMA decay rate for running estimate of squared gradient.
      momentum (float or None): Use heavy ball momentum instead of instant grad.
      eps (float, optional): term added to the denominator to improve
          numerical stability (default: 1e-8)
      weight_decay (float, optional): weight decay (NOT ADAMW (default: 0))
  """
  defaults = {'weight_decay': None, 'eps': 1e-8}

  def __init__(self, params, lr, decay, momentum, weight_decay=None, eps=1e-8):
    super().__init__(params=params,
                     defaults={'lr': lr, 'decay': decay,
                               'momentum': momentum, 'eps': eps,
                               'weight_decay': weight_decay})

  def create_buffers(self, name, param):
    """Prepare exp_avg and exp_avg_sq buffers."""
    state = {'step': jnp.zeros(jax.local_device_count())}
    state['rms'] = jnp.ones_like(param)
    if self.get_hyper(name, 'momentum') is not None:
      state['momentum'] = jnp.zeros_like(param)
    return state

  def update_param(self, param, grad, state, opt_params):
    """The actual update step for this optimizer."""
    if param is None:
      return param, state
    state['step'] = state['step'] + 1
    # Apply weight decay
    if opt_params.get('weight_decay') is not None:
      grad = grad + param * opt_params['weight_decay']
    # EMA of the squared gradient
    state['rms'] = (state['rms'] * opt_params['decay']
                    + (1 - opt_params['decay']) * (grad ** 2))
    scaled_grad = (opt_params['lr'] * grad
                   / (state['rms'] + opt_params['eps']) ** 0.5)
    if state['momentum'] is not None:
      state['momentum'] = (state['momentum'] * opt_params['momentum']
                           + scaled_grad)
      step_size = state['momentum']
    else:
      step_size = scaled_grad

    return param - step_size, state


class Fromage(Optimizer):
  """Fromage optimizer, Bernstein et al. arXiv.org/abs/2002.03432.

  This version optionally includes weight decay.
  Attributes:
      params (iterable): nested list of params to optimize
      lr (float): learning rate.
      eps (float, optional): Minimum allowable norm. This term is required for
        in case parameters are zero-initialized (default: 1e-5).
      weight_decay (float, optional): weight decay (default: 0).
  """

  defaults = {'weight_decay': None, 'eps': 1e-5}

  def __init__(self, params, lr, weight_decay=None, eps=1e-5):
    super().__init__(
        params, defaults={'lr': lr, 'weight_decay': weight_decay, 'eps': eps})

  def create_buffers(self, name, param):  # pylint: disable=unused-argument
    """Prepares all momentum buffers for each parameter."""
    return {'step': jnp.zeros(1)}

  def update_param(self, param, grad, state, opt_params):
    """The actual update step for this optimizer."""
    if param is None:
      return param, state
    if opt_params['weight_decay'] is not None:
      grad = grad + param * opt_params['weight_decay']
    grad_norm = jnp.maximum(jnp.linalg.norm(grad), opt_params['eps'])
    param_norm = jnp.maximum(jnp.linalg.norm(param), opt_params['eps'])
    mult = jax.lax.rsqrt(1 + opt_params['lr'] ** 2)
    out = (param - opt_params['lr'] * grad * (param_norm / grad_norm)) * mult
    return out, state


def compute_norm(x, axis, keepdims):
  """Returns norm over arbitrary axis."""
  norm = jnp.sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5
  return norm


def unitwise_norm(x):
  """Computes norms of each output unit separately, assuming (HW)IO weights."""
  if len(jnp.squeeze(x).shape) <= 1:  # Scalars and vectors
    axis = None
    keepdims = False
  elif len(x.shape) in [2, 3]:  # Linear layers of shape IO
    axis = 0
    keepdims = True
  elif len(x.shape) == 4:  # Conv kernels of shape HWIO
    axis = [0, 1, 2,]
    keepdims = True
  else:
    raise ValueError(f'Got a parameter with shape not in [1, 2, 3, 4]! {x}')
  return compute_norm(x, axis, keepdims)


class SGD_AGC(Optimizer):  # pylint:disable=invalid-name
  """SGD with Unit-Adaptive Gradient-Clipping.

  References:
    [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
    Recognition Without Normalization.
  """
  defaults = {'weight_decay': None, 'momentum': None, 'dampening': 0,
              'nesterov': None, 'clipping': 0.01, 'eps': 1e-3}

  def __init__(self, params, lr, weight_decay=None,
               momentum=None, dampening=0, nesterov=None,
               clipping=0.01, eps=1e-3):
    super().__init__(
        params, defaults={'lr': lr, 'weight_decay': weight_decay,
                          'momentum': momentum, 'dampening': dampening,
                          'clipping': clipping, 'nesterov': nesterov,
                          'eps': eps})

  def create_buffers(self, name, param):
    return SGD.create_buffers(self, name, param)

  def update_param(self, param, grad, state, opt_params):
    """Clips grads if necessary, then applies the optimizer update."""
    if param is None:
      return param, state
    if opt_params['clipping'] is not None:
      param_norm = jnp.maximum(unitwise_norm(param), opt_params['eps'])
      grad_norm = unitwise_norm(grad)
      max_norm = param_norm * opt_params['clipping']
      # If grad norm > clipping * param_norm, rescale
      trigger = grad_norm > max_norm
      # Note the max(||G||, 1e-6) is technically unnecessary here, as
      # the clipping shouldn't trigger if the grad norm is zero,
      # but we include it in practice as a "just-in-case".
      clipped_grad = grad * (max_norm / jnp.maximum(grad_norm, 1e-6))
      grad = jnp.where(trigger, clipped_grad, grad)
    return SGD.update_param(self, param, grad, state, opt_params)


class Hybrid(Optimizer):
  """Optimizer which permits passing param groups with different base opts.


  The API for this class follows the case for any other optimizer where one
  specifies a list of dicts with separate hyperparams, but in this case it
  requires the user to also specify an 'opt' key for each group, such as e.g.
  [{'params': params0, 'opt': optim.Adam, 'lr': 0.1}].

  The user must also provide values for any arg in the selected optimizers which
  does not have a default value associated
  """

  def __init__(self, param_groups):
    if any(['opt' not in group for group in param_groups]):
      raise ValueError('All parameter groups must have an opt key!')
    self.defaults = ChainMap(*[group['opt'].defaults for group in param_groups])
    super().__init__(param_groups, defaults=dict(self.defaults))

  def create_buffers(self, name, param):
    return self.get_hyper(name, 'opt').create_buffers(self, name, param)

  def update_param(self, param, grad, state, opt_params):
    return opt_params['opt'].update_param(self, param, grad, state, opt_params)
