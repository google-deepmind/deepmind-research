# Copyright 2020 DeepMind Technologies Limited.
#
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
"""Loss functions to be used by LayerCollection."""
import abc
from typing import Tuple, Optional, Union, Sequence

import jax
import jax.numpy as jnp

from kfac_ferminet_alpha import distributions
from kfac_ferminet_alpha import layers_and_loss_tags as tags
from kfac_ferminet_alpha import utils

ArrayPair = Tuple[jnp.ndarray, jnp.ndarray]
FloatArray = Union[float, jnp.ndarray]
Index = Tuple[int]


class LossFunction(abc.ABC):
  """Abstract base class for loss functions.

  Note that unlike typical loss functions used in neural networks these are
  neither summed nor averaged over the batch and hence the output of evaluate()
  will not be a scalar. It is up to the user to then to correctly manipulate
  them as needed.
  """

  def __init__(self, weight: FloatArray):
    self._weight = weight

  @property
  def weight(self) -> FloatArray:
    return self._weight

  @property
  @abc.abstractmethod
  def targets(self) -> Optional[jnp.ndarray]:
    """The targets being predicted by the model.

    Returns:
      None or Tensor of appropriate shape for calling self._evaluate() on.
    """
    pass

  @property
  @abc.abstractmethod
  def inputs(self) -> Sequence[jnp.ndarray]:
    """The inputs to the loss function (excluding the targets)."""
    pass

  @abc.abstractmethod
  def copy_with_different_inputs(self, inputs: Sequence[jnp.ndarray]):
    pass

  def evaluate(
      self,
      targets: Optional[jnp.ndarray] = None,
      coefficient_mode: str = "regular",
  ) -> jnp.ndarray:
    """Evaluate the loss function on the targets."""
    if targets is None and self.targets is None:
      raise ValueError("Cannot evaluate losses with unspecified targets.")
    elif targets is None:
      targets = self.targets
    if coefficient_mode == "regular":
      multiplier = self.weight
    elif coefficient_mode == "sqrt":
      multiplier = jnp.sqrt(self.weight)
    elif coefficient_mode == "off":
      multiplier = 1.0
    else:
      raise ValueError(f"Unrecognized coefficient_mode={coefficient_mode}.")
    return self._evaluate(targets) * multiplier

  @abc.abstractmethod
  def _evaluate(self, targets: jnp.ndarray) -> jnp.ndarray:
    """Evaluates the negative log probability of the targets.

    Args:
      targets: Tensor that distribution can calculate log_prob() of.

    Returns:
      negative log probability of each target, summed across all targets.
    """
    pass

  def grad_of_evaluate(
      self,
      targets: Optional[jnp.ndarray],
      coefficient_mode: str,
  ) -> Sequence[jnp.ndarray]:
    """Evaluates the gradient of the loss function.

    Note that the targets of the loss must not be `None`.

    Args:
      targets: The potential targets on which to evaluate the gradient.
      coefficient_mode: The coefficient mode to use for evaluation.

    Returns:
      The gradient of the loss evaluation function with respect to the inputs.
    """
    def evaluate_sum(inputs: Sequence[jnp.ndarray]) -> jnp.ndarray:
      instance = self.copy_with_different_inputs(inputs)
      return jnp.sum(instance.evaluate(targets, coefficient_mode))
    return jax.grad(evaluate_sum)(self.inputs)

  def multiply_ggn(self, vector: jnp.ndarray) -> jnp.ndarray:
    """Right-multiply a vector by the GGN.

    Here the 'GGN' is the GGN matrix (whose definition is slightly flexible)
    of the loss function with respect to its inputs.

    Args:
      vector: The vector to multiply.  Must be the same shape(s) as the 'inputs'
        property.

    Returns:
      The vector right-multiplied by the GGN.  Will be of the same shape(s)
      as the 'inputs' property.
    """
    return utils.scalar_mul(self.multiply_ggn_unweighted(vector), self.weight)

  @abc.abstractmethod
  def multiply_ggn_unweighted(self, vector: jnp.ndarray) -> jnp.ndarray:
    """Same as `multiply_ggn`, but without taking into account the weight."""
    pass

  def multiply_ggn_factor(self, vector: jnp.ndarray) -> jnp.ndarray:
    """Right-multiply a vector by a factor B of the GGN.

    Here the 'GGN' is the GGN matrix (whose definition is slightly flexible)
    of the loss function with respect to its inputs.  Typically this will be
    block-diagonal across different cases in the batch, since the loss function
    is typically summed across cases.

    Note that B can be any matrix satisfying B * B^T = G where G is the GGN,
    but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply.  Must be of the shape given by the
        'ggn_factor_inner_shape' property.

    Returns:
      The vector right-multiplied by B.  Will be of the same shape(s) as the
      'inputs' property.
    """
    return utils.scalar_mul(
        self.multiply_ggn_factor_unweighted(vector), jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_ggn_factor_unweighted(self, vector: jnp.ndarray) -> jnp.ndarray:
    """Same as `multiply_ggn_factor`, but without taking into account the weight."""
    pass

  def multiply_ggn_factor_transpose(self, vector: jnp.ndarray) -> jnp.ndarray:
    """Right-multiply a vector by the transpose of a factor B of the GGN.

    Here the 'GGN' is the GGN matrix (whose definition is slightly flexible)
    of the loss function with respect to its inputs.  Typically this will be
    block-diagonal across different cases in the batch, since the loss function
    is typically summed across cases.

    Note that B can be any matrix satisfying B * B^T = G where G is the GGN,
    but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply.  Must be the same shape(s) as the 'inputs'
        property.

    Returns:
      The vector right-multiplied by B^T.  Will be of the shape given by the
      'ggn_factor_inner_shape' property.
    """
    return utils.scalar_mul(
        self.multiply_ggn_factor_transpose_unweighted(vector),
        jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_ggn_factor_transpose_unweighted(
      self,
      vector: jnp.ndarray
  ) -> jnp.ndarray:
    """Same as `multiply_ggn_factor_transpose`, but without taking into account the weight."""
    pass

  def multiply_ggn_factor_replicated_one_hot(self, index: Index) -> jnp.ndarray:
    """Right-multiply a replicated-one-hot vector by a factor B of the GGN.

    Here the 'GGN' is the GGN matrix (whose definition is slightly flexible)
    of the loss function with respect to its inputs.  Typically this will be
    block-diagonal across different cases in the batch, since the loss function
    is typically summed across cases.

    A 'replicated-one-hot' vector means a tensor which, for each slice along the
    batch dimension (assumed to be dimension 0), is 1.0 in the entry
    corresponding to the given index and 0 elsewhere.

    Note that B can be any matrix satisfying B * B^T = G where G is the GGN,
    but will agree with the one used in the other methods of this class.

    Args:
      index: A tuple representing in the index of the entry in each slice that
        is 1.0. Note that len(index) must be equal to the number of elements of
        the 'ggn_factor_inner_shape' tensor minus one.

    Returns:
      The vector right-multiplied by B^T. Will be of the same shape(s) as the
      'inputs' property.
    """
    return utils.scalar_mul(
        self.multiply_ggn_factor_replicated_one_hot_unweighted(index),
        jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_ggn_factor_replicated_one_hot_unweighted(
      self,
      index: Index
  ) -> jnp.ndarray:
    pass

  @property
  @abc.abstractmethod
  def ggn_factor_inner_shape(self) -> Sequence[int]:
    """The shape of the tensor returned by multiply_ggn_factor."""
    pass


class NegativeLogProbLoss(LossFunction):
  """Abstract base class for loss functions that are negative log probs."""

  @property
  def inputs(self):
    return self.params

  @property
  @abc.abstractmethod
  def params(self):
    """Parameters to the underlying distribution."""
    pass

  def multiply_fisher(self, vector: jnp.ndarray) -> jnp.ndarray:
    """Right-multiply a vector by the Fisher.

    Args:
      vector: The vector to multiply.  Must be the same shape(s) as the 'inputs'
        property.

    Returns:
      The vector right-multiplied by the Fisher.  Will be of the same shape(s)
      as the 'inputs' property.
    """
    return utils.scalar_mul(
        self.multiply_fisher_unweighted(vector), self.weight)

  @abc.abstractmethod
  def multiply_fisher_unweighted(self, vector: jnp.ndarray) -> jnp.ndarray:
    pass

  def multiply_fisher_factor(self, vector: jnp.ndarray) -> jnp.ndarray:
    """Right-multiply a vector by a factor B of the Fisher.

    Here the 'Fisher' is the Fisher information matrix (i.e. expected outer-
    product of gradients) with respect to the parameters of the underlying
    probability distribution (whose log-prob defines the loss). Typically this
    will be block-diagonal across different cases in the batch, since the
    distribution is usually (but not always) conditionally iid across different
    cases.

    Note that B can be any matrix satisfying B * B^T = F where F is the Fisher,
    but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply.  Must be of the shape given by the
        'fisher_factor_inner_shape' property.

    Returns:
      The vector right-multiplied by B. Will be of the same shape(s) as the
      'inputs' property.
    """
    return utils.scalar_mul(
        self.multiply_fisher_factor_unweighted(vector), jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_fisher_factor_unweighted(
      self,
      vector: jnp.ndarray
  ) -> jnp.ndarray:
    pass

  def multiply_fisher_factor_transpose(
      self,
      vector: jnp.ndarray
  ) -> jnp.ndarray:
    """Right-multiply a vector by the transpose of a factor B of the Fisher.

    Here the 'Fisher' is the Fisher information matrix (i.e. expected outer-
    product of gradients) with respect to the parameters of the underlying
    probability distribution (whose log-prob defines the loss). Typically this
    will be block-diagonal across different cases in the batch, since the
    distribution is usually (but not always) conditionally iid across different
    cases.

    Note that B can be any matrix satisfying B * B^T = F where F is the Fisher,
    but will agree with the one used in the other methods of this class.

    Args:
      vector: The vector to multiply.  Must be the same shape(s) as the 'inputs'
        property.

    Returns:
      The vector right-multiplied by B^T.  Will be of the shape given by the
      'fisher_factor_inner_shape' property.
    """
    return utils.scalar_mul(
        self.multiply_fisher_factor_transpose_unweighted(vector),
        jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_fisher_factor_transpose_unweighted(
      self,
      vector: jnp.ndarray
  ) -> jnp.ndarray:
    pass

  def multiply_fisher_factor_replicated_one_hot(
      self,
      index: Index
  ) -> jnp.ndarray:
    """Right-multiply a replicated-one-hot vector by a factor B of the Fisher.

    Here the 'Fisher' is the Fisher information matrix (i.e. expected outer-
    product of gradients) with respect to the parameters of the underlying
    probability distribution (whose log-prob defines the loss). Typically this
    will be block-diagonal across different cases in the batch, since the
    distribution is usually (but not always) conditionally iid across different
    cases.

    A 'replicated-one-hot' vector means a tensor which, for each slice along the
    batch dimension (assumed to be dimension 0), is 1.0 in the entry
    corresponding to the given index and 0 elsewhere.

    Note that B can be any matrix satisfying B * B^T = H where H is the Fisher,
    but will agree with the one used in the other methods of this class.

    Args:
      index: A tuple representing in the index of the entry in each slice that
        is 1.0. Note that len(index) must be equal to the number of elements of
        the 'fisher_factor_inner_shape' tensor minus one.

    Returns:
      The vector right-multiplied by B. Will be of the same shape(s) as the
      'inputs' property.
    """
    return utils.scalar_mul(
        self.multiply_fisher_factor_replicated_one_hot_unweighted(index),
        jnp.sqrt(self.weight))

  @abc.abstractmethod
  def multiply_fisher_factor_replicated_one_hot_unweighted(
      self,
      index: Index
  ) -> jnp.ndarray:
    pass

  @property
  @abc.abstractmethod
  def fisher_factor_inner_shape(self) -> Sequence[int]:
    """The shape of the tensor returned by multiply_fisher_factor."""
    pass

  @abc.abstractmethod
  def sample(self, rng_key: jnp.ndarray) -> jnp.ndarray:
    """Sample 'targets' from the underlying distribution."""
    pass

  def grad_of_evaluate_on_sample(
      self,
      rng_key: jnp.ndarray,
      coefficient_mode: str,
  ) -> Sequence[jnp.ndarray]:
    """Evaluates the gradient of the log probability on a random sample.

    Args:
      rng_key: Jax PRNG key for sampling.
      coefficient_mode: The coefficient mode to use for evaluation.

    Returns:
      The gradient of the log probability of targets sampled from the
      distribution.
    """
    return self.grad_of_evaluate(self.sample(rng_key), coefficient_mode)


class NaturalParamsNegativeLogProbLoss(NegativeLogProbLoss, abc.ABC):
  """Base class for neg log prob losses whose inputs are 'natural' parameters.

  We will take the GGN of the loss to be the Fisher associated with the
  distribution, which also happens to be equal to the Hessian for this class
  of loss functions.  See here: https://arxiv.org/abs/1412.1193

  'Natural parameters' are defined for exponential-family models. See for
  example: https://en.wikipedia.org/wiki/Exponential_family
  """

  def multiply_ggn_unweighted(self, vector: jnp.ndarray) -> jnp.ndarray:
    return self.multiply_fisher_unweighted(vector)

  def multiply_ggn_factor_unweighted(self, vector: jnp.ndarray) -> jnp.ndarray:
    return self.multiply_fisher_factor_unweighted(vector)

  def multiply_ggn_factor_transpose_unweighted(
      self,
      vector: jnp.ndarray
  ) -> jnp.ndarray:
    return self.multiply_fisher_factor_transpose_unweighted(vector)

  def multiply_ggn_factor_replicated_one_hot_unweighted(
      self,
      index: Index
  ) -> jnp.ndarray:
    return self.multiply_fisher_factor_replicated_one_hot_unweighted(index)

  @property
  def ggn_factor_inner_shape(self) -> Sequence[int]:
    return self.fisher_factor_inner_shape


class DistributionNegativeLogProbLoss(NegativeLogProbLoss):
  """Base class for neg log prob losses that use the distribution classes."""

  @property
  @abc.abstractmethod
  def dist(self):
    """The underlying distribution instance."""
    pass

  def _evaluate(self, targets: jnp.ndarray):
    return -self.dist.log_prob(targets)

  def sample(self, rng_key: jnp.ndarray):
    return self.dist.sample(seed=rng_key)

  @property
  def fisher_factor_inner_shape(self) -> Sequence[int]:
    return self.dist.mean().shape


class NormalMeanNegativeLogProbLoss(DistributionNegativeLogProbLoss,
                                    NaturalParamsNegativeLogProbLoss):
  """Neg log prob loss for a normal distribution parameterized by a mean vector.


  Note that the covariance is treated as the identity divided by 2.
  Also note that the Fisher for such a normal distribution with respect the mean
  parameter is given by:

     F = (1 / variance) * I

  See for example https://www.ii.pwr.edu.pl/~tomczak/PDF/[JMT]Fisher_inf.pdf.
  """

  def __init__(
      self,
      mean: jnp.ndarray,
      targets: Optional[jnp.ndarray] = None,
      variance: float = 0.5,
      weight: float = 1.0,
  ):
    super().__init__(weight=weight)
    self._mean = mean
    self._targets = targets
    self._variance = variance
    if not isinstance(variance, float):
      raise ValueError("The `variance` argument should be python float.")

  @property
  def targets(self) -> Optional[jnp.ndarray]:
    return self._targets

  @property
  def dist(self):
    scale_diag = jnp.full_like(self._mean, jnp.sqrt(self._variance))
    return distributions.MultivariateNormalDiag(self._mean, scale_diag)

  @property
  def params(self):
    return self._mean,

  def copy_with_different_inputs(self, inputs: Sequence[jnp.ndarray]):
    [mean] = inputs
    return NormalMeanNegativeLogProbLoss(
        mean=mean,
        targets=self.targets,
        variance=self._variance,
        weight=self.weight,
    )

  def multiply_fisher_unweighted(self, vector: jnp.ndarray) -> jnp.ndarray:
    return vector / self._variance

  def multiply_fisher_factor_unweighted(
      self,
      vector: jnp.ndarray,
  ) -> jnp.ndarray:
    return vector / jnp.sqrt(self._variance)

  def multiply_fisher_factor_transpose_unweighted(
      self,
      vector: jnp.ndarray,
  )  -> jnp.ndarray:
    return self.multiply_fisher_factor_unweighted(vector)  # it's symmetric

  def multiply_fisher_factor_replicated_one_hot_unweighted(
      self,
      index: Index,
  ) -> jnp.ndarray:
    assert len(index) == 1, f"Length of index was {len(index)}."
    index = index[0]
    ones_slice = jnp.ones([self._mean.shape[0]])[..., None]
    output_slice = ones_slice / jnp.sqrt(self._variance)
    return insert_slice_in_zeros(output_slice, 1, self._mean.shape[1], index)


def insert_slice_in_zeros(
    slice_to_insert: jnp.ndarray,
    dim: int,
    dim_size: int,
    position: int,
) -> jnp.ndarray:
  """Inserts slice into a larger tensor of zeros.

  Forms a new tensor which is the same shape as slice_to_insert, except that
  the dimension given by 'dim' is expanded to the size given by 'dim_size'.
  'position' determines the position (index) at which to insert the slice within
  that dimension.

  Assumes slice_to_insert.shape[dim] = 1.

  Args:
    slice_to_insert: The slice to insert.
    dim: The dimension which to expand with zeros.
    dim_size: The new size of the 'dim' dimension.
    position: The position of 'slice_to_insert' in the new tensor.

  Returns:
    The new tensor.

  Raises:
    ValueError: If the slice's shape at the given dim is not 1.
  """
  slice_shape = slice_to_insert.shape
  if slice_shape[dim] != 1:
    raise ValueError(f"Expected slice_to_insert.shape to have {dim} dim of 1,"
                     f" but was {slice_to_insert.shape[dim]}.")

  before = [0] * int(len(slice_shape))
  after = before[:]
  before[dim] = position
  after[dim] = dim_size - position - 1
  return jnp.pad(slice_to_insert, list(zip(before, after)))


#  _______            _____            _     _             _   _
# |__   __|          |  __ \          (_)   | |           | | (_)
#    | | __ _  __ _  | |__) |___  __ _ _ ___| |_ _ __ __ _| |_ _  ___  _ __
#    | |/ _` |/ _` | |  _  // _ \/ _` | / __| __| '__/ _` | __| |/ _ \| '_ \
#    | | (_| | (_| | | | \ \  __/ (_| | \__ \ |_| | | (_| | |_| | (_) | | | |
#    |_|\__,_|\__, | |_|  \_\___|\__, |_|___/\__|_|  \__,_|\__|_|\___/|_| |_|
#              __/ |              __/ |
#             |___/              |___/


NormalMeanNegativeLogProbLoss_tag = tags.LossTag(
    NormalMeanNegativeLogProbLoss, num_inputs=1)


def register_normal_predictive_distribution(
    mean: jnp.ndarray,
    targets: Optional[jnp.ndarray] = None,
    variance: float = 0.5,
    weight: float = 1.0,
):
  """Registers a normal predictive distribution.

  This corresponds to a squared error loss of the form
     weight/(2*var) * ||target - mean||^2

  Args:
    mean: A tensor defining the mean vector of the distribution. The first
      dimension must be the batch size.
    targets: (OPTIONAL) The targets for the loss function.  Only required if one
      wants to use the "empirical Fisher" instead of the true Fisher (which is
      controlled by the 'estimation_mode' to the optimizer).
      (Default: None)
    variance: float. The variance of the distribution. Note that the default
      value of 0.5 corresponds to a standard squared error loss weight *
      ||target - prediction||^2. If you want your squared error loss to be of
      the form 0.5*coeff*||target - prediction||^2 you should use
      variance=1.0.
      (Default: 0.5)
    weight: A scalar coefficient to multiply the log prob loss associated with
      this distribution. The Fisher will be multiplied by the corresponding
      factor. In general this is NOT equivalent to changing the temperature of
      the distribution, but in the ase of normal distributions it may be.
      (Default: 1.0)

  Returns:
    The mean and targets as dependable on the tag.
  """
  if targets is None:
    targets = jnp.zeros_like(mean)
  return NormalMeanNegativeLogProbLoss_tag.bind(
      mean, targets, variance=variance, weight=weight, return_loss=False)


def register_squared_error_loss(
    prediction: jnp.ndarray,
    targets: Optional[jnp.ndarray] = None,
    weight: float = 1.0,
):
  """Registers a squared error loss function.

  This assumes the squared error loss of the form ||target - prediction||^2,
  averaged across the mini-batch. If your loss uses a coefficient of 0.5
  you need to set the "weight" argument to reflect this.

  Args:
    prediction: The prediction made by the network (i.e. its output). The first
      dimension must be the batch size.
    targets: (OPTIONAL) The targets for the loss function.  Only required if one
      wants to use the "empirical Fisher" instead of the true Fisher (which is
      controlled by the 'estimation_mode' to the optimizer).
      (Default: None)
    weight: A float coefficient to multiply the loss function by.
      (Default: 1.0)
  Returns:
    The mean and targets as dependable on the tag.
  """
  return register_normal_predictive_distribution(
      prediction, targets=targets, variance=0.5, weight=weight)
