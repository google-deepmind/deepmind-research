# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Construct DAGs representing causal graphs, and perform inference on them."""

import collections

import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
from tensorflow_probability.substrates import jax as tfp
import tree


class Node:
  """A node in a graphical model.

  Conceptually, this represents a random variable in a causal probabilistic
  model. It knows about its 'parents', i.e. other Nodes upon which this Node
  causally depends. The user is responsible for ensuring that any graph built
  with this class is acyclic.

  A node knows how to represent its probability density, conditional on the
  values of its parents.

  The node needs to have a 'name', corresponding to the series within the
  dataframe that should be used to populate it.
  """

  def __init__(self, distribution_module, parents=(), hidden=False):
    """Initialise a `Node` instance.

    Args:
      distribution_module: An instance of `DistributionModule`, a Haiku module
        that is suitable for modelling the conditional distribution of this
        node given any parents.
      parents: `Iterable`, optional. A (potentially nested) collection of nodes
        which are direct ancestors of `self`.
      hidden: `bool`, optional. Whether this node is hidden. Hidden nodes are
        permitted to not have corresponding observations.
    """
    parents = tree.flatten(parents)

    self._distribution_module = distribution_module
    self._column = distribution_module.column
    self._index = distribution_module.index
    self._hidden = hidden
    self._observed_value = None

    # When implementing the path-specific counterfactual fairness algorithm,
    # we need the concept of a distribution conditional on the 'corrected'
    # values of the parents. This is achieved via the 'node_to_replacement'
    # argument of make_distribution.

    # However, in order to work with the `fix_categorical` and `fix_continuous`
    # functions, we need to assign counterfactual values for parents at
    # evaluation time.
    self._parent_to_value = collections.OrderedDict(
        (parent, None) for parent in parents)

    # This is the conditional distribution using no replacements, i.e. it is
    # conditioned on the observed values of parents.
    self._distribution = None

  def __repr__(self):
    return 'Node<{}>'.format(self.name)

  @property
  def dim(self):
    """The dimensionality of this node."""
    return self._distribution_module.dim

  @property
  def name(self):
    return self._column

  @property
  def hidden(self):
    return self._hidden

  @property
  def observed_value(self):
    return self._observed_value

  def find_ancestor(self, name):
    """Returns an ancestor node with the given name."""
    if self.name == name:
      return self
    for parent in self.parents:
      found = parent.find_ancestor(name)
      if found is not None:
        return found

  @property
  def parents(self):
    return tuple(self._parent_to_value)

  @property
  def distribution_module(self):
    return self._distribution_module

  @property
  def distribution(self):
    self._distribution = self.make_distribution()
    return self._distribution

  def make_distribution(self, node_to_replacement=None):
    """Make a conditional distribution for this node | parents.

    By default we use values (representing 'real data') from the parent
    nodes as inputs to the distribution, however we can alternatively swap out
    any of these for arbitrary arrays by specifying `node_to_replacement`.

    Args:
      node_to_replacement: `None`, `dict: Node -> DeviceArray`.
        If specified, use the indicated array.

    Returns:
      `tfp.distributions.Distribution`
    """
    cur_parent_to_value = self._parent_to_value
    self._parent_to_value = collections.OrderedDict(
        (parent, parent.observed_value) for parent in cur_parent_to_value.keys()
    )

    if node_to_replacement is None:
      parent_values = self._parent_to_value.values()
      return self._distribution_module(*parent_values)
    args = []
    for node, value in self._parent_to_value.items():
      if node in node_to_replacement:
        replacement = node_to_replacement[node]
        args.append(replacement)
      else:
        args.append(value)
    return self._distribution_module(*args)

  def populate(self, data, node_to_replacement=None):
    """Given a dataframe, populate node data.

    If the Node does not have data present, this is taken to be a sign of
      a) An error if the node is not hidden.
      b) Fine if the node is hidden.
    In case a) an exception will be raised, and in case b) observed)v will
    not be mutated.

    Args:
      data: tf.data.Dataset
      node_to_replacement: None | dict(Node -> array). If not None, use the
        given ndarray data rather than extracting data from the frame. This is
        only considered when looking at the inputs to a distribution.

    Raises:
      RuntimeError: If `data` doesn't contain the necessary feature, and the
        node is not hidden.
    """
    column = self._column
    hidden = self._hidden
    replacement = None
    if node_to_replacement is not None and self in node_to_replacement:
      replacement = node_to_replacement[self]

    if replacement is not None:
      # If a replacement is present, this takes priority over any other
      # consideration.
      self._observed_value = replacement
      return

    if self._index < 0:
      if not hidden:
        raise RuntimeError(
            'Node {} is not hidden, and column {} is not in the frame.'.format(
                self, column))
      # Nothing to do - there is no data, and the node is hidden.
      return

    # Produce the observed value for this node.
    self._observed_value = self._distribution_module.prepare_data(data)


class DistributionModule(hk.Module):
  """Common base class for a Haiku module representing a distribution.

  This provides some additional functionality common to all modules that would
  be used as arguments to the `Node` class above.
  """

  def __init__(self, column, index, dim):
    """Initialise a `DistributionModule` instance.

    Args:
      column: `string`. The name of the random variable to which this
        distribution corresponds, and should match the name of the series in
        the pandas dataframe.
      index: `int`. The index of the corresponding feature in the dataset.
      dim: `int`. The output dimensionality of the distribution.
    """
    super().__init__(name=column.replace('-', '_'))
    self._column = column
    self._index = index
    self._dim = dim

  @property
  def dim(self):
    """The output dimensionality of this distribution."""
    return self._dim

  @property
  def column(self):
    return self._column

  @property
  def index(self):
    return self._index

  def prepare_data(self, data):
    """Given a general tensor, return an ndarray if required.

    This method implements the functionality delegated from
    `Node._prepare_data`, and it is expected that subclasses will override the
    implementation appropriately.

    Args:
      data: A tf.data.Dataset.

    Returns:
      `np.ndarray` of appropriately converted values for this series.
    """
    return data[:, [self._index]]

  def _package_args(self, args):
    """Concatenate args into a single tensor.

    Args:
      args: `List[DeviceArray]`, length > 0.
        Each array is of shape (batch_size, ?) or (batch_size,). The former
        will occur if looking at e.g. a one-hot encoded categorical variable,
        and the latter in the case of a continuous variable.

    Returns:
      `DeviceArray`, (batch_size, num_values).
    """
    return jnp.concatenate(args, axis=1)


class Gaussian(DistributionModule):
  """A Haiku module that maps some inputs into a normal distribution."""

  def __init__(self, column, index, dim=1, hidden_shape=(),
               hidden_activation=jnp.tanh, scale=None):
    """Initialise a `Gaussian` instance with some dimensionality."""
    super(Gaussian, self).__init__(column, index, dim)
    self._hidden_shape = tuple(hidden_shape)
    self._hidden_activation = hidden_activation
    self._scale = scale

    self._loc_net = hk.nets.MLP(self._hidden_shape + (self._dim,),
                                activation=self._hidden_activation)

  def __call__(self, *args):
    if args:
      # There are arguments - these represent the variables on which we are
      # conditioning. We set the mean of the output distribution to be a
      # function of these values, parameterised with an MLP.
      concatenated_inputs = self._package_args(args)
      loc = self._loc_net(concatenated_inputs)
    else:
      # There are no arguments, so instead have a learnable location parameter.
      loc = hk.get_parameter('loc', shape=[self._dim], init=jnp.zeros)

    if self._scale is None:
      # The scale has not been explicitly specified, in which case it is left
      # to be single value, i.e. not a function of the conditioning set.
      log_var = hk.get_parameter('log_var', shape=[self._dim], init=jnp.ones)
      scale = jnp.sqrt(jnp.exp(log_var))
    else:
      scale = jnp.float32(self._scale)

    return tfp.distributions.Normal(loc=loc, scale=scale)

  def prepare_data(self, data):
    # For continuous data, we ensure the data is of dtype float32, and
    # additionally that the resulant shape is (num_examples, 1)
    # Note that this implementation only works for dim=1, however this is
    # currently also enforced by the fact that pandas series cannot be
    # multidimensional.
    result = data[:, [self.index]].astype(jnp.float32)
    if len(result.shape) == 1:
      result = jnp.expand_dims(result, axis=1)
    return result


class GaussianMixture(DistributionModule):
  """A Haiku module that maps some inputs into a mixture of normals."""

  def __init__(self, column, num_components, dim=1):
    """Initialise a `GaussianMixture` instance with some dimensionality.

    Args:
      column: `string`. The name of the column.
      num_components: `int`. The number of gaussians in this mixture.
      dim: `int`. The dimensionality of the variable.
    """
    super().__init__(column, -1, dim)
    self._num_components = num_components
    self._loc_net = hk.nets.MLP([self._dim])
    self._categorical_logits_module = hk.nets.MLP([self._num_components])

  def __call__(self, *args):
    # Define component Gaussians to be independent functions of args.
    locs = []
    scales = []
    for _ in range(self._num_components):
      loc = hk.get_parameter('loc', shape=[self._dim], init=jnp.zeros)
      log_var = hk.get_parameter('log_var', shape=[self._dim], init=jnp.ones)
      scale = jnp.sqrt(jnp.exp(log_var))
      locs.extend(loc)
      scales.extend(scale)

    # Define the Categorical distribution which switches between these
    categorical_logits = hk.get_parameter('categorical_logits',
                                          shape=[self._num_components],
                                          init=jnp.zeros)

    # Enforce positivity in the logits
    categorical_logits = jax.nn.sigmoid(categorical_logits)

    # If we have a multidimensional node, then the normal distributions above
    # have a batch shape of (dim,). We want to select between these using the
    # categorical distribution, so tile the logits to match this shape
    expanded_logits = jnp.repeat(categorical_logits, self._dim)

    categorical = tfp.distributions.Categorical(logits=expanded_logits)

    return tfp.distributions.MixtureSameFamily(
        mixture_distribution=categorical,
        components_distribution=tfp.distributions.Normal(
            loc=locs, scale=scales))


class MLPMultinomial(DistributionModule):
  """A Haiku module that consists of an MLP + multinomial distribution."""

  def __init__(self, column, index, dim, hidden_shape=(),
               hidden_activation=jnp.tanh):
    """Initialise an MLPMultinomial instance.

    Args:
      column: `string`. Name of the corresponding dataframe column.
      index: `int`. The index of the input data for this feature.
      dim: `int`. Number of categories.
      hidden_shape: `Iterable`, optional. Shape of hidden layers.
      hidden_activation: `Callable`, optional. Non-linearity for hidden
        layers.
    """
    super(MLPMultinomial, self).__init__(column, index, dim)
    self._hidden_shape = tuple(hidden_shape)
    self._hidden_activation = hidden_activation
    self._logit_net = hk.nets.MLP(self._hidden_shape + (self.dim,),
                                  activation=self._hidden_activation)

  @classmethod
  def from_frame(cls, data, column, hidden_shape=()):
    """Create an MLPMultinomial instance from a pandas dataframe and column."""
    # Helper method that uses the dataframe to work out how many categories
    # are in the given column. The dataframe is not used for any other purpose.
    if not isinstance(data[column].dtype, pd.api.types.CategoricalDtype):
      raise ValueError('{} is not categorical.'.format(column))
    index = list(data.columns).index(column)
    num_categories = len(data[column].cat.categories)
    return cls(column, index, num_categories, hidden_shape)

  def __call__(self, *args):
    if args:
      concatenated_inputs = self._package_args(args)
      logits = self._logit_net(concatenated_inputs)
    else:
      logits = hk.get_parameter('b', shape=[self.dim], init=jnp.zeros)
    return tfp.distributions.Multinomial(logits=logits, total_count=1.0)

  def prepare_data(self, data):
    # For categorical data, we convert to a one-hot representation using the
    # pandas category 'codes'. These are integers, and will have a definite
    # ordering that is identical between runs.
    codes = data[:, self.index]
    codes = codes.astype(jnp.int32)
    return jnp.eye(self.dim)[codes]


def populate(nodes, dataframe, node_to_replacement=None):
  """Populate observed values for nodes."""
  for node in nodes:
    node.populate(dataframe, node_to_replacement=node_to_replacement)
