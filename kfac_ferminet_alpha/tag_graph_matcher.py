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
"""A module for tagging and graph manipulation."""
import collections
import functools
import itertools
from typing import Any, NamedTuple, Sequence

from absl import logging
import jax
from jax import core as jax_core
from jax import lax
from jax import util as jax_util
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp
import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np
import ordered_set

from kfac_ferminet_alpha import layers_and_loss_tags as tags

USE_NETWORKX = False


def match_nodes(g1, g2, mapping, node1, node2):
  """Matching nodes when doing graph search."""

  if not kfac_node_match(g1.nodes[node1], g2.nodes[node2]):
    return False
  # Check predecessors
  p1 = set(n for n in g1.predecessors(node1) if n in mapping.keys())
  p2 = set(n for n in g2.predecessors(node2) if n in mapping.values())
  if len(p1) != len(p2):
    return False
  for p1_i in p1:
    if mapping[p1_i] not in p2:
      return False
  # Check successors
  s1 = set(n for n in g1.successors(node1) if n in mapping.keys())
  s2 = set(n for n in g2.successors(node2) if n in mapping.values())
  if len(s1) != len(s2):
    return False
  for s1_i in s1:
    if mapping[s1_i] not in s2:
      return False
  return True


def generate_candidates(g1, g2, mapping, node1, node2):
  """Generates the initial candidates for graph search."""
  # Check predecessors
  p1 = set(n for n in g1.predecessors(node1) if n not in mapping.keys())
  p2 = set(n for n in g2.predecessors(node2) if n not in mapping.values())
  candidates = ordered_set.OrderedSet(itertools.product(p1, p2))
  s1 = set(n for n in g1.successors(node1) if n not in mapping.keys())
  s2 = set(n for n in g2.successors(node2) if n not in mapping.values())
  candidates.update(list(itertools.product(s1, s2)))
  return candidates


def find_mappings(pattern, graph, mapping, terminals):
  """Finds all mappings from graph search of the pattern."""
  if len(mapping) == len(pattern):
    for k, v in terminals.items():
      v.add(mapping[k])
    return [frozenset(mapping.items())]
  mappings = set()
  nodes_list = list(mapping.keys())
  for node1 in reversed(nodes_list):
    for s1 in pattern.successors(node1):
      if s1 not in mapping.keys():
        for s2 in graph.successors(mapping[node1]):
          if s2 not in mapping.values():
            if s1 not in terminals or s2 not in terminals[s1]:
              if match_nodes(pattern, graph, mapping, s1, s2):
                mapping[s1] = s2
                mappings.update(
                    find_mappings(pattern, graph, mapping, terminals))
                mapping.pop(s1)
    for p1 in pattern.predecessors(node1):
      if p1 not in mapping.keys():
        for p2 in graph.predecessors(mapping[node1]):
          if p2 not in mapping.values():
            if p1 not in terminals or p2 not in terminals[p1]:
              if match_nodes(pattern, graph, mapping, p1, p2):
                mapping[p1] = p2
                mappings.update(
                    find_mappings(pattern, graph, mapping, terminals))
                mapping.pop(p1)
  return mappings


def match_pattern(pattern, graph):
  """Given a pattern returns all matches inside the graph."""
  if USE_NETWORKX:
    matcher = isomorphism.GraphMatcher(
        graph, pattern, node_match=kfac_node_match)
    mappings = list(
        dict((k, v)
             for v, k in mapping.items())
        for mapping in matcher.subgraph_isomorphisms_iter())
  else:
    mapping = collections.OrderedDict()
    params1 = [n for n in pattern.nodes if pattern.nodes[n]["op"] == "param"]
    params2 = [n for n in graph.nodes if graph.nodes[n]["op"] == "param"]
    terminals = {
        n: set() for n in pattern.nodes if not list(pattern.successors(n))
    }

    mappings = set()
    for node1, node2 in itertools.product(params1, params2):
      mapping[node1] = node2
      mappings.update(find_mappings(pattern, graph, mapping, terminals))
      mapping.pop(node1)
      for v in terminals.values():
        v.clear()
    mappings = list(dict(mapping) for mapping in mappings)

  var_mappings = []
  for mapping in mappings:
    var_mappings.append(dict())
    for k, v in mapping.items():
      cond = pattern.nodes[k]["op"] in ("param", "array")
      source = pattern.nodes[k]["var"] if cond else k
      target = graph.nodes[v]["var"] if cond else graph.nodes[v]["eqn"]
      var_mappings[-1][source] = target

  return var_mappings


def read_env(env, var):
  # Literals are values baked into the Jaxpr
  if isinstance(var, jax.core.Literal):
    return var.val
  return env[var]


def write_env(env, var, val):
  env[var] = val


def abstract_single_value(value):
  if isinstance(value, jnp.ndarray):
    value = jax_core.ShapedArray(np.shape(value), np.result_type(value))
    return pe.PartialVal.unknown(value)
  else:
    return value


def abstract_args(args):
  return jax.tree_map(abstract_single_value, args)


def _extract_call_jaxpr(primitive, params):
  if not (primitive.call_primitive or primitive.map_primitive):
    return None, params
  else:
    params = dict(params)
    return params.pop("call_jaxpr"), params


def evaluate_eqn(eqn, in_values, write_func):
  """Evaluate a single Jax equation and writes the outputs."""
  in_values = list(in_values)
  # This is logic specifically to handle `xla_call`
  call_jaxpr, params = _extract_call_jaxpr(eqn.primitive, eqn.params)
  if call_jaxpr:
    subfuns = [
        jax.core.lu.wrap_init(
            functools.partial(jax.core.eval_jaxpr, call_jaxpr, ()))
    ]
  else:
    subfuns = []
  ans = eqn.primitive.bind(*(subfuns + in_values), **params)
  if eqn.primitive.multiple_results:
    jax_util.safe_map(write_func, eqn.outvars, ans)
  else:
    write_func(eqn.outvars[0], ans)
  return ans


def clean_jaxpr_eqns(jaxpr, preserve_tags=True):
  """Performs dead code elimination on the jaxpr, preserving loss and layer tags."""
  eqns = []
  dependants = set(jaxpr.outvars)
  for eqn in reversed(jaxpr.eqns):
    check = False
    for v in eqn.outvars:
      if v in dependants:
        dependants.remove(v)
        check = True
    if isinstance(eqn.primitive, (tags.LossTag, tags.LayerTag)):
      check = check or preserve_tags
    if check:
      eqns.append(eqn)
      new_dependants = set(
          v for v in eqn.invars if not isinstance(v, jax_core.Literal))
      dependants = dependants.union(new_dependants)
  # Dependants should only be invars
  dependants = dependants - set(jaxpr.invars + jaxpr.constvars)

  if dependants:
    raise ValueError("Something went wrong with the dead code elimination.")
  return reversed(eqns)


def broadcast_merger(f):
  """Transforms `f` into a function where all consecutive broadcasts are merged."""

  def merged_func(*func_args):
    typed_jaxpr, out_avals = jax.make_jaxpr(f, return_shape=True)(*func_args)
    out_tree = jax.tree_structure(out_avals)
    jaxpr, consts = typed_jaxpr.jaxpr, typed_jaxpr.literals

    # Mapping from variable -> value
    env = dict()
    read = functools.partial(read_env, env)
    write = functools.partial(write_env, env)

    # Bind args and consts to environment
    flat_args = jax.tree_flatten(func_args)[0]
    jax_util.safe_map(write, jaxpr.invars, flat_args)
    jax_util.safe_map(write, jaxpr.constvars, consts)

    # Bind args and consts to environment
    jax_util.safe_map(write, jaxpr.invars, flat_args)
    jax_util.safe_map(write, jaxpr.constvars, consts)

    # Loop through equations and evaluate primitives using `bind`
    broadcasts_outputs = dict()
    for eqn in clean_jaxpr_eqns(jaxpr):
      # We ignore broadcasting of constants
      if (eqn.primitive.name == "broadcast_in_dim" and
          not all(isinstance(v, jax_core.Literal) for v in eqn.invars)):
        if eqn.invars[0] in broadcasts_outputs:
          x, dims = broadcasts_outputs[eqn.invars[0]]
          kept_dims = eqn.params["broadcast_dimensions"]
          kept_dims = [kept_dims[d] for d in dims]
          y = lax.broadcast_in_dim(x, eqn.params["shape"], kept_dims)
          jax_util.safe_map(write, eqn.outvars, [y])
          broadcasts_outputs[eqn.outvars[0]] = (x, kept_dims)
        else:
          inputs = jax_util.safe_map(read, eqn.invars)
          evaluate_eqn(eqn, inputs, write)
          broadcasts_outputs[eqn.outvars[0]] = (
              inputs[0], eqn.params["broadcast_dimensions"])
      else:
        evaluate_eqn(eqn, jax_util.safe_map(read, eqn.invars), write)
    return jax.tree_unflatten(out_tree, jax_util.safe_map(read, jaxpr.outvars))

  return merged_func


class JaxGraph(NamedTuple):
  jaxpr: Any
  consts: Any
  params: Any
  params_tree: Any
  in_tree: Any
  out_tree: Any
  digraph: nx.DiGraph
  tagging_func: Any


SPECIAL_OP_COMPARE_RULES = dict()


def default_compare(node1, node2):
  if node1["op"] != node2["op"]:
    return False
  params1, params2 = node1["eqn"].params, node2["eqn"].params
  if set(params1.keys()) != set(params2.keys()):
    return False
  for k in params1.keys():
    if params1[k] != params2[k]:
      return False
  return True


def reshape_compare(node1, node2):
  """Compares two reshape nodes."""
  assert node1["op"] == node2["op"] == "reshape"
  params1, params2 = node1["eqn"].params, node2["eqn"].params
  if params1["dimensions"] != params2["dimensions"]:
    return False
  return True


def broadcast_in_dim_compare(node1, node2):
  """Compares two reshape nodes."""
  assert node1["op"] == node2["op"] == "broadcast_in_dim"
  return True


def conv_compare(node1, node2):
  """Compares two conv_general_dialted nodes."""
  assert node1["op"] == node2["op"] == "conv_general_dilated"
  params1, params2 = node1["eqn"].params, node2["eqn"].params
  for k in ("window_strides", "padding", "lhs_dilation", "rhs_dilation",
            "lhs_shape", "rhs_shape"):
    if len(params1[k]) != len(params2[k]):
      return False
  if (len(params1["dimension_numbers"].lhs_spec) !=  #
      len(params2["dimension_numbers"].lhs_spec)):
    return False
  if (len(params1["dimension_numbers"].rhs_spec) !=  #
      len(params2["dimension_numbers"].rhs_spec)):
    return False
  if (len(params1["dimension_numbers"].out_spec) !=  #
      len(params2["dimension_numbers"].out_spec)):
    return False
  if ((params1["feature_group_count"] > 1) !=  #
      (params2["feature_group_count"] > 1)):
    return False
  if ((params1["batch_group_count"] > 1) !=  #
      (params2["batch_group_count"] > 1)):
    return False
  return True


SPECIAL_OP_COMPARE_RULES["reshape"] = reshape_compare
SPECIAL_OP_COMPARE_RULES["broadcast_in_dim"] = broadcast_in_dim_compare
SPECIAL_OP_COMPARE_RULES["conv_general_dilated"] = conv_compare


def kfac_node_match(node1, node2):
  """Checks if two nodes are equivalent."""
  # Parameters match with each other and nothing else
  if node1["op"] == "param" and node2["op"] == "param":
    return True
    # return node1["rank"] == node2["rank"]
  if node1["op"] == "param" or node2["op"] == "param":
    return False
  # Arrays always match each other and nothing else
  if node1["op"] == "array" and node2["op"] == "array":
    return True
  if node1["op"] == "array" or node2["op"] == "array":
    return False
  # Operators match first on name
  if node1["op"] != node2["op"]:
    return False
  compare = SPECIAL_OP_COMPARE_RULES.get(node1["op"], default_compare)
  return compare(node1, node2)


def var_to_str(var):
  """Returns a string representation of the variable of a Jax expression."""
  if isinstance(var, jax.core.Literal):
    return str(var)
  elif not isinstance(var, jax.core.Var):
    raise ValueError(f"Idk what to do with this {type(var)}?")
  c = int(var.count)
  if c == -1:
    return "_"
  str_rep = ""
  while c > 25:
    str_rep += chr(c % 26 + ord("a"))
    c = c // 26
  str_rep += chr(c + ord("a"))
  return str_rep[::-1]


def extract_param_vars_flat(jaxpr, in_tree, params_index):
  if params_index is None:
    params_index = []
  elif isinstance(params_index, int):
    params_index = [params_index]
  in_vars = jax.tree_unflatten(in_tree, jaxpr.invars)
  return jax.tree_flatten([in_vars[i] for i in params_index])


def fill_jaxpr_to_graph(graph, jaxpr, in_vars=None, out_vars=None):
  """Fills the graph with the jaxpr."""
  in_vars = in_vars or [var_to_str(v) for v in jaxpr.invars + jaxpr.constvars]
  in_map = dict(zip(jaxpr.invars + jaxpr.constvars, in_vars))
  out_vars = out_vars or [var_to_str(v) for v in jaxpr.outvars]
  out_map = dict(zip(jaxpr.outvars, out_vars))

  for eqn in jaxpr.eqns:
    in_vars = []
    for v in eqn.invars:
      if isinstance(v, jax.core.Literal):
        in_vars.append(var_to_str(v))
      else:
        in_vars.append(in_map.get(v, var_to_str(v)))
    out_vars = [out_map.get(v, var_to_str(v)) for v in eqn.outvars]
    in_str = ",".join(in_vars)
    out_str = ",".join(out_vars)
    if isinstance(eqn.primitive, tags.LossTag):
      func_name = "__loss_tag"
    elif isinstance(eqn.primitive, tags.LayerTag):
      func_name = "__layer_tag"
    else:
      func_name = eqn.primitive.name
    node_c = f"{func_name}({in_str})->{out_str}"
    graph.add_node(node_c, op=eqn.primitive.name, eqn=eqn)

    # Create incoming edges
    for v, name in zip(eqn.invars, in_vars):
      if not isinstance(v, jax.core.Literal):
        graph.add_edge(name, node_c)

    # Create output nodes and edges
    for v, name in zip(eqn.outvars, out_vars):
      graph.add_node(name, op="array", var=v)
      graph.add_edge(node_c, name)


def create_digraph(jaxpr, params):
  """Creates a directed graph from the given jaxpr and parameters."""
  graph = nx.DiGraph()
  # Create input nodes
  for v in jaxpr.invars + jaxpr.constvars:
    if v in params:
      graph.add_node(var_to_str(v), op="param", var=v)
    else:
      graph.add_node(var_to_str(v), op="array", var=v)
  fill_jaxpr_to_graph(graph, jaxpr)

  return graph


def function_to_jax_graph(func, args, params_index, tagging_func=None):
  """Creates a `JaxGraph` instance from the provided function."""
  in_tree = jax.tree_structure(args)
  typed_jaxpr = jax.make_jaxpr(func)(*args)
  jaxpr, consts = typed_jaxpr.jaxpr, typed_jaxpr.literals
  params, params_tree = extract_param_vars_flat(jaxpr, in_tree, params_index)

  digraph = create_digraph(jaxpr, params)
  if tagging_func is not None:
    tagging_func = functools.partial(tagging_func, jaxpr)
  return JaxGraph(
      jaxpr=jaxpr,
      consts=consts,
      params=params,
      params_tree=params_tree,
      in_tree=in_tree,
      out_tree=None,
      digraph=digraph,
      tagging_func=tagging_func)


def print_nice_jaxpr(jaxpr):
  for eqn in jaxpr.eqns:
    print(tuple(eqn.invars), "->", eqn.primitive.name, tuple(eqn.outvars))


def auto_register_tags(func,
                       func_args,
                       params_index: int = 0,
                       register_only_generic: bool = False,
                       compute_only_loss_tags: bool = True,
                       patterns_to_skip: Sequence[str] = ()):
  """Transform the function to one that is populated with tags."""
  func = broadcast_merger(func)
  graph = function_to_jax_graph(func, func_args, params_index=params_index)
  matches = dict()

  # Extract the tagged losses variables and all their ancestors
  loss_output_vars = []
  num_losses = 0
  loss_ancestors = set()
  for node in graph.digraph.nodes:
    if node.startswith("__loss_tag"):
      num_losses += 1
      ancestors = nx.ancestors(graph.digraph, node)
      ancestors.add(node)
      for output_node in node.split("->")[-1].split(","):
        ancestors.add(output_node)
        loss_output_vars.append(graph.digraph.nodes[output_node]["var"])
      loss_ancestors = loss_ancestors.union(ancestors)
  loss_output_vars = tuple(loss_output_vars)

  # Extract the sub-graph that leads to losses
  sub_graph = nx.induced_subgraph(graph.digraph, loss_ancestors)

  # First collect all parameters that are already part of a layer tag
  tagged_params = dict()
  pattern_counters = dict()
  for tag_node in (
      node for node in sub_graph.nodes if node.startswith("__layer_tag")):
    inputs = graph.digraph.nodes[tag_node]["eqn"].invars
    tag_instance = graph.digraph.nodes[tag_node]["eqn"].primitive
    if tag_instance.name == "generic_tag":
      tag_params = tag_instance.split_all_inputs(inputs)[0]
    else:
      tag_params = tag_instance.split_all_inputs(inputs)[2]
    pattern_number = pattern_counters.get(tag_instance.name, 0)
    for param in tag_params:
      if param not in graph.params:
        raise ValueError(f"You have registered a layer tag with parameter "
                         f"that is not part of the parameters at index "
                         f"{params_index}.")
      if param in tagged_params:
        raise ValueError(f"You have registered twice the parameter {param}.")
      tagged_params[param] = f"Manual[{tag_instance.name}_{pattern_number}]"
    if tag_instance.name not in pattern_counters:
      pattern_counters[tag_instance.name] = 1
    else:
      pattern_counters[tag_instance.name] += 1

  if not register_only_generic:
    for pattern_name, patterns in get_graph_patterns():
      if pattern_name in patterns_to_skip:
        logging.info("Skipping graph pattern %s", pattern_name)
        continue
      logging.info("Matching graph pattern %s", pattern_name)
      for pattern in patterns:
        for match_map in match_pattern(pattern.digraph, sub_graph):
          if len(pattern.jaxpr.outvars) > 1:
            raise NotImplementedError()
          output = pattern.jaxpr.outvars[0]
          if matches.get(match_map[output]) is not None:
            raise ValueError(f"Found more than one match for equation "
                             f"{match_map[output]}. Examine the jaxpr:\n "
                             f"{graph.jaxpr}")
          # Mark the parameters as already tagged
          match_params = set()
          match_params_already_tagged = False
          for param in match_map.values():
            if param in graph.params:
              match_params.add(param)
              if param in tagged_params.keys():
                match_params_already_tagged = True
          # Register the match only if no parameters are already registered
          if not match_params_already_tagged:
            matches[match_map[output]] = (match_map, pattern.tagging_func)
            pattern_number = pattern_counters.get(pattern_name, 0)
            for param in match_params:
              tagged_params[param] = f"Auto[{pattern_name}_{pattern_number}]"
            if pattern_name not in pattern_counters:
              pattern_counters[pattern_name] = 1
            else:
              pattern_counters[pattern_name] += 1

  # Mark remaining parameters as orphans
  orphan_params = sorted(
      set(graph.params) - set(tagged_params.keys()), key=lambda v: v.count)
  params_regs = [tagged_params.get(p, "Orphan") for p in graph.params]
  params_regs = jax.tree_unflatten(graph.params_tree, params_regs)
  logging.info("=" * 50)
  logging.info("Graph parameter registrations:")
  logging.info(params_regs)
  logging.info("=" * 50)

  # Construct a function with all of the extra tag registrations
  @functools.wraps(func)
  def wrapped_auto_registered(*args):
    flat_args, _ = jax.tree_flatten(args)
    # Mapping from variable -> value
    env = {}

    read = functools.partial(read_env, env)
    write = functools.partial(write_env, env)

    def tag(var):
      if matches.get(var) is not None:
        inv_map, tagging_func = matches[var]
        var_map = {k: v for k, v in inv_map.items() if not isinstance(k, str)}
        val_map = jax.tree_map(read, var_map)
        val = tagging_func(inv_map, val_map)
        env[var] = val

    # Bind args and consts to environment
    jax_util.safe_map(write, graph.jaxpr.invars, flat_args)
    jax_util.safe_map(write, graph.jaxpr.constvars, graph.consts)

    # Register any orphan parameters as generic
    for param_var in orphan_params:
      write(param_var, tags.register_generic(read(param_var)))

    # Set the correct output variables
    if compute_only_loss_tags:
      output_vars = loss_output_vars
      out_tree = jax.tree_structure(loss_output_vars)
    else:
      output_vars = graph.jaxpr.outvars
      out_tree = graph.out_tree

    # Loop through equations and evaluate primitives using `bind`
    losses_evaluated = 0
    for eqn in graph.jaxpr.eqns:
      evaluate_eqn(eqn, jax_util.safe_map(read, eqn.invars), write)
      jax_util.safe_map(tag, eqn.outvars)

      # If we want to output only tagged losses
      if isinstance(eqn.primitive, tags.LossTag):
        losses_evaluated += 1
      if compute_only_loss_tags and num_losses == losses_evaluated:
        break

    outputs = jax_util.safe_map(read, output_vars)
    return jax.tree_unflatten(out_tree, outputs)

  return wrapped_auto_registered


# Registered graphs
NAME_TO_JAX_GRAPH = dict()
DEFERRED_REGISTRATIONS = []


def register_function(name, func, tagging_func, example_args, params_index,
                      precedence):
  """Registers a function as a pattern in the graph matcher registry.

  The graph matcher needs to trace at least once the full function, which means
  you need to provide it with dummy arguments. The shapes of the arguments do
  not matter, as the graph matcher ignores their values, however the rank does.
  Especially if there is some broadcasting happening you should register with
  every possible broadcast pattern. As a general advice avoid using a shape to
  be 1, unless you want the pattern to specifically match that, as some
  operations, like squeeze for example, can have special behaviour then.

  Args:
    name: The name of the pattern that is being registered to.
    func: The function that performs the computation.
    tagging_func: Function that correctly creates the tag.
    example_args: Example arguments that can be inputted into `func`.
    params_index: Specifies at which index of the `example_args` are considered
      a parameter.
    precedence: This specifies what precedence the graph matcher is going to
      assign to the provided pattern. The graph matcher will go from lowest to
      highest precedence, randomly breaking ties, when matching. Note that the
      pattern that matches a parameter with the lowest precedence will get
      registered and no other will. Specifically useful when there is a pattern
      for a layer with and without bias, in which case the with bias
      registration always should go with lower precedence.
  """

  # This is required because we can not use Jax before InitGoogle() runs
  def register():
    jnp_args = jax.tree_map(jnp.asarray, example_args)
    graph = function_to_jax_graph(
        func, jnp_args, params_index=params_index, tagging_func=tagging_func)
    if NAME_TO_JAX_GRAPH.get(name) is None:
      NAME_TO_JAX_GRAPH[name] = (precedence, [])
    assert precedence == NAME_TO_JAX_GRAPH[name][0]
    NAME_TO_JAX_GRAPH[name][1].append(graph)

  DEFERRED_REGISTRATIONS.append(register)


def get_graph_patterns():
  """Returns all graph patterns sorted by their precedence."""
  while DEFERRED_REGISTRATIONS:
    DEFERRED_REGISTRATIONS.pop()()
  return [(name, pattern) for name, (_, pattern) in sorted(
      NAME_TO_JAX_GRAPH.items(), key=lambda pair: pair[1][0])]


# Dense with bias
register_function(
    "dense_with_bias",
    tags.dense_func,
    tags.dense_tagging,
    [np.zeros([11, 13]), [np.zeros([13, 7]), np.zeros([7])]],
    params_index=1,
    precedence=0)

# Dense without bias
register_function(
    "dense_no_bias",
    tags.dense_func,
    tags.dense_tagging, [np.zeros([11, 13]), [np.zeros([13, 7])]],
    params_index=1,
    precedence=1)

# Conv2d with bias
register_function(
    "conv2d_with_bias",
    tags.conv2d_func,
    tags.conv2d_tagging,
    [np.zeros([2, 8, 8, 5]), [np.zeros([3, 3, 5, 4]),
                              np.zeros([4])]],
    params_index=1,
    precedence=0)

# Conv2d without bias
register_function(
    "conv2d_no_bias",
    tags.conv2d_func,
    tags.conv2d_tagging, [np.zeros([2, 8, 8, 5]), [np.zeros([3, 3, 5, 4])]],
    params_index=1,
    precedence=1)

# Standard scale and shift with both scale and shift
register_function(
    "scale_and_shift",
    functools.partial(
        tags.scale_and_shift_func, has_scale=True, has_shift=True),
    functools.partial(
        tags.scale_and_shift_tagging, has_scale=True, has_shift=True),
    [np.zeros([2, 13]), [np.zeros([13]), np.zeros([13])]],
    params_index=1,
    precedence=0)

# Same but no broadcasting
register_function(
    "scale_and_shift",
    functools.partial(
        tags.scale_and_shift_func, has_scale=True, has_shift=True),
    functools.partial(
        tags.scale_and_shift_tagging, has_scale=True, has_shift=True),
    [np.zeros([13]), [np.zeros([13]), np.zeros([13])]],
    params_index=1,
    precedence=0)

# Scale and shift as implemented in batch norm layers in Haiku
register_function(
    "scale_and_shift",
    tags.batch_norm_func,
    functools.partial(
        tags.batch_norm_tagging_func, has_scale=True, has_shift=True),
    [[np.zeros([2, 13]), np.zeros([13])], [np.zeros([13]),
                                           np.zeros([13])]],
    params_index=1,
    precedence=0)

# Same but no broadcasting
register_function(
    "scale_and_shift",
    tags.batch_norm_func,
    functools.partial(
        tags.batch_norm_tagging_func, has_scale=True, has_shift=True),
    [[np.zeros([13]), np.zeros([13])], [np.zeros([13]),
                                        np.zeros([13])]],
    params_index=1,
    precedence=0)

# Only scale
register_function(
    "scale_only",
    functools.partial(
        tags.scale_and_shift_func, has_scale=True, has_shift=False),
    functools.partial(
        tags.scale_and_shift_tagging, has_scale=True, has_shift=False),
    [np.zeros([2, 13]), [np.zeros([13])]],
    params_index=1,
    precedence=1)
