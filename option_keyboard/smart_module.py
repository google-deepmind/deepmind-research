# pylint: disable=g-bad-file-header
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
"""Smart module export/import utilities."""

import inspect
import pickle

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.io import gfile
import tensorflow_hub as hub
import tree as nest
import wrapt


_ALLOWED_TYPES = (bool, float, int, str)


def _getcallargs(signature, *args, **kwargs):
  bound_args = signature.bind(*args, **kwargs)
  bound_args.apply_defaults()
  inputs = bound_args.arguments
  inputs.pop("self", None)
  return inputs


def _to_placeholder(arg):
  if arg is None or isinstance(arg, bool):
    return arg

  arg = tf.convert_to_tensor(arg)
  return tf.placeholder(dtype=arg.dtype, shape=arg.shape)


class SmartModuleExport(object):
  """Helper class for exporting TF-Hub modules."""

  def __init__(self, object_factory):
    self._object_factory = object_factory
    self._wrapped_object = self._object_factory()
    self._variable_scope = tf.get_variable_scope()
    self._captured_calls = {}
    self._captured_attrs = {}

  def _create_captured_method(self, method_name):
    """Creates a wrapped method that captures its inputs."""
    with tf.variable_scope(self._variable_scope):
      method_ = getattr(self._wrapped_object, method_name)

    @wrapt.decorator
    def wrapper(method, instance, args, kwargs):
      """Wrapped method to capture inputs."""
      del instance

      specs = inspect.signature(method)
      inputs = _getcallargs(specs, *args, **kwargs)

      with tf.variable_scope(self._variable_scope):
        output = method(*args, **kwargs)

      self._captured_calls[method_name] = [inputs, specs]

      return output

    return wrapper(method_)  # pylint: disable=no-value-for-parameter

  def __getattr__(self, name):
    """Helper method for accessing an attributes of the wrapped object."""
    # if "_wrapped_object" not in self.__dict__:
    #   return super(ExportableModule, self).__getattr__(name)

    with tf.variable_scope(self._variable_scope):
      attr = getattr(self._wrapped_object, name)

    if inspect.ismethod(attr) or inspect.isfunction(attr):
      return self._create_captured_method(name)
    else:
      if all([isinstance(v, _ALLOWED_TYPES) for v in nest.flatten(attr)]):
        self._captured_attrs[name] = attr
      return attr

  def __call__(self, *args, **kwargs):
    return self._create_captured_method("__call__")(*args, **kwargs)

  def export(self, path, session, overwrite=False):
    """Build the TF-Hub spec, module and sync ops."""

    method_specs = {}

    def module_fn():
      """A module_fn for use with hub.create_module_spec()."""
      # We will use a copy of the original object to build the graph.
      wrapped_object = self._object_factory()

      for method_name, method_info in self._captured_calls.items():
        captured_inputs, captured_specs = method_info
        tensor_inputs = nest.map_structure(_to_placeholder, captured_inputs)
        method_to_call = getattr(wrapped_object, method_name)
        tensor_outputs = method_to_call(**tensor_inputs)

        flat_tensor_inputs = nest.flatten(tensor_inputs)
        flat_tensor_inputs = {
            str(k): v for k, v in zip(
                range(len(flat_tensor_inputs)), flat_tensor_inputs)
        }
        flat_tensor_outputs = nest.flatten(tensor_outputs)
        flat_tensor_outputs = {
            str(k): v for k, v in zip(
                range(len(flat_tensor_outputs)), flat_tensor_outputs)
        }

        method_specs[method_name] = dict(
            specs=captured_specs,
            inputs=nest.map_structure(lambda _: None, tensor_inputs),
            outputs=nest.map_structure(lambda _: None, tensor_outputs))

        signature_name = ("default"
                          if method_name == "__call__" else method_name)
        hub.add_signature(signature_name, flat_tensor_inputs,
                          flat_tensor_outputs)

      hub.attach_message(
          "methods", tf.train.BytesList(value=[pickle.dumps(method_specs)]))
      hub.attach_message(
          "properties",
          tf.train.BytesList(value=[pickle.dumps(self._captured_attrs)]))

    # Create the spec that will be later used in export.
    hub_spec = hub.create_module_spec(module_fn, drop_collections=["sonnet"])

    # Get variables values
    module_weights = [
        session.run(v) for v in self._wrapped_object.get_all_variables()
    ]

    # create the sync ops
    with tf.Graph().as_default():
      hub_module = hub.Module(hub_spec, trainable=True, name="hub")

      assign_ops = []
      assign_phs = []
      for _, v in sorted(hub_module.variable_map.items()):
        ph = tf.placeholder(shape=v.shape, dtype=v.dtype)
        assign_phs.append(ph)
        assign_ops.append(tf.assign(v, ph))

      with tf.Session() as module_session:
        module_session.run(tf.local_variables_initializer())
        module_session.run(tf.global_variables_initializer())
        module_session.run(
            assign_ops, feed_dict=dict(zip(assign_phs, module_weights)))

        if overwrite and gfile.exists(path):
          gfile.rmtree(path)
        gfile.makedirs(path)
        hub_module.export(path, module_session)


class SmartModuleImport(object):
  """A class for importing graph building objects from TF-Hub modules."""

  def __init__(self, module):
    self._module = module
    self._method_specs = pickle.loads(
        self._module.get_attached_message("methods",
                                          tf.train.BytesList).value[0])
    self._properties = pickle.loads(
        self._module.get_attached_message("properties",
                                          tf.train.BytesList).value[0])

  def _create_wrapped_method(self, method):
    """Creates a wrapped method that converts nested inputs and outputs."""

    def wrapped_method(*args, **kwargs):
      """A wrapped method around a TF-Hub module signature."""

      inputs = _getcallargs(self._method_specs[method]["specs"], *args,
                            **kwargs)
      nest.assert_same_structure(self._method_specs[method]["inputs"], inputs)
      flat_inputs = nest.flatten(inputs)
      flat_inputs = {
          str(k): v for k, v in zip(range(len(flat_inputs)), flat_inputs)
      }

      signature = "default" if method == "__call__" else method
      flat_outputs = self._module(
          flat_inputs, signature=signature, as_dict=True)
      flat_outputs = [v for _, v in sorted(flat_outputs.items())]

      output_spec = self._method_specs[method]["outputs"]
      if output_spec is None:
        if len(flat_outputs) != 1:
          raise ValueError(
              "Expected output containing a single tensor, found {}".format(
                  flat_outputs))
        outputs = flat_outputs[0]
      else:
        outputs = nest.unflatten_as(output_spec, flat_outputs)

      return outputs

    return wrapped_method

  def __getattr__(self, name):
    if name in self._method_specs:
      return self._create_wrapped_method(name)

    if name in self._properties:
      return self._properties[name]

    return getattr(self._module, name)

  def __call__(self, *args, **kwargs):
    return self._create_wrapped_method("__call__")(*args, **kwargs)
