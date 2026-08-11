"""Minimal Sonnet 1.x compatibility shim using tf.compat.v1 APIs.

Replaces dm-sonnet<2 so the meshgraphnets code runs under TensorFlow 2.x
(with eager execution disabled) without any Sonnet dependency.
"""

import contextlib
import tensorflow.compat.v1 as tf

# ---------------------------------------------------------------------------
# Global name counter – replicates Sonnet 1.x auto-unique-naming.
# Each AbstractModule subclass registers its name here; duplicates get a
# numeric suffix (_1, _2, ...).  The dict is reset each Python process, so
# variable names are stable across train / eval invocations as long as
# modules are constructed in the same order.
# ---------------------------------------------------------------------------
_name_counts = {}


def _unique_scope_name(name):
    count = _name_counts.get(name, 0)
    _name_counts[name] = count + 1
    return name if count == 0 else '%s_%d' % (name, count)


# ---------------------------------------------------------------------------
# AbstractModule
# ---------------------------------------------------------------------------

class AbstractModule:
    """Drop-in replacement for snt.AbstractModule."""

    def __init__(self, name='module'):
        self._module_name = _unique_scope_name(name)

    @contextlib.contextmanager
    def _enter_variable_scope(self, reuse=None):
        """Context manager used in __init__ to scope variable creation."""
        with tf.variable_scope(self._module_name, reuse=reuse):
            yield

    def __call__(self, *args, **kwargs):
        """Calls _build inside the module's variable scope (AUTO_REUSE)."""
        with tf.variable_scope(self._module_name, reuse=tf.AUTO_REUSE):
            return self._build(*args, **kwargs)

    def _build(self, *args, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# reuse_variables decorator
# ---------------------------------------------------------------------------

def reuse_variables(method):
    """Replacement for @snt.reuse_variables.

    Wraps the decorated method so it runs inside the module's variable scope
    with AUTO_REUSE, matching variables created in _build / __call__.
    """
    def wrapper(self, *args, **kwargs):
        with tf.variable_scope(self._module_name, reuse=tf.AUTO_REUSE):
            return method(self, *args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm(AbstractModule):
    """Layer normalisation over the last axis."""

    def __init__(self, name='layer_norm'):
        super().__init__(name=name)

    def _build(self, inputs):
        size = int(inputs.shape[-1])
        gamma = tf.get_variable('gamma', [size], dtype=tf.float32,
                                initializer=tf.ones_initializer())
        beta = tf.get_variable('beta', [size], dtype=tf.float32,
                               initializer=tf.zeros_initializer())
        mean, variance = tf.nn.moments(inputs, axes=[-1], keepdims=True)
        normalised = (inputs - mean) / tf.sqrt(variance + 1e-5)
        return gamma * normalised + beta


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------

class Sequential:
    """Runs a list of callables sequentially."""

    def __init__(self, layers, name='sequential'):
        self._layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class _MLP(AbstractModule):
    """Multi-layer perceptron with ReLU activations."""

    def __init__(self, output_sizes, activate_final=False, name='mlp'):
        super().__init__(name=name)
        self._output_sizes = output_sizes
        self._activate_final = activate_final

    def _build(self, inputs):
        x = inputs
        for i, size in enumerate(self._output_sizes):
            with tf.variable_scope('linear_%d' % i):
                in_size = int(x.shape[-1])
                w = tf.get_variable('w', shape=[in_size, size],
                                    dtype=tf.float32,
                                    initializer=tf.glorot_uniform_initializer())
                b = tf.get_variable('b', shape=[size],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                x = tf.matmul(x, w) + b
            if i < len(self._output_sizes) - 1 or self._activate_final:
                x = tf.nn.relu(x)
        return x


class nets:
    MLP = _MLP
