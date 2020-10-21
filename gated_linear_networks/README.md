# Gated Linear Networks

Gated Linear Networks (GLNs) are a family of backpropation-free neural networks.
Each neuron in a GLN predicts the target density (or probability mass) based on
the outputs of the previous layer and is trained under a logarthmic loss.

## GLN variants

Neurons have probabilistic "activation functions". Implementations are provided
for the following distributions:

-   Gaussian, for regression.

-   Bernoulli, for binary classification and multi-class classification using a
    one-vs-all scheme.

## Examples

Usage examples are provided in [`examples`](examples).

## Implementation details

### Constraint satisfaction

Because each neuron implements a probability density/mass function we need to
ensure that they are well defined. For example, the scale parameter for a
Gaussian density needs to be positive. We implement these constraints using
linear projections and clipping.

### Aggregation

Because each neuron predicts the target, we can use any neuron output as the
"network output", and are not bound to the last layer. Typically last layer
neuron(s) are the best predictors, but they might take longer to converge in
theory. In this implementation, we use a single neuron at the last layer, which
then forms the network output.

There are alternative ways of aggregating, e.g. see Switching Aggregation in
Appendix D of *Gaussian Gated Linear Networks* (link:
https://arxiv.org/pdf/2006.05964.pdf).

## References

Coming soon.
