# REGAL

This dataset contains dataflow computational graphs generated procedurally,
intended for training and evaluating algorithms that optimize execution (e.g.
placement and scheduling), in
[TensorFlow's CostGraphDef](https://github.com/tensorflow/tensorflow/blob/59ee7f9138482d85cd93c004aca961bea35820c7/tensorflow/core/framework/cost_graph.proto#L12)
[protocol buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) format and
encoded as
[text](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.text_format).

Original paper
[REGAL: Transfer Learning For Fast Optimization of Computation Graphs](https://arxiv.org/abs/1905.02494)
(Paliwal, Gimeno, Nair, Li, Lubin, Kohli, Vinyals)

## Folder structure

There are 10000 training graphs, 1000 validation graphs and 1000 test graphs.
The file names follow the format of "graph_" plus a hash of the graph topology
plus ".pbtxt".

## Filtering

For each set (train, valid, test) there are not two graphs with the same
topology. We used the Biased Random Key Genetic Algorithm (BRKGA) to filter out
graphs that did not have "room for improvement"

"Room for improvement" was defined as the union of two conditions:

*   if BRKGA with a low fitness evaluation limit (number of calls to fitness
    function) did not fit the hardware constraints and BRKGA with a high number
    did.
*   if BRKGA with a high fitness evaluation limit was 20% faster in running time
    that BRKGA with a low number.

## Example Graph

```protobuf
node {
  name: "_SOURCE"
}
node {
  name: "node_0"
  id: 1
  control_input: 0
}
node {
  name: "node_1"
  id: 2
  output_info {
    size: 70
    alias_input_port: -1
  }
  control_input: 0
  compute_cost: 58
}
node {
  name: "node_2"
  id: 3
  output_info {
    size: 52
    alias_input_port: -1
  }
  control_input: 0
  compute_cost: 47
}
node {
  name: "node_3"
  id: 4
  input_info {
    preceding_node: 2
  }
  output_info {
    size: 55
    alias_input_port: -1
  }
  control_input: 0
  compute_cost: 58
}
```

## Dataset Location

The dataset is available in the following
[link](https://storage.googleapis.com/synthetic-graphs-dataset/synthetic-graphs.tar.gz)

## Dataset Metadata

The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">REGAL CostGraphDef Synthetic Dataset</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/deepmind/deepmind_research/regal</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://github.com/deepmind/deepmind_research/regal</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">
      This dataset contains dataflow computational graphs generated
      procedurally, intended for training and evaluating algorithms that
      optimize execution (e.g. placement and scheduling), in
      [TensorFlow's CostGraphDef](https://github.com/tensorflow/tensorflow/blob/59ee7f9138482d85cd93c004aca961bea35820c7/tensorflow/core/framework/cost_graph.proto#L12)
      [protocol buffer](https://en.wikipedia.org/wiki/Protocol_Buffers)
      format and encoded as
      [text](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.text_format).
      </code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">DeepMind</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/DeepMind</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>citation</td>
    <td><code itemprop="citation">https://identifiers.org/arxiv:1905.02494</code></td>
  </tr>
</table>
</div>

## Disclaimer

This is not an officially supported Google product.
