# A dataset of CAD sketches

<p align="center">
  <img width="75%" src="media/sketch_data.gif" />
</p>

## Overview

This repository contains the dataset used in ["Computer-Aided Design as Language"](https://arxiv.org/abs/2105.02769).
We provide the following splits:
  * Training (`4,656,607` sketches)
  * Validation (`50,000` sketches)
  * Test (`50,000` sketches)

## Quickstart

First, download the dataset files:
```shell
bash download_dataset.sh
```
This will place the splits under `data` subfolder.

In order to read the data, you will need [protocol buffer](https://developers.google.com/protocol-buffers)
compiler and [Tensorflow](https://www.tensorflow.org/):
```shell
apt install -y protobuf-compiler
virtualenv --python=python3.6 "${ENV}"
${ENV}/bin/activate
pip install tensorflow
```

Next, you need to compile `.proto` files that define the layout of entries in
the dataset:
```shell
protoc --python_out=. *.proto
```

Finally, you can use the generated classes to access the examples. The following
`python` snippet reads and prints the first 5 elements from the training split:
```python
import tensorflow as tf

import example_pb2

dataset = tf.data.TFRecordDataset("data/train.tfrecord")

for raw_record in dataset.take(5).as_numpy_iterator():
  example = example_pb2.Example()
  example.ParseFromString(raw_record)
  print(example, "\n")
```

Please refer to `example.proto` for details on the data layout.

## Citation

If you use this dataset in your research, please cite:
```
@article{ganin2021computer,
  title={Computer-aided design as language},
  author={Ganin, Yaroslav and Bartunov, Sergey and Li, Yujia and Keller, Ethan and Saliceti, Stefano},
  journal={arXiv preprint arXiv:2105.02769},
  year={2021}
}
```

## License

The code is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
The dataset is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Disclaimer

This is not an official Google product.
