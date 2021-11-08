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

"""Ucf101 with custom decoding params."""

import tensorflow as tf
import tensorflow_datasets as tfds

# Utilities functions.

tf.compat.v1.enable_eager_execution()

_CITATION = """\
@article{DBLP:journals/corr/abs-1212-0402,
  author    = {Khurram Soomro and
               Amir Roshan Zamir and
               Mubarak Shah},
  title     = {{UCF101:} {A} Dataset of 101 Human Actions Classes From Videos in
               The Wild},
  journal   = {CoRR},
  volume    = {abs/1212.0402},
  year      = {2012},
  url       = {http://arxiv.org/abs/1212.0402},
  archivePrefix = {arXiv},
  eprint    = {1212.0402},
  timestamp = {Mon, 13 Aug 2018 16:47:45 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1212-0402},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_LABELS_FNAME = 'video/ucf101_labels.txt'


class ModUcf101(tfds.video.Ucf101):
  """Ucf101 action recognition dataset with better quality.
  """

  def _info(self):

    ffmpeg_extra_args = ('-qscale:v', '2', '-r', '25', '-t', '00:00:20')

    video_shape = (
        None, self.builder_config.height, self.builder_config.width, 3)
    labels_names_file = tfds.core.tfds_path(_LABELS_FNAME)
    features = tfds.features.FeaturesDict({
        'video': tfds.features.Video(video_shape,
                                     ffmpeg_extra_args=ffmpeg_extra_args,
                                     encoding_format='jpeg'),  # pytype: disable=wrong-arg-types  # gen-stub-imports
        'label': tfds.features.ClassLabel(names_file=labels_names_file),
    })
    return tfds.core.DatasetInfo(
        builder=self,
        description='A 101-label video classification dataset.',
        features=features,
        homepage='https://www.crcv.ucf.edu/data-sets/ucf101/',
        citation=_CITATION,
    )
