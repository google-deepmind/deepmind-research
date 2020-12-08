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

"""UCF101 linear evaluation."""

import functools
from typing import Any, Dict, Optional

from absl import app
from absl import flags
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import sklearn
from sklearn import preprocessing
import sklearn.linear_model
import sklearn.svm
import tensorflow as tf
import tensorflow_datasets as tfds

from mmv import config
from mmv.models import mm_embeddings
from mmv.utils import checkpoint
from mmv.utils import ucf101_dataset


flags.DEFINE_string('checkpoint_path', '~/tmp/mmv_s3d.pkl',
                    'The directory to load pre-trained weights from.')
flags.DEFINE_string('dataset_folder', '/tmp/ucf101',
                    'The directory with the ucf101 dataset.')

flags.DEFINE_integer('eval_batch_size', 1,
                     'The batch size for evaluation.')
flags.DEFINE_integer('train_batch_size', 16,
                     'The batch size for training.')
flags.DEFINE_integer('num_train_epochs', 10,
                     'How many epochs to collect features during training.')
flags.DEFINE_integer('num_test_windows', 10,
                     'How many windows to average on during test.')
flags.DEFINE_integer('min_resize', 224,
                     'Min value to resize images to during preprocessing.')
flags.DEFINE_integer('crop_size', 224,
                     'Value to resize images to during preprocessing.')
flags.DEFINE_integer('num_frames', 32,
                     'Number of video frames.')
flags.DEFINE_integer('stride', 2,
                     'Stride for video frames.')
flags.DEFINE_integer('ucf101_split', 1,
                     'Which split of ucf101 to use.')


FLAGS = flags.FLAGS


def get_sampling_offset(sequence: tf.Tensor,
                        num_steps: Optional[int],
                        is_training: bool,
                        stride: int = 1,
                        seed: Optional[int] = None) -> tf.Tensor:
  """Calculates the initial offset for a sequence where all steps will fit.

  Args:
    sequence: any tensor where the first dimension is timesteps.
    num_steps: The number of timesteps we will output. If None,
      deterministically start at the first frame.
    is_training: A boolean indicates whether the graph is for training or not.
      If False, the starting frame always the first frame.
    stride: distance to sample between timesteps.
    seed: a deterministic seed to use when sampling.
  Returns:
    The first index to begin sampling from. A best effort is made to provide a
    starting index such that all requested steps fit within the sequence (i.e.
    offset + 1 + (num_steps - 1) * stride < len(sequence)). If this is not
    satisfied, the starting index is chosen randomly from the full sequence.
  """
  if num_steps is None or not is_training:
    return tf.constant(0)
  sequence_length = tf.shape(sequence)[0]
  max_offset = tf.cond(
      tf.greater(sequence_length, (num_steps - 1) * stride),
      lambda: sequence_length - (num_steps - 1) * stride,
      lambda: sequence_length)
  offset = tf.random.uniform(
      (),
      maxval=tf.cast(max_offset, tf.int32),
      dtype=tf.int32,
      seed=seed)
  return offset


def sample_or_pad_sequence_indices(sequence: tf.Tensor,
                                   num_steps: Optional[int],
                                   is_training: bool,
                                   repeat_sequence: bool = True,
                                   stride: int = 1,
                                   offset: Optional[int] = None) -> tf.Tensor:
  """Returns indices to take for sampling or padding a sequence to fixed size.

  Samples num_steps from the sequence. If the sequence is shorter than
  num_steps, the sequence loops. If the sequence is longer than num_steps and
  is_training is True, then we seek to a random offset before sampling. If
  offset is provided, we use that deterministic offset.

  This method is appropriate for sampling from a tensor where you want every
  timestep between a start and end time. See sample_stacked_sequence_indices for
  more flexibility.

  Args:
    sequence: any tensor where the first dimension is timesteps.
    num_steps: how many steps (e.g. frames) to take. If None, all steps from
      start to end are considered and `is_training` has no effect.
    is_training: A boolean indicates whether the graph is for training or not.
      If False, the starting frame is deterministic.
    repeat_sequence: A boolean indicates whether the sequence will repeat to
      have enough steps for sampling. If False, a runtime error is thrown if
      num_steps * stride is longer than sequence length.
    stride: distance to sample between timesteps.
    offset: a deterministic offset to use regardless of the is_training value.

  Returns:
    Indices to gather from the sequence Tensor to get a fixed size sequence.
  """
  sequence_length = tf.shape(sequence)[0]
  sel_idx = tf.range(sequence_length)

  if num_steps:
    if offset is None:
      offset = get_sampling_offset(sequence, num_steps, is_training, stride)

    if repeat_sequence:
      # Repeats sequence until num_steps are available in total.
      num_repeats = tf.cast(
          tf.math.ceil(
              tf.math.divide(
                  tf.cast(num_steps * stride + offset, tf.float32),
                  tf.cast(sequence_length, tf.float32)
                  )), tf.int32)
      sel_idx = tf.tile(sel_idx, [num_repeats])
    steps = tf.range(offset, offset + num_steps * stride, stride)
  else:
    steps = tf.range(0, sequence_length, stride)
  return tf.gather(sel_idx, steps)


def random_sample_sequence(sequence: tf.Tensor,
                           num_steps: int,
                           stride: int = 1) -> tf.Tensor:
  """Randomly sample a segment of size num_steps from a given sequence."""

  indices = sample_or_pad_sequence_indices(
      sequence=sequence,
      num_steps=num_steps,
      is_training=True,  # Random sample.
      repeat_sequence=True,  # Will repeat the sequence if request more.
      stride=stride,
      offset=None)
  indices.set_shape((num_steps,))
  output = tf.gather(sequence, indices)
  return output


def sample_linspace_sequence(sequence: tf.Tensor,
                             num_windows: int,
                             num_steps: int,
                             stride: int = 1) -> tf.Tensor:
  """Samples num_windows segments from sequence with linearly spaced offsets.

  The samples are concatenated in a single Tensor in order to have the same
  format structure per timestep (e.g. a single frame). If num_steps * stride is
  bigger than the number of timesteps, the sequence is repeated. This function
  can be used in evaluation in order to extract enough segments in order to span
  the entire sequence.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_windows: Number of windows retrieved from the sequence.
    num_steps: Number of steps (e.g. frames) to take.
    stride: Distance to sample between timesteps.

  Returns:
    A single Tensor with first dimension num_windows * num_steps. The Tensor
    contains the concatenated list of num_windows tensors which offsets have
    been linearly spaced from input.
  """
  sequence_length = tf.shape(sequence)[0]
  max_offset = tf.maximum(0, sequence_length - num_steps * stride)
  offsets = tf.linspace(0.0, tf.cast(max_offset, tf.float32), num_windows)
  offsets = tf.cast(offsets, tf.int32)

  all_indices = []
  for i in range(num_windows):
    all_indices.append(
        sample_or_pad_sequence_indices(
            sequence=sequence,
            num_steps=num_steps,
            is_training=False,
            repeat_sequence=True,  # Will repeat the sequence if request more.
            stride=stride,
            offset=offsets[i]))

  indices = tf.concat(all_indices, axis=0)
  indices.set_shape((num_windows * num_steps,))
  output = tf.gather(sequence, indices)

  return output


def resize_smallest(frames: tf.Tensor, min_resize: int) -> tf.Tensor:
  """Resizes frames so that min(height, width) is equal to min_resize.

  This function will not do anything if the min(height, width) is already equal
  to min_resize. This allows to save compute time.

  Args:
    frames: A Tensor of dimension [timesteps, input_h, input_w, channels].
    min_resize: Minimum size of the final image dimensions.
  Returns:
    A Tensor of shape [timesteps, output_h, output_w, channels] of type
      frames.dtype where min(output_h, output_w) = min_resize.
  """
  shape = tf.shape(frames)
  input_h = shape[1]
  input_w = shape[2]

  output_h = tf.maximum(min_resize, (input_h * min_resize) // input_w)
  output_w = tf.maximum(min_resize, (input_w * min_resize) // input_h)

  def resize_fn():
    frames_resized = tf.image.resize(frames, (output_h, output_w))
    return tf.cast(frames_resized, frames.dtype)

  should_resize = tf.math.logical_or(tf.not_equal(input_w, output_w),
                                     tf.not_equal(input_h, output_h))
  frames = tf.cond(should_resize, resize_fn, lambda: frames)

  return frames


def process_samples(features_dict, num_frames=32, stride=1, is_training=True,
                    min_resize=224, crop_size=224, num_windows=1):
  """Process video frames."""

  video = features_dict['video']

  if is_training:
    assert num_windows == 1
    video = random_sample_sequence(video, num_frames, stride)
    is_flipped = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
    video = tf.cond(tf.equal(is_flipped, 1),
                    true_fn=lambda: tf.image.flip_left_right(video),
                    false_fn=lambda: video)
  else:
    video = sample_linspace_sequence(video, num_windows, num_frames, stride)

  # Resize smallest side.
  video = resize_smallest(video, min_resize)

  if is_training:
    # Random crop.
    video = tf.image.random_crop(video, [num_frames, crop_size, crop_size, 3])
  else:
    # Central crop.
    video = tf.image.resize_with_crop_or_pad(video, crop_size, crop_size)

  video = tf.cast(video, tf.float32)
  video /= 255.0  # Set between [0, 1].

  features_dict['video'] = video
  return features_dict


def space_to_depth_batch(features_dict):
  images = features_dict['video']
  _, l, h, w, c = images.shape
  images = tf.reshape(images, [-1, l // 2, 2, h // 2, 2, w // 2, 2, c])
  images = tf.transpose(images, [0, 1, 3, 5, 2, 4, 6, 7])
  images = tf.reshape(images, [-1, l // 2, h // 2, w // 2, 8 * c])
  features_dict['video'] = images
  return features_dict


def reshape_windows(features_dict, num_frames):
  x = features_dict['video']
  x = tf.reshape(x, (-1, num_frames, x.shape[2], x.shape[3], x.shape[4]))
  features_dict['video'] = x
  return features_dict


def compute_accuracy_metrics(pred, gt, prefix=''):
  order_pred = np.argsort(pred, axis=1)
  assert len(gt.shape) == len(order_pred.shape) == 2
  top1_pred = order_pred[:, -1:]
  top5_pred = order_pred[:, -5:]
  top1_acc = np.mean(top1_pred == gt)
  top5_acc = np.mean(np.max(top5_pred == gt, 1))
  return {prefix + 'top1': top1_acc,
          prefix + 'top5': top5_acc}


def forward_fn(images: jnp.ndarray,
               audio_spectrogram: jnp.ndarray,
               word_ids: jnp.ndarray,
               is_training: bool,
               model_config: Dict[str, Any]):
  """Forward pass of the model."""

  # This should contain the pre-trained weights. We set it to zero because it
  # will be loaded from the checkpoint.
  language_model_vocab_size = 65536
  word_embedding_dim = 300
  dummy_embedding_matrix = jnp.zeros(shape=(language_model_vocab_size,
                                            word_embedding_dim))

  module = mm_embeddings.AudioTextVideoEmbedding(
      **model_config,
      word_embedding_matrix=dummy_embedding_matrix)
  return module(images=images,
                audio_spectrogram=audio_spectrogram,
                word_ids=word_ids,
                is_training=is_training)['vid_repr']


def main(argv):
  del argv

  sklearn_reg = 0.001
  model_config = config.get_model_config(FLAGS.checkpoint_path)

  forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
  forward_apply = jax.jit(functools.partial(forward.apply,
                                            is_training=False,
                                            model_config=model_config))

  # Get the UCF101 config.
  dset_config = tfds.video.ucf101.Ucf101.BUILDER_CONFIGS[FLAGS.ucf101_split]

  builder = ucf101_dataset.ModUcf101(
      data_dir=FLAGS.dataset_folder,
      config=dset_config)
  # Create the tfrecord files (no-op if already exists)
  dl_config = tfds.download.DownloadConfig(verify_ssl=False)
  builder.download_and_prepare(download_config=dl_config)

  # Generate the training dataset.
  train_ds = builder.as_dataset(split='train', shuffle_files=False)
  train_ds = train_ds.map(lambda x: process_samples(  # pylint: disable=g-long-lambda
      x, num_frames=FLAGS.num_frames, stride=FLAGS.stride, is_training=True,
      min_resize=FLAGS.min_resize, crop_size=FLAGS.crop_size))
  train_ds = train_ds.batch(batch_size=FLAGS.train_batch_size)
  if model_config['visual_backbone'] == 's3d':
    train_ds = train_ds.map(space_to_depth_batch)
  train_ds = train_ds.repeat(FLAGS.num_train_epochs)

  # Generate the test dataset.
  test_ds = builder.as_dataset(split='test', shuffle_files=False)
  test_ds = test_ds.map(lambda x: process_samples(  # pylint: disable=g-long-lambda
      x, num_frames=FLAGS.num_frames, stride=FLAGS.stride, is_training=False,
      min_resize=FLAGS.min_resize, crop_size=FLAGS.crop_size,
      num_windows=FLAGS.num_test_windows))
  test_ds = test_ds.batch(batch_size=FLAGS.eval_batch_size)
  test_ds = test_ds.map(lambda x: reshape_windows(  # pylint: disable=g-long-lambda
      x, num_frames=FLAGS.num_frames))

  if model_config['visual_backbone'] == 's3d':
    test_ds = test_ds.map(space_to_depth_batch)
  test_ds = test_ds.repeat(1)

  pretrained_weights = checkpoint.load_checkpoint(FLAGS.checkpoint_path)
  params = pretrained_weights['params']
  state = pretrained_weights['state']

  # Collect training samples.
  audio_frames = 96
  mel_filters = 40
  num_tokens = 16
  dummy_audio = jnp.zeros(
      shape=(FLAGS.train_batch_size, audio_frames, mel_filters, 1))
  dummy_word_ids = jnp.zeros(
      shape=(FLAGS.train_batch_size, num_tokens), dtype=jnp.int32)

  train_features = []
  train_labels = []
  print('Computing features on train')
  training_examples = iter(tfds.as_numpy(train_ds))
  for train_ex in training_examples:
    vid_representation, _ = forward_apply(params=params,
                                          state=state,
                                          images=train_ex['video'],
                                          audio_spectrogram=dummy_audio,
                                          word_ids=dummy_word_ids)
    train_labels.append(train_ex['label'])
    train_features.append(vid_representation)
    if len(train_labels) % 50 == 0:
      print(f'Processed {len(train_labels)} examples.')

  train_labels = np.concatenate(train_labels, axis=0)
  train_features = np.concatenate(train_features, axis=0)
  print(f'Finish collecting train features of shape {train_features.shape}')

  # Collect test samples.
  dummy_audio = jnp.zeros(
      shape=(FLAGS.eval_batch_size, audio_frames, mel_filters, 1))
  dummy_word_ids = jnp.zeros(
      shape=(FLAGS.eval_batch_size, num_tokens), dtype=jnp.int32)

  test_features = []
  test_labels = []
  print('Computing features on test')
  test_examples = iter(tfds.as_numpy(test_ds))
  for test_ex in test_examples:
    vid_representation_test, _ = forward_apply(params=params,
                                               state=state,
                                               images=test_ex['video'],
                                               audio_spectrogram=dummy_audio,
                                               word_ids=dummy_word_ids)
    test_labels.append(test_ex['label'])
    test_features.append(vid_representation_test)
    if len(test_labels) % 50 == 0:
      print(f'Processed {len(test_labels)} examples.')

  test_features = np.concatenate(test_features, axis=0)
  test_labels = np.concatenate(test_labels, axis=0)
  print(f'Finish collecting test features of shape {test_features.shape}')

  # Train classifier
  print('Training linear classifier!')
  classifier = sklearn.svm.LinearSVC(C=sklearn_reg)
  scaler = preprocessing.StandardScaler().fit(train_features)
  train_features = scaler.transform(train_features)
  classifier.fit(train_features, train_labels.ravel())
  print('Training done !')

  # Evaluation.
  test_features = scaler.transform(test_features)
  print('Running inference on train')
  pred_train = classifier.decision_function(train_features)
  print('Running inference on test')
  pred_test = classifier.decision_function(test_features)
  if FLAGS.num_test_windows > 1:
    pred_test = np.reshape(
        pred_test, (test_labels.shape[0], -1, pred_test.shape[1]))
    pred_test = pred_test.mean(axis=1)

  # Compute accuracies.
  metrics = compute_accuracy_metrics(pred_train, train_labels[:, None],
                                     prefix='train_')
  metrics.update(
      compute_accuracy_metrics(pred_test, test_labels[:, None], prefix='test_'))
  print(metrics)

if __name__ == '__main__':
  app.run(main)
