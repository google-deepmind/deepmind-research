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

"""Configuration parameters for MMV."""


def get_model_config(ckpt_path):
  """Returns the model configuration to be used with each checkpoint."""

  config = {
      'audio_backbone': 'resnet50',
      'audio_model_kwargs': {
          'bn_config': {
              'create_offset': True,
              'create_scale': True,
              'decay_rate': 0.9,
              'eps': 1.0e-5
          }
      },
      'bn_config_proj': {
          'create_offset': True,
          'create_scale': True,
          'decay_rate': 0.9,
          'eps': 1.0e-5
      },
      'config_audio_text': {
          'embedding_dim': 512,
          'toaud_bn_after_proj': False,
          'toaud_head_mode': 'linear',
          'totxt_bn_after_proj': False,
          'totxt_head_mode': 'linear'
      },
      'config_video_audio': {
          'embedding_dim': 512,
          'toaud_bn_after_proj': True,
          'toaud_head_mode': 'mlp@512',
          'tovid_bn_after_proj': False,
          'tovid_head_mode': 'linear'
      },
      'config_video_text': {
          'embedding_dim': 256,
          'totxt_bn_after_proj': True,
          'totxt_head_mode': 'linear',
          'tovid_bn_after_proj': False,
          'tovid_head_mode': 'linear'
      },
      'mm_embedding_graph': 'fac_relu',
      'name': 'text_audio_video',
      'sentence_dim': 2048,
      'use_xreplica_bn': True,
      'vision_model_kwargs': {
          'bn_config': {
              'create_offset': True,
              'create_scale': True,
              'decay_rate': 0.9,
              'eps': 1.0e-5
          },
          'n_frames': 32,
          'width_mult': 1,
      },
  }

  if 's3d' in ckpt_path:
    config['visual_backbone'] = 's3d'

  if 'tsm_resnet_x1' in ckpt_path:
    config['visual_backbone'] = 'resnet50tsm'

  if 'tsm_resnet_x2' in ckpt_path:
    config['visual_backbone'] = 'resnet50tsm'
    config['vision_model_kwargs']['width_mult'] = 2

  return config
