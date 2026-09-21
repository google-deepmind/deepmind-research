# Copyright 2020 DeepMind Technologies Limited.
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

"""Example of loading rewards from the metadata file."""

from absl import app
from absl import flags
import numpy as np
import sqlalchemy

from sketchy import metadata_schema

flags.DEFINE_string(
    'metadata', '/tmp/metadata.sqlite', 'Path to metadata file.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  engine = sqlalchemy.create_engine('sqlite:///' + FLAGS.metadata)
  session = sqlalchemy.orm.sessionmaker(bind=engine)()

  episodes = session.query(metadata_schema.Episode).join(
      metadata_schema.RewardSequence).limit(5)

  for episode in episodes:
    rewards = np.frombuffer(episode.Rewards[0].Values, dtype=np.float32)
    print('---')
    print(f'Episode: {episode.EpisodeId}')
    print(f'Episode file: {episode.DataPath}')
    print(f'Reward type: {episode.Rewards[0].Type}')
    print(f'Reward values: {rewards}')


if __name__ == '__main__':
  app.run(main)
