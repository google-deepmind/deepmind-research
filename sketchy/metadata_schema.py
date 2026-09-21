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

"""Sqlalchemy schema for the metadata db."""

import sqlalchemy

from sqlalchemy.ext import declarative


Column = sqlalchemy.Column
Integer = sqlalchemy.Integer
String = sqlalchemy.String
LargeBinary = sqlalchemy.LargeBinary
ForeignKey = sqlalchemy.ForeignKey

# pylint: disable=invalid-name

# https://docs.sqlalchemy.org/en/13/orm/tutorial.html

Base = declarative.declarative_base()


EpisodeTag = sqlalchemy.Table(
    'EpisodeTags', Base.metadata,
    Column(
        'EpisodeId', String, ForeignKey('Episodes.EpisodeId'),
        primary_key=True),
    Column('Tag', String, ForeignKey('Tags.Name'), primary_key=True))
"""Table relating episodes and tags.

Attributes:
  EpisodeId: A string of digits that uniquely identifies the episode.
  Tag: Human readable tag name.
"""


class Episode(Base):
  """Table describing individual episodes.

  Attributes:
    EpisodeId: A string of digits that uniquely identifies the episode.
    TaskId: A human readable name for the task corresponding to the behavior
        that generated the episode.
    DataPath: The name of the episode file holding the data for this episode.
    Timestamp: A unix timestamp recording when the episode was generated.
    EpisodeType: A string describing the type of policy that generated the
        episode.  Possible values are:
        - `EPISODE_ROBOT_AGENT`: The behavior policy is a learned or scripted
          controller.
        - `EPISODE_ROBOT_TELEOPERATION`: The behavior policy is a human
           teleoperating the robot.
        - `EPISODE_ROBOT_DAGGER`: The behavior policy is a mix of controller
          and human generated actions.
    Tags: A list of tags attached to this episode.
    Rewards: A list of `RewardSequence`s containing sketched rewards for this
      episode.
  """
  __tablename__ = 'Episodes'
  EpisodeId = Column(String, primary_key=True)
  TaskId = Column(String)
  DataPath = Column(String)
  Timestamp = Column(Integer)
  EpisodeType = Column(String)
  Tags = sqlalchemy.orm.relationship(
      'Tag', secondary=EpisodeTag, back_populates='Episodes')
  Rewards = sqlalchemy.orm.relationship(
      'RewardSequence', backref='Episode')


class Tag(Base):
  """Table of tags that can be attached to episodes.

  Attributes:
    Name: Human readable tag name.
    Episodes: The epsidoes that have been annotated with this tag.
  """
  __tablename__ = 'Tags'
  Name = Column(String, primary_key=True)
  Episodes = sqlalchemy.orm.relationship(
      'Episode', secondary=EpisodeTag, back_populates='Tags')


class RewardSequence(Base):
  """Table describing reward sequences for episodes.

  Attributes:
    EpisodeId: Foreign key into the `Episodes` table.
    RewardSequenceId: Distinguishes multiple rewards for the same episode.
    RewardTaskId: A human readable name of the task for this reward signal.
        Typically the same as the corresponding `TaskId` in the `Episodes`
        table.
    Type: A string describing the type of reward signal.  Currently the only
        value is `REWARD_SKETCH`.
    User: The name of the user who produced this reward sequence.
    Values: A sequence of float32 values, packed as a binary blob.  There is one
        float value for each frame of the episode, corresponding to the
        annotated reward.
  """
  __tablename__ = 'RewardSequences'
  EpisodeId = Column(
      'EpisodeId', String, ForeignKey('Episodes.EpisodeId'), primary_key=True)
  RewardSequenceId = Column(String, primary_key=True)
  RewardTaskId = Column('RewardTaskId', String)
  Type = Column(String)
  User = Column(String)
  Values = Column(LargeBinary)


class ArchiveFile(Base):
  """Table describing where episodes are stored in archives.

  This information is relevant if you want to download or extract a specific
  episode from the archives they are distributed in.

  Attributes:
    EpisodeId: Foreign key into the `Episodes` table.
    ArchiveFile: Name of the archive file containing the corresponding episode.
  """
  __tablename__ = 'ArchiveFiles'
  EpisodeId = Column(
      'EpisodeId', String, ForeignKey('Episodes.EpisodeId'), primary_key=True)
  ArchiveFile = Column(String)


# pylint: enable=invalid-name
