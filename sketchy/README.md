# Sketchy data

This is a dataset accompanying the paper
[Scaling data-driven robotics with reward sketching and batch reinforcement learning](https://arxiv.org/abs/1909.12200).
If you use this dataset in your research please cite

```
@article{cabi2019,
  title={Scaling data-driven robotics with reward sketching and batch reinforcement learning},
  author={Serkan Cabi and
    Sergio G{\'o}mez Colmenarejo and
    Alexander Novikov and
    Ksenia Konyushkova and
    Scott Reed and
    Rae Jeong and
    Konrad Zolna and
    Yusuf Aytar and
    David Budden and
    Mel Vecerik and
    Oleg Sushkov and
    David Barker and
    Jonathan Scholz and
    Misha Denil and
    Nando de Freitas and
    Ziyu Wang},
  journal={arXiv preprint arXiv:1909.12200},
  year={2019}
}
```

## See example data

There is a small amount of example data included in this repository.  To examine
it, run the following commands from the repository root (i.e. one level up from
this folder):

```
python3 -m venv .sketchy_env
source .sketchy_env/bin/activate
pip install --upgrade pip
pip install -r sketchy/requirements.txt
python -m sketchy.dataset_example --show_images
```

For an example of loading rewards for episodes see `reward_example.py`.

## Download the full dataset

Run `./download.sh path/to/download/folder` to download the full dataset.  The
full dataset requires ~5.0TB of disk space to download, and extracts to approximately the same size.

You can edit `download.sh` to download subsets of the data.

Once the dataset has been downloaded it can be extracted wtih
`./extract.sh path/to/download/folder`.

### Named subsets

We provide several named subsets of the full dataset, which can be easily
downloaded on their own.  See `download.sh` for a description of the subsets
that are provided.

The episodes in each of these named subsets are identified by a tag in the
metadata.
If you would like to curate your own subset you can download the metadata
file and inspect the `ArchiveFiles` table (see below) to figure out which
archive files contain the episodes you want.

# Dataset Contents

The dataset is distribted as a *metadata file* (`metadata.sqlite`) and a
collection of *archive files* (with names ending in `.tar.bz2`).

The metadata file contains information about the episodes, including annotated
rewards for a subset of the episodes.

Each archive file contains several *episode files*, which have names like
`10000313341320364033_b615a417-ce34-41a8-8411-2a1ce3f3bd07`.

Each episode file is a
[tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) file,
containing a sequence of *timesteps* for a single episode.

Each timestep is a `tf.train.Example` proto containing features corresponding to
the observations and actions from a particular point in time.

## Metadata

The metadata file, `metadata.sqlite`, is a sqlite database containing metadata
describing the contents of the files in the dataset.

The following sections describe the important metadata tables.  You can find the
full schema by running

```
sqlite3 metadata.sqlite <<< .schema
```

### Episodes

- `EpisodeId`: A string of digits that uniquely identifies the episode.
- `TaskId`: A human readable name for the task corresponding to the behavior
  that generated the episode.
- `DataPath`: The name of the episode file holding the data for this episode.
- `EpisodeType`: A string describing the type of policy that generated the
  episode.  Possible values are:
  - `EPISODE_ROBOT_AGENT`: The behavior policy is a learned or scripted
    controller.
  - `EPISODE_ROBOT_TELEOPERATION`: The behavior policy is a human teleoperating
    the robot.
  - `EPISODE_ROBOT_DAGGER`: The behavior policy is a mix of controller and human
    generated actions.
- `Timestamp`: A unix timestamp recording when the episode was generated.

### EpisodeTags

- `EpisodeId`: Foreign key into the `Episodes` table.
- `Tag`: A human readable identifier for some aspect of the episode (e.g. which
  object set is used).

### RewardSequences

- `EpisodeId`: Foreign key into the `Episodes` table.
- `RewardSequenceId`: Distinguishes multiple rewards for the same episode.
- `RewardTaskId`: A human readable name of the task for this reward signal.
  Typically the same as the corresponding `TaskId` in the `Episodes` table.
- `Type`: A string describing the type of reward signal.  Currently the only
  value is `REWARD_SKETCH`.
- `Values`: A sequence of float32 values, packed as a binary blob.  There is one
  float value for each frame of the episode, corresponding to the annotated
  reward.

### ArchiveFiles

- `EpisodeId`: Foreign key into the `Episodes` table.
- `ArchiveFile`: Name of the archive file containing the corresponding episode.

## Episodes

Each episode file is a
[tfrecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) file
containing a sequence of timesteps, encoded as
[`tf.train.Example`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto)
protos.

Each episode file contains a single episode, and each timestep within an episode
contains all of the observations and actions associated with a that timestep as
a single `tf.train.Example`. Within each episode file the timesteps are
temporally ordered, so reading a file from beginning to end will visit all of
the timesteps from the episode in the order they occurred.

Observations and actions occur at 10Hz.

## Timesteps

Each timestep is a collection of observations and actions. Actions stored with a
timestep correspond to actions taken in response to the observations they are
stored with.

For a description of the shapes and types of the timestep data, see the data
loader in `sketchy.py`.

# Dataset Metadata

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
    <td><code itemprop="name">Sketchy</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/deepmind/deepmind-research/tree/master/sketchy</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://github.com/deepmind/deepmind-research/tree/master/sketchy</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">
      Data accompanying
[Scaling data-driven robotics with reward sketching and batch reinforcement learning](https://arxiv.org/abs/1909.12200).
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
    <td><code itemprop="citation">https://identifiers.org/arxiv:1909.12200</code></td>
  </tr>
</table>
</div>
