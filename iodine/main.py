# Copyright 2019 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=g-importing-member, g-multiple-import, g-import-not-at-top
# pylint: disable=protected-access, g-bad-import-order, missing-docstring
# pylint: disable=unused-variable, invalid-name, no-value-for-parameter

from copy import deepcopy
import os.path
import warnings
from absl import logging
import numpy as np
from sacred import Experiment, SETTINGS

# Ignore all tensorflow deprecation warnings
logging._warn_preinit_stderr = 0
warnings.filterwarnings("ignore", module=".*tensorflow.*")
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)
import sonnet as snt
from sacred.stflow import LogFileWriter
from iodine.modules import utils
from iodine import configurations

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment("iodine")


@ex.config
def default_config():
  continue_run = False  # set to continue experiment from an existing checkpoint
  checkpoint_dir = ("checkpoints/iodine"
                   )  # if continue_run is False, "_{run_id}" will be appended
  save_summaries_steps = 10
  save_checkpoint_steps = 1000

  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5

  learn_rate = 0.001
  batch_size = 4
  stop_after_steps = int(1e6)

  # Details for the dataset, model and optimizer are left empty here.
  # They can be found in the configurations for individual datasets,
  # which are provided in configurations.py and added as named configs.
  data = {}  # Dataset details will go here
  model = {}  # Model details will go here
  optimizer = {}  # Optimizer details will go here


ex.named_config(configurations.clevr6)
ex.named_config(configurations.multi_dsprites)
ex.named_config(configurations.tetrominoes)


@ex.capture
def build(identifier, _config):
  config_copy = deepcopy(_config[identifier])
  return utils.build(config_copy, identifier=identifier)


def get_train_step(model, dataset, optimizer):
  loss, scalars, _ = model(dataset("train"))
  global_step = tf.train.get_or_create_global_step()
  grads = optimizer.compute_gradients(loss)
  gradients, variables = zip(*grads)
  global_norm = tf.global_norm(gradients)
  gradients, global_norm = tf.clip_by_global_norm(
      gradients, 5.0, use_norm=global_norm)
  grads = zip(gradients, variables)
  train_op = optimizer.apply_gradients(grads, global_step=global_step)

  with tf.control_dependencies([train_op]):
    overview = model.get_overview_images(dataset("summary"))
    scalars["debug/global_grad_norm"] = global_norm

    summaries = {
        k: tf.summary.scalar(k, v) for k, v in scalars.items()
    }
    summaries.update(
        {k: tf.summary.image(k, v) for k, v in overview.items()})

    return tf.identity(global_step), scalars, train_op


@ex.capture
def get_checkpoint_dir(continue_run, checkpoint_dir, _run, _log):
  if continue_run:
    assert os.path.exists(checkpoint_dir)
    _log.info("Continuing run from checkpoint at {}".format(checkpoint_dir))
    return checkpoint_dir

  run_id = _run._id
  if run_id is None:  # then no observer was added that provided an _id
    if not _run.unobserved:
      _log.warning(
          "No run_id given or provided by an Observer. (Re-)using run_id=1.")
    run_id = 1
  checkpoint_dir = checkpoint_dir + "_{run_id}".format(run_id=run_id)
  _log.info(
      "Starting a new run using checkpoint dir: '{}'".format(checkpoint_dir))
  return checkpoint_dir


@ex.capture
def get_session(chkp_dir, loss, stop_after_steps, save_summaries_steps,
                save_checkpoint_steps):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  hooks = [
      tf.train.StopAtStepHook(last_step=stop_after_steps),
      tf.train.NanTensorHook(loss),
  ]

  return tf.train.MonitoredTrainingSession(
      hooks=hooks,
      config=config,
      checkpoint_dir=chkp_dir,
      save_summaries_steps=save_summaries_steps,
      save_checkpoint_steps=save_checkpoint_steps,
  )


@ex.command(unobserved=True)
def load_checkpoint(use_placeholder=False, session=None):
  dataset = build("data")
  model = build("model")
  if use_placeholder:
    inputs = dataset.get_placeholders()
  else:
    inputs = dataset()

  info = model.eval(inputs)
  if session is None:
    session = tf.Session()
  saver = tf.train.Saver()
  checkpoint_dir = get_checkpoint_dir()
  checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
  saver.restore(session, checkpoint_file)

  print('Successfully restored Checkpoint "{}"'.format(checkpoint_file))
  # print variables
  variables = tf.global_variables() + tf.local_variables()
  for row in snt.format_variables(variables, join_lines=False):
    print(row)

  return {
      "session": session,
      "model": model,
      "info": info,
      "inputs": inputs,
      "dataset": dataset,
  }


@ex.automain
@LogFileWriter(ex)
def main(save_summaries_steps):
  checkpoint_dir = get_checkpoint_dir()

  dataset = build("data")
  model = build("model")
  optimizer = build("optimizer")
  gstep, train_step_exports, train_op = get_train_step(model, dataset,
                                                       optimizer)

  loss, ari = [], []
  with get_session(checkpoint_dir, train_step_exports["loss/total"]) as sess:
    while not sess.should_stop():
      out = sess.run({
          "step": gstep,
          "loss": train_step_exports["loss/total"],
          "ari": train_step_exports["loss/ari_nobg"],
          "train": train_op,
      })
      loss.append(out["loss"])
      ari.append(out["ari"])
      step = out["step"]
      if step % save_summaries_steps == 0:
        mean_loss = np.mean(loss)
        mean_ari = np.mean(ari)
        ex.log_scalar("loss", mean_loss, step)
        ex.log_scalar("ari", mean_ari, step)
        print("{step:>6d} Loss: {loss: >12.2f}\t\tARI-nobg:{ari: >6.2f}".format(
            step=step, loss=mean_loss, ari=mean_ari))
        loss, ari = [], []
