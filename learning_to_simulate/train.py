# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=line-too-long
"""Training script for https://arxiv.org/pdf/2002.09405.pdf.

Example usage (from parent directory):
`python -m learning_to_simulate.train --data_path={DATA_PATH} --model_path={MODEL_PATH}`

Evaluate model from checkpoint (from parent directory):
`python -m learning_to_simulate.train --data_path={DATA_PATH} --model_path={MODEL_PATH} --mode=eval`

Produce rollouts (from parent directory):
`python -m learning_to_simulate.train --data_path={DATA_PATH} --model_path={MODEL_PATH} --output_path={OUTPUT_PATH} --mode=eval_rollout`


"""
# train.py
# -----------------------------------------------------------------------------
# Copyright (c) 2025 YourName.
# Adapted from DeepMind’s “Learning to Simulate” (arXiv:2002.09405)
# -----------------------------------------------------------------------------
import collections
import functools
import json
import os
import pickle
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
import tree

from learning_to_simulate import learned_simulator, noise_utils, reading_utils

# -----------------------------------------------------------------------------
# Constants & Config
# -----------------------------------------------------------------------------
Stats = collections.namedtuple('Stats', ['mean', 'std'])
INPUT_SEQUENCE_LENGTH = 6  # last 5 velocities + current
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def get_kinematic_mask(particle_types):
    """Boolean mask for kinematic particles."""
    return tf.equal(particle_types, KINEMATIC_PARTICLE_ID)


def prepare_one_step(features):
    """Prepare single-step input/target pairs."""
    pos = tf.transpose(features['position'], [1, 0, 2])
    y = pos[:, -1]
    features['position'] = pos[:, :-1]
    features['n_particles_per_example'] = tf.shape(pos)[:1]
    if 'step_context' in features:
        features['step_context'] = features['step_context'][-2][tf.newaxis]
    return features, y


def prepare_rollout_inputs(features):
    """Prepare full trajectory for rollout."""
    pos = tf.transpose(features['position'], [1, 0, 2])
    context = {
        'position': pos[:, :-1],
        'n_particles_per_example': tf.shape(pos)[:1],
        'particle_type': features['particle_type'],
    }
    if 'step_context' in features:
        context['step_context'] = features['step_context']
    y_dummy = pos[:, -1]
    return context, y_dummy


def batch_concat(ds, batch_size):
    """Concatenate along particle axis for batching variable-size inputs."""
    elem_spec = ds.element_spec
    def reduce_fn(init, elems):
        return tree.map_structure(lambda a, b: tf.concat([a, b], axis=0), init, elems)
    def window_to_batch(window):
        init = tree.map_structure(lambda ts: tf.zeros([0]+ts.shape.as_list()[1:], ts.dtype), elem_spec)
        return window.reduce(init, reduce_fn)
    return ds.window(batch_size).flat_map(window_to_batch)


def load_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'r') as fp:
        return json.load(fp)


def build_simulator(metadata, model_kwargs, noise_std):
    """Instantiate the learned simulator."""
    def cast(x): return np.array(x, np.float32)
    def combine(x, y): return np.sqrt(x**2 + y**2)

    stats = {
        'acceleration': Stats(cast(metadata['acc_mean']), combine(cast(metadata['acc_std']), noise_std)),
        'velocity': Stats(cast(metadata['vel_mean']), combine(cast(metadata['vel_std']), noise_std))
    }
    if 'context_mean' in metadata:
        stats['context'] = Stats(cast(metadata['context_mean']), cast(metadata['context_std']))

    return learned_simulator.LearnedSimulator(
        num_dimensions=metadata['dim'],
        connectivity_radius=metadata['default_connectivity_radius'],
        graph_network_kwargs=model_kwargs,
        boundaries=metadata['bounds'],
        num_particle_types=NUM_PARTICLE_TYPES,
        normalization_stats=stats,
        particle_type_embedding_size=model_kwargs.get('embedding_size', 16)
    )


# -----------------------------------------------------------------------------
# Model Logic
# -----------------------------------------------------------------------------
def one_step_loss(sim, features, y_true, noise_std):
    """Compute one-step prediction loss."""
    pos_seq = features['position']
    part_types = features['particle_type']
    mask = ~get_kinematic_mask(part_types)
    mask_f = tf.cast(mask, tf.float32)[..., None, None]

    noise = noise_utils.get_random_walk_noise_for_position_sequence(pos_seq, noise_std_last_step=noise_std)
    noise *= mask_f

    pred_acc, true_acc = sim.get_predicted_and_target_normalized_accelerations(
        next_position=y_true,
        position_sequence=pos_seq,
        position_sequence_noise=noise,
        n_particles_per_example=features['n_particles_per_example'],
        particle_types=part_types,
        global_context=features.get('step_context')
    )

    loss = tf.reduce_mean(tf.square(tf.boolean_mask(pred_acc - true_acc, mask[..., None])))
    pred_next = sim(pos_seq, features['n_particles_per_example'], part_types, features.get('step_context'))
    position_mse = tf.reduce_mean(tf.square(pred_next - y_true))
    return loss, position_mse, pred_next


def rollout_prediction(sim, features, metadata):
    """Run rollout for full trajectory."""
    seq_len = metadata['sequence_length']
    num_steps = seq_len - INPUT_SEQUENCE_LENGTH
    init = features['position'][:, :INPUT_SEQUENCE_LENGTH]
    part_types = features['particle_type']
    mask = get_kinematic_mask(part_types)[..., None, None]

    ground = features['position'][:, INPUT_SEQUENCE_LENGTH:]
    predictions = []

    current = init
    for step in tf.range(num_steps):
        pred = sim(current, features['n_particles_per_example'], part_types, features.get('step_context'))
        pred = tf.where(mask, ground[:, step:step+1], pred)
        predictions.append(pred)
        current = tf.concat([current[:, 1:], pred], axis=1)

    pred_stack = tf.concat(predictions, axis=1)
    mse = tf.reduce_mean(tf.square(pred_stack - ground))
    return mse, pred_stack


# -----------------------------------------------------------------------------
# Dataset Builders
# -----------------------------------------------------------------------------
def build_dataset(data_path, split, batch_size, mode):
    """Return tf.data.Dataset and meta dict."""
    metadata = load_metadata(data_path)
    files = [os.path.join(data_path, f'{split}.tfrecord')]
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(lambda x: reading_utils.parse_serialized_simulation_example(x, metadata=metadata),
                num_parallel_calls=tf.data.AUTOTUNE)

    if mode in ['train', 'eval']:
        ds = ds.flat_map(functools.partial(reading_utils.split_trajectory, window_length=INPUT_SEQUENCE_LENGTH+1))
        ds = ds.map(prepare_one_step, num_parallel_calls=tf.data.AUTOTUNE)
        if mode == 'train':
            ds = ds.shuffle(512).repeat()
        ds = batch_concat(ds, batch_size)
    elif mode == 'rollout':
        assert batch_size == 1
        ds = ds.map(prepare_rollout_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, metadata


# -----------------------------------------------------------------------------
# Training / Evaluation / Rollout
# -----------------------------------------------------------------------------
def train_loop(args):
    """Main training loop."""
    strategy = tf.distribute.MirroredStrategy() if args.multi_gpu else tf.distribute.get_strategy()
    model_kwargs = {
        'latent_size': args.latent_size,
        'mlp_hidden_size': args.hidden_size,
        'mlp_num_hidden_layers': args.hidden_layers,
        'num_message_passing_steps': args.message_passing_steps,
        'embedding_size': args.embedding_size
    }

    with strategy.scope():
        ds_train, metadata = build_dataset(args.data_path, 'train', args.batch_size, 'train')
        sim = build_simulator(metadata, model_kwargs, args.noise_std)

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, sim=sim)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, args.model_dir, max_to_keep=5)

        summary_writer = tf.summary.create_file_writer(os.path.join(args.model_dir, 'logs'))
        step_var = checkpoint.step

        @tf.function
        def train_step(features, y_true):
            with tf.GradientTape() as tape:
                loss, pos_mse, _ = one_step_loss(sim, features, y_true, args.noise_std)
            grads = tape.gradient(loss, sim.trainable_variables)
            optimizer.apply_gradients(zip(grads, sim.trainable_variables))
            return loss, pos_mse

        it = iter(ds_train)
        while int(step_var.numpy()) <= args.max_steps:
            features, y_true = next(it)
            loss, pos_mse = train_step(features, y_true)

            if step_var % args.log_every == 0:
                tf.print(f"[Train | Step {step_var.numpy():06d}] Loss: {loss:.6f}, Pos MSE: {pos_mse:.6f}")
                with summary_writer.as_default():
                    tf.summary.scalar('train_loss', loss, step=step_var)
                    tf.summary.scalar('train_pos_mse', pos_mse, step=step_var)

            if step_var % args.save_every == 0:
                ckpt_manager.save()
                tf.print(f"Checkpoint saved at step {step_var.numpy()}")

            step_var.assign_add(1)

def eval_loop(args):
    """One-step evaluation loop."""
    ds_eval, metadata = build_dataset(args.data_path, args.eval_split, args.batch_size, 'eval')
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        sim = build_simulator(metadata, vars(args), args.noise_std)
        checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(),
                                         sim=sim)
        checkpoint.restore(tf.train.latest_checkpoint(args.model_dir)).expect_partial()

        mse_vals = []
        for features, y_true in ds_eval.take(args.eval_steps):
            loss, pos_mse, _ = one_step_loss(sim, features, y_true, args.noise_std)
            mse_vals.append(pos_mse)
        avg_mse = tf.reduce_mean(mse_vals)
        tf.print(f"[Eval] One-step Position MSE over {len(mse_vals)} batches: {avg_mse:.6f}")

def rollout_loop(args):
    """Perform rollout and save predicted trajectories."""
    ds_rollout, metadata = build_dataset(args.data_path, args.eval_split, batch_size=1, mode='rollout')
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        sim = build_simulator(metadata, vars(args), args.noise_std)
        checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), sim=sim)
        checkpoint.restore(tf.train.latest_checkpoint(args.model_dir)).expect_partial()

        os.makedirs(args.output_dir, exist_ok=True)
        for i, (features, _) in enumerate(ds_rollout):
            mse, pred = rollout_prediction(sim, features, metadata)
            out = {
                'predicted_rollout': pred.numpy(),
                'ground_truth': features['position'][:, INPUT_SEQUENCE_LENGTH:].numpy(),
                'particle_type': features['particle_type'].numpy(),
                'metadata': metadata
            }
            path = os.path.join(args.output_dir, f"rollout_{args.eval_split}_{i:04d}.pkl")
            with open(path, "wb") as fp:
                pickle.dump(out, fp)
            tf.print(f"Saved rollout {i} with MSE {mse:.6f}")

# -----------------------------------------------------------------------------
# Command-line Interface
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Learning to Simulate Trainer")
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval", "rollout"])
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="rollouts")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=2_000_000)
    p.add_argument("--noise_std", type=float, default=6.7e-4)
    p.add_argument("--latent_size", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--hidden_layers", type=int, default=2)
    p.add_argument("--message_passing_steps", type=int, default=10)
    p.add_argument("--embedding_size", type=int, default=16)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--eval_split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--multi_gpu", action="store_true", help="Enable MirroredStrategy")
    return p.parse_args()

def main():
    args = parse_args()
    tf.print(f"Starting run in mode={args.mode} | timestamp={datetime.utcnow()}")
    if args.mode == "train":
        train_loop(args)
    elif args.mode == "eval":
        eval_loop(args)
    elif args.mode == "rollout":
        rollout_loop(args)
    else:
        raise ValueError(f"Unrecognized mode {args.mode}")

if __name__ == "__main__":
    main()

