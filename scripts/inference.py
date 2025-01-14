import os
import jax
import jax.numpy as jnp
import numpy as np
import sys
from ml_collections import config_flags, ConfigDict
import tensorflow as tf

sys.path.append(".")

import numpy as np
from absl import app, flags, logging as absl_logging
from palivla.model_components import ModelComponents
from palivla.optimizer import make_optimizer
from palivla.spec import ModuleSpec, OptimizerSpec
from palivla.utils import host_broadcast_str
from octo.data import traj_transforms
from jax.sharding import NamedSharding, PartitionSpec as P
import orbax.checkpoint as ocp
from palivla.components.train_state import ShardingMetadata
from scalax.sharding import (
    MeshShardingHelper,
    FSDPShardingRule,
    PartitionSpec,
)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")

def make_sharding(config: ConfigDict):
    mesh = MeshShardingHelper([-1], ["fsdp"])
    sharding_metadata = ShardingMetadata(
        mesh=mesh,
        model_sharding_rule=FSDPShardingRule(
            "fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"]
        ),
    )
    return sharding_metadata

def restructure(traj):
    # apply a standardization function, if provided
    if standardize_fn is not None:
        traj = ModuleSpec.instantiate(standardize_fn)(traj)

    if not all(k in traj for k in REQUIRED_KEYS):
        raise ValueError(
            f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. "
            "Did you write a `standardize_fn`?"
        )

    # extracts images, depth images and proprio from the "observation" dict
    traj_len = tf.shape(traj["action"])[0]
    old_obs = traj["observation"]
    new_obs = {}
    for new, old in image_obs_keys.items():
        if old is None:
            new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
        else:
            new_obs[f"image_{new}"] = old_obs[old]

    for new, old in depth_obs_keys.items():
        if old is None:
            new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
        else:
            new_obs[f"depth_{new}"] = old_obs[old]

    if proprio_obs_key is not None:
        new_obs["proprio"] = tf.cast(old_obs[proprio_obs_key], tf.float32)

    # add timestep info
    new_obs["timestep"] = tf.range(traj_len)

    # extracts `language_key` into the "task" dict, or samples uniformly if `language_key` fnmatches multiple keys
    task = {}
    if language_key is not None:
        task["language_instruction"] = sample_match_keys_uniform(traj, language_key)
        if task["language_instruction"].dtype != tf.string:
            raise ValueError(
                f"Language key {language_key} has dtype {task['language_instruction'].dtype}, "
                "but it must be tf.string."
            )
    # add reward and mask
    num_final_repeat = 1
    num_pos = tf.minimum(num_final_repeat, traj_len)
    reward = tf.concat(
        # [-tf.ones(traj_len - num_pos, dtype=tf.float32), tf.zeros(num_pos, dtype=tf.float32)], axis=0
            [tf.zeros(traj_len - num_pos, dtype=tf.float32), tf.ones(num_pos, dtype=tf.float32)], axis=0
    )
    mask = tf.concat(
        [tf.ones(traj_len - num_pos, dtype=tf.float32), tf.zeros(num_pos, dtype=tf.float32)], axis=0
    )
    mc_return = tf.scan(
        lambda prev_return, x: x[0] + mc_discount * prev_return * x[1],
        [reward, mask],
        initializer=0.0,
        reverse=True
    )
    
    # # repeat last action
    # next_action = tf.concat([traj["action"][1:, ...], traj["action"][-1:, ...]], axis=0)
    # import pdb; pdb.set_trace()
    # next_obs = {k: tf.concat([v[1:, ...], v[-1:, ...]], axis=0) for k, v in new_obs.items()}

    # This only works for bridge, since the dataset includes this metadata.
    frame_key = tf.strings.join([
        tf.repeat(name, traj_len),
        traj["traj_metadata"]["episode_metadata"]["file_path"],
        tf.repeat(tf.constant("#"), traj_len),
        tf.strings.as_string(traj["traj_metadata"]["episode_metadata"]["episode_id"]),
        tf.repeat(tf.constant(":"), traj_len),
        tf.strings.as_string(tf.range(traj_len)),
    ])

    traj = {
        "observation": new_obs,
        "task": task,
        "action": tf.cast(traj["action"], tf.float32),
        "dataset_name": tf.repeat(name, traj_len),
        "frame_key": frame_key,
        "reward": reward,
        "td_mask": mask,
        "mc_return": mc_return,
        # "next_action": tf.cast(next_action, tf.float32),
        # "next_observation": next_obs,
    }

    return traj


def main(_):
    if flags.FLAGS.platform == "tpu":
        jax.distributed.initialize()

    tf.random.set_seed(jax.process_index())

    config = flags.FLAGS.config
    sharding_metadata = make_sharding(config)
    model = ModelComponents.load_static(config.resume_checkpoint_dir, sharding_metadata)
    manager = ocp.CheckpointManager(config.resume_checkpoint_dir, options=ocp.CheckpointManagerOptions())
    model.load_state(config.resume_checkpoint_step, manager)
    # Load in the image and the prompt
    prompt = "Go to the door"
    image = np.random.randn(1, 224, 224, 3)
    batch = {"task" : 
                {"language_instruction" : prompt},
             "observation": 
                {"image": image}
            }
    
    batch = traj_transforms.add_pad_mask_dict(batch)

    # Predict the output 
    predicted_actions, actions_mask, tokens = model.predict(batch, action_dim=2, action_horizon=10, return_tokens=True, include_action_tokens=False)

    print(predicted_actions.shape)

if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "configs/smoke_test.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    app.run(main)





