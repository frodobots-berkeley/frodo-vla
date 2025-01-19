import os
import jax
import jax.numpy as jnp
import numpy as np
import sys
from ml_collections import config_flags, ConfigDict
import tensorflow as tf
from PIL import Image
from google.cloud import logging
from google.cloud import storage
import matplotlib.pyplot as plt
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
# from google.cloud import storage
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

print(logging.Client())

# Load data config
METRIC_WAYPOINT_SPACING = {
    "cory_hall": 0.06,
    "go_stanford": 0.12,
    "recon": 0.25,
    "sacson": 0.255,
    "scand": 0.38,
    "seattle": 0.35,
    "tartan_drive": 0.72,
}

def make_sharding(config: ConfigDict):
    mesh = MeshShardingHelper([-1], ["fsdp"])
    sharding_metadata = ShardingMetadata(
        mesh=mesh,
        model_sharding_rule=FSDPShardingRule(
            "fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"]
        ),
    )
    return sharding_metadata

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
    action_horizon = config["dataset_kwargs"]["traj_transform_kwargs"]["action_horizon"]
    storage_client = storage.Client()
    bucket = storage_client.bucket('vlm-guidance-misc')
    blob = bucket.get_blob('3.jpg')  # use get_blob to fix generation number, so we don't get corruption if blob is overwritten while we read it.
    with blob.open(mode="rb") as file:
        image = Image.open(file)
        image = image.resize((224, 224))
        image = np.expand_dims(np.array(image.convert("RGB")), 0).repeat(4, axis=0)
        batch = {"task" : 
                    {"language_instruction" : np.array([prompt.encode()]*4), 
                    "pad_mask_dict": {"language_instruction": np.array([1]*4)}},
                "observation": 
                    {"image_primary": image, 
                    "pad_mask_dict": {"image_primary": np.array([1]*4, dtype=bool)}},
                "action": np.random.randn(4, 1, 2).astype(np.float64),    
                }
        # Predict the output 
        predicted_actions, actions_mask, tokens = model.predict(batch, action_dim=2, action_horizon=action_horizon, return_tokens=True, include_action_tokens=False)
        predicted_actions = predicted_actions[0].squeeze()
        summed_actions = np.cumsum(predicted_actions*METRIC_WAYPOINT_SPACING["sacson"], axis=1)
        summed_actions -= summed_actions[0]
        print(summed_actions)

    # Plot the image and the waypoints
    fig, ax = plt.subplots(1, 2, figsize=5, 10))
    ax[0].imshow(image[0])
    ax[0].set_title("Image")
    ax[1].plot(summed_actions[:, 0], summed_actions[:, 1])


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "configs/smoke_test.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    app.run(main)





