import os
import jax
import jax.numpy as jnp
import numpy as np
import sys
from ml_collections import config_flags, ConfigDict
import tensorflow as tf
from PIL import Image

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
from google.cloud import storage
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
    storage_client = storage.Client()
    bucket = storage_client.bucket('vlm-guidance-misc')
    blob = bucket.get_blob('3.jpg')  # use get_blob to fix generation number, so we don't get corruption if blob is overwritten while we read it.
    with blob.open() as file:
        image = Image.open(file)
    image = image.resize((224, 224))
    image = np.array(image.convert("RGB")).repeat(4, axis=0)
    print(image.shape)
    batch = {"task" : 
                {"language_instruction" : np.array([prompt.encode()]*4), 
                 "pad_mask_dict": {"language_instruction": np.array([1]*4)}},
             "observation": 
                {"image_primary": image, 
                 "pad_mask_dict": {"image_primary": np.array([1]*4, dtype=bool)}},
             "action": np.random.randn(4, 1, 2).astype(np.float64),    
            }
    # Predict the output 
    predicted_actions, actions_mask, tokens = model.predict(batch, action_dim=2, action_horizon=10, return_tokens=True, include_action_tokens=False)

    print(predicted_actions)

if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "configs/smoke_test.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    app.run(main)





