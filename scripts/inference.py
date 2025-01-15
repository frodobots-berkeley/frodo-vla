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
                {"language_instruction" : np.array([prompt.encode()])},
             "observation": 
                {"image": tf.convert_to_tensor(image, dtype=tf.int64)},
             "action": tf.convert_to_tensor(np.random.randn(1, 1, 2), dtype=tf.float32),
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





