import os
import jax
import jax.numpy as jnp
import numpy as np
import sys
import wandb
import time
import cv2
from ml_collections import config_flags, ConfigDict
import tensorflow as tf
from PIL import Image
from typing import Optional, List
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

print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[:1], "GPU")
print("JAX VISIBLE DEVICES: ", jax.devices())

# Utility functions
def pil_to_base64(img):
    img.save("temp.jpg")
    with open("temp.jpg", "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def make_sharding(config: ConfigDict):
    mesh = MeshShardingHelper([-1], ["fsdp"])
    sharding_metadata = ShardingMetadata(
        mesh=mesh,
        model_sharding_rule=FSDPShardingRule(
            "fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"]
        ),
    )
    return sharding_metadata

def run_inference(model, prompt, image, config, inference_device="gpu"):

    if config.get("inference_device") is not None:
        infererence_device = config["inference_device"]

    os.makedirs("~/temp_viz", exist_ok=True)
    action_horizon = config["dataset_kwargs"]["traj_transform_kwargs"]["action_horizon"]

    if inference_device == "tpu":
        image = np.expand_dims(np.array(image), 0).repeat(4, axis=0)

        batch = {"task" : 
                    {"language_instruction" : np.array([prompt.encode("utf-8")]*4), 
                    "pad_mask_dict": {"language_instruction": np.array([1]*4)}},
                "observation": 
                    {"image_primary": image, 
                    "pad_mask_dict": {"image_primary": np.array([1]*4, dtype=bool)}},
                "action": np.random.randn(4, 1, 2).astype(np.float64),    
                }
    else:
        image = np.expand_dims(np.array(image), 0)

        batch = {"task" : 
                    {"language_instruction" : np.array([prompt.encode("utf-8")]), 
                    "pad_mask_dict": {"language_instruction": np.array([1])}},
                "observation": 
                    {"image_primary": image, 
                    "pad_mask_dict": {"image_primary": np.array([1], dtype=bool)}},
                "action": np.random.randn(1, 1, 2).astype(np.float64),    
                }

    # Predict the output 
    if config.get("sampler") is not None:
        sampler = config["sampler"]
        if sampler == "greedy":
            temperature = None
    else:
        sampler = "greedy"
        temperature = None
    predicted_actions, actions_mask, tokens = model.predict(batch, action_dim=2, action_horizon=action_horizon, return_tokens=True, include_action_tokens=False, sampler=sampler, temperature=temperature)

    return predicted_actions

if __name__ == "__main__":
    config_flags.DEFINE_config_file(
            "config", "configs/smoke_test.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    flags.DEFINE_string("resume_checkpoint_dir", None, "Path to the checkpoint directory.")
    flags.DEFINE_integer("resume_checkpoint_step", None, "Step to resume from.")
    flags.DEFINE_string("prompt", "", "Prompt to generate action from.")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

    FLAGS = flags.FLAGS
    FLAGS(sys.argv) 
    config = flags.FLAGS.config

    print()
    print("Config:", config)

    # Overwrite the config with the one from input
    config.resume_checkpoint_dir = f"{flags.FLAGS.resume_checkpoint_dir}"
    config.resume_checkpoint_step = flags.FLAGS.resume_checkpoint_step

    input_prompt = flags.FLAGS.prompt

    if flags.FLAGS.platform == "tpu":
        jax.distributed.initialize()
    sharding_metadata = make_sharding(config)

    print("Loading model...", config.resume_checkpoint_dir)
    model = ModelComponents.load_static(config.resume_checkpoint_dir, sharding_metadata, weights_only=config.weights_only)
    manager = ocp.CheckpointManager(config.resume_checkpoint_dir, options=ocp.CheckpointManagerOptions())
    model.load_state(config.resume_checkpoint_step, manager, weights_only=config.weights_only)
    prompt = flags.FLAGS.prompt
    

    for i in range(10):
        start_time = time.time()
        obs = Image.fromarray(np.random.randn(96, 96, 3).astype(np.uint8))
        predicted_actions, viz = run_inference(model, prompt, obs, config)
        print(f"Inference took: {time.time() - start_time} seconds")
        print("Predicted actions: ", predicted_actions)
        print("Viz: ", viz)
        print("Done!")




