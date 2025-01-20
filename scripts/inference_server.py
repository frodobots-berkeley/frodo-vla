import os
import numpy as np
import sys
import wandb
import cv2
from ml_collections import config_flags, ConfigDict
import tensorflow as tf
from PIL import Image
from typing import Optional, List
import matplotlib.pyplot as plt
sys.path.append(".")
import numpy as np
from absl import app, flags, logging as absl_logging
from flask import Flask, request, jsonify

# Google
from google.cloud import logging
from google.cloud import storage

# Palivla
from palivla.model_components import ModelComponents
from palivla.optimizer import make_optimizer
from palivla.spec import ModuleSpec, OptimizerSpec
from palivla.utils import host_broadcast_str
from palivla.inference import (
    METRIC_WAYPOINT_SPACING, 
    IMAGE_SIZE, 
    CAMERA_METRICS,
    VIZ_IMAGE_SIZE
)
from palivla.components.train_state import ShardingMetadata
from palivla.inference import run_inference, pil_to_base64, make_sharding
from octo.data import traj_transforms

# Jax imports
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import orbax.checkpoint as ocp
from scalax.sharding import (
    MeshShardingHelper,
    FSDPShardingRule,
    PartitionSpec,
)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")

tf.random.set_seed(jax.process_index())
model = None
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="vla-nav-inference",
    mode="online",
)

app = Flask(__name__)

@app.route('/gen_action', methods=["POST"])
def gen_action():
    global config, model

    # If first time getting inference, load the model
    if model is None: 
        config = flags.FLAGS.config
        if flags.FLAGS.platform == "tpu":
            jax.distributed.initialize()
        sharding_metadata = make_sharding(config)
        model = ModelComponents.load_static(config.resume_checkpoint_dir, sharding_metadata)
        manager = ocp.CheckpointManager(config.resume_checkpoint_dir, options=ocp.CheckpointManagerOptions())
        model.load_state(config.resume_checkpoint_step, manager)

    # Receive data 
    data = request.get_json()
    obs = base64.b64decode(data['obs'])
    prompt = data['prompt']

    # Run inference
    action = run_inference(model, obs, prompt)
    response = jsonify(action=action)
    return response

if __name__ == "__main__":
    # CLI FLAGS
    config_flags.DEFINE_config_file(
            "config", "configs/smoke_test.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    app.run(host="0.0.0.0", port=5001)