import os
import numpy as np
import sys
import wandb
import cv2
import time
from ml_collections import config_flags, ConfigDict
import tensorflow as tf
from PIL import Image
from typing import Optional, List
import matplotlib.pyplot as plt
sys.path.append(".")
import numpy as np
from absl import app, flags, logging as absl_logging
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import ngrok
import base64
from io import BytesIO
from PIL import Image

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
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="vla-nav-inference",
    mode="online",
)

app = Flask(__name__)
run_with_ngrok(app)

config = None
model = None
avg_time = []

@app.route('/gen_action', methods=["POST"])
def gen_action():
    global config, model, run

    # If first time getting inference, load the model
    if model is None: 
        FLAGS = flags.FLAGS
        FLAGS(sys.argv) 
        config = flags.FLAGS.config

        # Overwrite the config with the one from input
        config.resume_checkpoint_dir = f"gs://vlm-guidance-logs/{flags.FLAGS.resume_checkpoint_dir}"
        config.resume_checkpoint_step = flags.FLAGS.resume_checkpoint_step

        input_prompt = flags.FLAGS.prompt

        if flags.FLAGS.platform == "tpu":
            jax.distributed.initialize()
        sharding_metadata = make_sharding(config)

        print("\nLoading model...", config.resume_checkpoint_dir)
        model = ModelComponents.load_static(config.resume_checkpoint_dir, sharding_metadata)
        manager = ocp.CheckpointManager(config.resume_checkpoint_dir, options=ocp.CheckpointManagerOptions())
        model.load_state(config.resume_checkpoint_step, manager)
        print("\nModel loaded!")

    # Receive data 
    data = request.get_json()
    obs_data = base64.b64decode(data['obs'])
    obs = Image.open(BytesIO(obs_data))
    api_prompt = data['prompt']

    if api_prompt != "":
        prompt = api_prompt
    else:
        prompt = input_prompt

    print(f"Prompt: {prompt}")

    # Run inference
    start_time = time.time()
    action, viz = run_inference(model, prompt, obs, config)
    
    run_time = time.time() - start_time
    avg_time.append(run_time)

    print(f"Avg. run time: {np.array(avg_time).mean()}s")
    
    viz = {k: wandb.Image(v) for k, v in viz.items()}
    run.log(viz)
    response = jsonify(action=action.tolist())
    return response

if __name__ == "__main__":
    # CLI FLAGS
    config_flags.DEFINE_config_file(
            "config", "configs/smoke_test.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    flags.DEFINE_string("resume_checkpoint_dir", "gs://vlm-guidance-logs/pleasant-hill-251", "Path to the checkpoint directory.")
    flags.DEFINE_integer("resume_checkpoint_step", 10000, "Step to resume from.")
    flags.DEFINE_string("prompt", "", "Prompt to generate action from.")
    app.run()