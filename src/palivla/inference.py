import os
import jax
import jax.numpy as jnp
import numpy as np
import sys
import wandb
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
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
print("VISIBLE DEVICES: ", jax.devices())
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices, "GPU")
print("VISIBLE DEVICES: ", jax.devices())

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

IMAGE_SIZE = (96, 96)
CAMERA_METRICS = {"camera_height" : 0.95, # meters
                "camera_x_offset" : 0.45, # distance between the center of the robot and the forward facing camera
                "camera_matrix" : {"fx": 272.547000, "fy": 266.358000, "cx": 320.000000, "cy": 220.000000},
                "dist_coeffs" : {"k1": -0.038483, "k2": -0.010456, "p1": 0.003930, "p2": -0.001007, "k3": 0.000000}}
VIZ_IMAGE_SIZE = (480, 640)  # (height, width)

# Utility functions
def pil_to_base64(img):
    img.save("temp.jpg")
    with open("temp.jpg", "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_camera_params():
    camera_height = CAMERA_METRICS["camera_height"]
    camera_x_offset = CAMERA_METRICS["camera_x_offset"]

    fx = CAMERA_METRICS["camera_matrix"]["fx"]
    fy = CAMERA_METRICS["camera_matrix"]["fy"]
    cx = CAMERA_METRICS["camera_matrix"]["cx"]
    cy = CAMERA_METRICS["camera_matrix"]["cy"]
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    k1 = CAMERA_METRICS["dist_coeffs"]["k1"]
    k2 = CAMERA_METRICS["dist_coeffs"]["k2"]
    p1 = CAMERA_METRICS["dist_coeffs"]["p1"]
    p2 = CAMERA_METRICS["dist_coeffs"]["p2"]
    k3 = CAMERA_METRICS["dist_coeffs"]["k3"]
    dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

    return camera_matrix, dist_coeffs

def draw_trajectory(img, traj):
    # project onto the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    camera_matrix, dist_coeffs = get_camera_params()
    camera_height = CAMERA_METRICS["camera_height"]
    camera_x_offset = CAMERA_METRICS["camera_x_offset"]

    xy_coords = traj
    traj_pixels = get_pos_pixels(
        xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=False
    )
    if len(traj_pixels.shape) == 2:
        ax.plot(
            traj_pixels[:250, 0],
            traj_pixels[:250, 1],
            color="blue",
            lw=2.5,
        )

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim((0.5, VIZ_IMAGE_SIZE[1] - 0.5))
        ax.set_ylim((VIZ_IMAGE_SIZE[0] - 0.5, 0.5))
        # return the image
        plt.savefig("~/temp_viz/projected.jpg")
        out_img = Image.open("~/temp_viz/projected.jpg")
        plt.close()
        return out_img
    else:
        return None

def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.

    Args:
        xy: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients


    Returns:
        uv: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = (0, 0, 0)

    xyz[..., 0] += camera_x_offset
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)

    return uv

def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    clip: Optional[bool] = False,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        points: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients

    Returns:
        pixels: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, VIZ_IMAGE_SIZE[0]),
                    np.clip(p[1], 0, VIZ_IMAGE_SIZE[1]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
            ]
        )
    return pixels

def make_sharding(config: ConfigDict):
    mesh = MeshShardingHelper([-1], ["fsdp"])
    sharding_metadata = ShardingMetadata(
        mesh=mesh,
        model_sharding_rule=FSDPShardingRule(
            "fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"]
        ),
    )
    return sharding_metadata

def run_inference(model, prompt, image, config):

    os.makedirs("~/temp_viz", exist_ok=True)
    action_horizon = config["dataset_kwargs"]["traj_transform_kwargs"]["action_horizon"]
    image = np.expand_dims(np.array(image), 0).repeat(4, axis=0)

    batch = {"task" : 
                {"language_instruction" : np.array([prompt.encode("utf-8")]*4), 
                "pad_mask_dict": {"language_instruction": np.array([1]*4)}},
            "observation": 
                {"image_primary": image, 
                "pad_mask_dict": {"image_primary": np.array([1]*4, dtype=bool)}},
            "action": np.random.randn(4, 1, 2).astype(np.float64),    
            }
    # Predict the output 
    predicted_actions, actions_mask, tokens = model.predict(batch, action_dim=2, action_horizon=action_horizon, return_tokens=True, include_action_tokens=False)
    predicted_actions = predicted_actions[0].squeeze()
    summed_actions = np.cumsum(predicted_actions, axis=0)
    summed_actions -= summed_actions[0]

    # Plot on the image 
    viz_image = Image.fromarray(image[0]).resize(VIZ_IMAGE_SIZE)
    out = draw_trajectory(np.array(viz_image), summed_actions)
    # Plot the image and the waypoints
    fig, ax = plt.subplots(1, 2, figsize=(5, 10))
    ax[0].imshow(image[0])
    ax[0].set_title("Image")
    ax[1].plot(summed_actions[:, 0], summed_actions[:, 1])
    ax[1].set_title("Output")
    plt.savefig("~/temp_viz/inference.jpg")
    viz = {"inference": "~/temp_viz/inference.jpg", "projected": "~/temp_viz/projected.jpg"}
    return predicted_actions, viz

if __name__ == "__main__":
    config_flags.DEFINE_config_file(
            "config", "configs/smoke_test.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    flags.DEFINE_string("resume_checkpoint_dir", "gs://cat-logs/serene-field-298", "Path to the checkpoint directory.")
    flags.DEFINE_integer("resume_checkpoint_step", 90000, "Step to resume from.")
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
    model = ModelComponents.load_static(config.resume_checkpoint_dir, sharding_metadata)
    manager = ocp.CheckpointManager(config.resume_checkpoint_dir, options=ocp.CheckpointManagerOptions())
    model.load_state(config.resume_checkpoint_step, manager)
    prompt = flags.FLAGS.prompt
    obs = Image.fromarray(np.random.randn(96, 96, 3).astype(np.uint8))
    run_inference(model, prompt, obs, config)




