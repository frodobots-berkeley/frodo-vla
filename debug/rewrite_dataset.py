import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
import glob 
import pickle 
import argparse
import dlimp as dl
from functools import partial
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

# import octo.data.obs_transforms as obs_transforms
# from octo.data.dataset import apply_frame_transforms

DATASETS = [
    "cory_hall",
    "go_stanford_cropped",
    "go_stanford2",
    "recon",
    "sacson",
    "scand",
    "seattle",
    "tartan_drive",
]

def lookup_in_dict(key_tensor, dictionary):
  """
  Looks up a string key tensor in a Python dictionary.

  Args:
    key_tensor: A tf.string tensor representing the key to lookup.
    dictionary: A Python dictionary with string keys.

  Returns:
    A tf.string tensor representing the value associated with the key,
    or an empty string tensor if the key is not found.
  """
  def lookup(key):
    return dictionary.get(key.decode(), "")

  return tf.py_function(
      func=lookup, 
      inp=[key_tensor], 
      Tout=tf.string
  )

# Fix issues with dataset from TFrecords 
def fix_dataset(traj, traj_info):

    breakpoint()

    # Get the metadata for this traj 
    traj_name = tf.strings.split(traj["traj_metadata"]["episode_metadata"]["file_path"], "/")[-1]
    traj_base_name = tf.strings.split(traj_name, "_start_")[0]
    traj_start = tf.cast(tf.strings.split(tf.strings.split(traj_name, "_start_")[-1], "_end_")[0], tf.int32)
    traj_end = tf.cast(tf.strings.split(tf.strings.split(traj_name, "_end_")[-1], "_")[0], tf.int32)

    # Modify the traj info for this trajectory
    curr_traj_info = lookup_in_dict(traj_base_name, traj_info)
    tf.print(curr_traj_info)

    # Check the number of non-white images in the traj
    images = traj["observation"]["image_decoded"]
    image_non_white = tf.reduce_any(tf.not_equal(images, 255), axis=-1)
    num_non_white = tf.reduce_sum(tf.cast(image_non_white, tf.float32))

    # Check two things: 
    # 1. Is the spacing between points close to that of the expected normalization factor
    # 2. Modify the yaw such that is closer to the original traj yaw

    # Check the spacing between points
    traj_pos = traj["observation"]["position"]
    traj_pos = tf.cast(traj_pos, tf.float32)
    deltas = tf.linalg.norm(traj_pos[:-1] - traj_pos[1:], axis=-1)
    spacing = tf.reduce_mean(deltas)
    normalization_factor = tf.cast(lookup_in_dict("normalization_factor", curr_traj_info), tf.float32)
    tf.print(f"Spacing for {traj_base_name} is {spacing} and normalization factor is {normalization_factor}")
    if tf.abs(spacing - normalization_factor) > 0.05:
        tf.print(f"Spacing issue for {traj_base_name} with spacing {spacing} and normalization factor {normalization_factor}")
        breakpoint()
    
    # Check the yaw
    traj_yaw = traj["observations"]["yaw"]
    non_cf_yaw = traj_yaw[:num_non_white]
    orig_yaw = curr_traj_info["yaw"]
    end = min(traj_start + num_non_white, traj_end)
    curr_orig_yaw = orig_yaw[traj_start:end+1]

    assert len(non_cf_yaw) == len(orig_yaw), f"Length mismatch for {traj_base_name}"

    # Compute the yaw of the original part of the trajectory 
    new_yaw = orig_yaw[traj_start:end + 1]

    breakpoint()
    # If the trajectory has a counterfactual, we need to generate the correct yaw for the counterfactual part
    if "cf" in traj_base_name:
        cf_start = end - num_non_white
        cf_end = traj_end
        cf_orig_yaw = orig_yaw[cf_start:cf_end + 1]
        cf_yaw = traj_yaw[cf_start:cf_end + 1]
        new_yaw = np.concatenate([new_yaw, cf_yaw])

def decode(
    obs: dict,
) -> dict:
    """Decodes images and depth images, and then optionally resizes them."""
    # just gets the part after "image_" or "depth_"

    image = obs["image"]
    if image.dtype == tf.string:
        if tf.strings.length(image) == 0:
            # this is a padding image
            image = tf.zeros((128, 128, 3), dtype=tf.uint8)
        else:
            image = tf.io.decode_image(
                image, expand_animations=False, dtype=tf.uint8
            )
    
    obs[f"image_decoded"] = image

    return obs

def apply_obs_transform(fn: Callable[[dict], dict], frame: dict) -> dict:
    frame["observation_decoded"] = fn(frame["observation"])
    return frame

def main(args):

    # Load in the dataset
    data_dir = args.data_dir
    name = args.dataset_name
    builder = tfds.builder(name, data_dir=data_dir)
    dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False)
    resize_size = (128, 128)
    num_parallel_calls = tf.data.AUTOTUNE

    # Load the dataset traj and yaw files
    traj_infos = {}
    for dataset_name in DATASETS:
        traj_info_file = f"traj_info/{dataset_name}.pkl"
        with open(traj_info_file, "rb") as f:
            traj_info = pickle.load(f)
        traj_infos.update(traj_info)

    # decode + resize images (and depth images)
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            decode,
        ),
        num_parallel_calls,
    )

    # Fix the dataset
    dataset = dataset.traj_map(partial(fix_dataset, traj_info=traj_infos), num_parallel_calls=num_parallel_calls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    main(args)