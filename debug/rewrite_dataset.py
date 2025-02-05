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
from octo.data.dataset import apply_frame_transforms

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


# Fix issues with dataset from TFrecords 
def fix_dataset(traj, traj_info):

    breakpoint()

    # Get the metadata for this traj 
    traj_name = traj["traj_metadata"]["episode_metadata"]["file_path"].split("/")[-1]
    traj_base_name = traj_name.split("_start_")[0]
    traj_start = int(traj_name.split("_start_")[-1].split("_end_")[0])
    traj_end = int(traj_name.split("_end_")[-1].split("_")[0])

    # Modify the traj info for this trajectory
    curr_traj_info = traj_info[traj_base_name]

    # Check the number of non-white images in the traj
    images = traj["observation"]["image"]
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
    normalization_factor = curr_traj_info["normalization_factor"]
    if tf.abs(spacing - normalization_factor) > 0.05:
        print(f"Spacing issue for {traj_base_name} with spacing {spacing} and normalization factor {normalization_factor}")
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

def decode_and_resize(
    obs: dict,
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]],
) -> dict:
    """Decodes images and depth images, and then optionally resizes them."""
    # just gets the part after "image_" or "depth_"
    image_names = {key[6:] for key in obs if key.startswith("image")}

    if isinstance(resize_size, tuple):
        resize_size = {name: resize_size for name in image_names}
    if isinstance(depth_resize_size, tuple):
        depth_resize_size = {name: depth_resize_size for name in depth_names}

    for name in image_names:
        if name not in resize_size:
            logging.warning(
                f"No resize_size was provided for image_{name}. This will result in 1x1 "
                "padding images, which may cause errors if you mix padding and non-padding images."
            )
        image = obs[f"image_{name}"]
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                # this is a padding image
                image = tf.zeros((*resize_size.get(name, (1, 1)), 3), dtype=tf.uint8)
            else:
                image = tf.io.decode_image(
                    image, expand_animations=False, dtype=tf.uint8
                )
        elif image.dtype != tf.uint8:
            raise ValueError(
                f"Unsupported image dtype: found image_{name} with dtype {image.dtype}"
            )
        if name in resize_size:
            image = dl.transforms.resize_image(image, size=resize_size[name])
        obs[f"image_{name}"] = image

    return obs

def apply_obs_transform(fn: Callable[[dict], dict], frame: dict) -> dict:
    frame["observation"] = dl.vmap(fn)(frame["observation"])
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
            partial(
                decode_and_resize,
                resize_size=resize_size,
                depth_resize_size=resize_size,
            ),
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