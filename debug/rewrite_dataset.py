import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
import glob 
import pickle 
import argparse
import sys
import dlimp as dl
from functools import partial
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union
# import torch.multiprocessing as mp
import os.path as osp
import traceback
import tyro

# import octo.data.obs_transforms as obs_transforms
# from octo.data.dataset import apply_frame_transforms

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
print(tf.executing_eagerly())


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
def fix_traj(traj, frames, traj_info):
    
    # Get the metadata for this traj 
    traj_name = tf.strings.split(traj["traj_metadata"]["episode_metadata"]["file_path"], "/")[-1]
    tf.print(traj_name, output_stream=sys.stdout)
    traj_base_name = tf.strings.split(traj_name, "_start_")[0]
    tf.print(traj_base_name, output_stream=sys.stdout)
    traj_start = tf.cast(tf.strings.split(tf.strings.split(traj_name, "_start_")[-1], "_end_")[0], tf.int32)[0]
    tf.print(traj_start, output_stream=sys.stdout)
    traj_end = tf.cast(tf.strings.split(tf.strings.split(traj_name, "_end_")[-1], "_")[0], tf.int32)[0]
    tf.print(traj_end, output_stream=sys.stdout)

    # Modify the traj info for this trajectory
    curr_traj_info = lookup_in_dict(traj_base_name, traj_info)
    tf.print(curr_traj_info, output_stream=sys.stdout)

    # Check the number of non-white images in the traj
    images = traj["observation_decoded"]["image_decoded"]
    image_non_white = tf.reduce_any(tf.not_equal(images, 255), axis=-1)
    num_non_white = tf.cast(tf.reduce_sum(tf.cast(image_non_white, tf.float32)), tf.int32)

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
    
    # Check the yaw
    traj_yaw = traj["observation"]["yaw"]
    non_cf_yaw = traj_yaw[:, :num_non_white]
    orig_yaw = tf.cast(lookup_in_dict("yaw", curr_traj_info), tf.float32)
    end = tf.minimum(traj_start + num_non_white, traj_end)
    curr_orig_yaw = orig_yaw[:, traj_start:end+1]

    tf.debugging.assert_equal(tf.shape(non_cf_yaw, 0), tf.shape(curr_orig_yaw, 0), message=f"Length mismatch for {traj_base_name}")

    # Compute the yaw of the original part of the trajectory 
    new_yaw = orig_yaw[traj_start:end + 1]

    # If the trajectory has a counterfactual, we need to generate the correct yaw for the counterfactual part
    if tf.strings.regex_full_match(traj_name, ".*cf.*"):
        cf_start = end - num_non_white
        cf_end = traj_end
        cf_orig_yaw = orig_yaw[traj_start:cf_start]
        cf_new = tf.atan2(traj_pos[cf_start+1:, 1] - traj_pos[cf_start:-1, 1], traj_pos[cf_start+1:, 0] - traj_pos[cf_start:-1, 0]) + cf_orig_yaw[:, -1]
        new_yaw = tf.concat([new_yaw, cf_new], axis=0)
        tf.print(new_yaw)
    
    traj["observation"]["yaw"] = new_yaw
    traj["observation"]["yaw_rotmat"] = tf.stack([tf.cos(new_yaw), -tf.sin(new_yaw), tf.sin(new_yaw), tf.cos(new_yaw)], axis=-1)
    breakpoint()
    return traj
        
def decode(
    obs: dict,
) -> dict:
    """Decodes images and depth images, and then optionally resizes them."""

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

def work_fn(worker_id, path_shard, output_dir, traj_infos, pbar_queue=None):
    try:
        tf.config.set_visible_devices([], "GPU")
        # torch.cuda.set_device(worker_id)
        paths = path_shards[worker_id]
        for path in paths:

            writer = tf.io.TFRecordWriter(osp.join(output_dir, osp.basename(path)))
            dataset = tf.data.TFRecordDataset([path]).map(features.deserialize_example)

            for example in dataset:
                traj = example["steps"].batch(int(1e9)).get_single_element()
                traj = tf.nest.map_structure(lambda x: x.numpy()[::subsample], traj)
                del example["steps"]
                example = tf.nest.map_structure(lambda x: x.numpy(), example)
                frames = traj["observation"]["image"]
                breakpoint()
                traj = fix_traj(traj, frames, traj_infos)
                
                # serialize and write
                example["steps"] = traj
                writer.write(new_features.serialize_example(example))
                pbar_queue.put(1)
            writer.close()
    except Exception:
        pbar_queue.put(traceback.format_exc())

def main(args):

    # Load in the dataset
    data_dir = args.data_dir
    name = args.dataset_name
    output_dir = args.output_dir
    num_workers = args.num_workers

    builder = tfds.builder(name, data_dir=data_dir)
    output_dir = osp.join(output_dir, *str(builder.data_path).split(osp.sep)[-2:])
    tf.io.gfile.makedirs(output_dir)
    paths = tf.io.gfile.glob(f"{builder.data_path}/*.tfrecord*")
    path_shards = np.array_split(paths, num_workers)
    
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

    # Write dataset as RLDS
    features_spec = builder.info.features
    
    if num_workers == 1:
        worker_id = 0
        work_fn(worker_id, path_shards, output_dir, traj_infos)
    else:
        ctx = mp.get_context("spawn")
        pbar_queue = ctx.SimpleQueue()
        
        pcontext = mp.spawn(
            fix_dataset,
            nprocs=num_workers,
            args=(
                path_shards,
                output_dir,
                traj_infos,
                pbar_queue,
            ),
            join=False,
        )
        pbar = tqdm.tqdm(total=builder.info.splits["all"].num_examples)
        n_running = num_workers
        while True:
            n = pbar_queue.get()
            if isinstance(n, str):
                print(n)
                break
            elif n is None:
                n_running -= 1
                if n_running == 0:
                    break
            else:
                pbar.update(n)
        pbar.close()
        pbar_queue.close()
        if not pcontext.join(timeout=5):
            raise RuntimeError("Failed to join processes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="gs://vlm-guidance-data/test")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    main(args)