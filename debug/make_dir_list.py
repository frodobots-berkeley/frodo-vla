import glob
import os
import sys
import numpy as np
import pickle as pkl
from tqdm import tqdm


# Get all the traj files corresponding to a given dataset and save then in a text file
METRIC_WAYPOINT_SPACING = {
    "cory_hall": 0.06,
    "go_stanford_cropped": 0.12,
    "go_stanford2": 0.12,
    "recon": 0.25,
    "sacson": 0.255,
    "scand": 0.38,
    "seattle": 0.35,
    "tartan_drive": 0.72,
}
ROOT_DIR = "/hdd/LLLwL_datasets/gnm_dataset"
os.makedirs("traj_info", exist_ok=True)
def main():

    for dataset_name in METRIC_WAYPOINT_SPACING.keys():
        traj_files = glob.glob(f"{ROOT_DIR}/{dataset_name}/*")
        print(f"Found {len(traj_files)} traj files for {dataset_name}")

        # Save the traj files in a pkl file
        info_dict = {}
        for traj_file in tqdm(traj_files):
            if not os.path.isdir(traj_file):
                continue
            print(traj_file)
            traj_file_root = traj_file.split("/")[-1] 
            traj_file_pickle = f"{traj_file}/traj_data_language.pkl"
            traj_data = np.load(traj_file_pickle, allow_pickle=True)
            yaw = traj_data["yaw"]
            info_dict[traj_file_root] = {"yaw": yaw, 
                                         "normalization_factor": METRIC_WAYPOINT_SPACING[dataset_name]}
        with open(f"traj_info/{dataset_name}.pkl", "wb") as f:
            pkl.dump(info_dict, f)
            

if __name__ == "__main__":
    main()