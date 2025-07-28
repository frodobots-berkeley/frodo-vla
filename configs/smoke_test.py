from ml_collections import ConfigDict
from palivla.base_config import get_config as get_base_config
from palivla.standardization_transforms import gnm_dataset_transform
from octo.data.utils.data_utils import NormalizationType
from octo.utils.spec import ModuleSpec

def get_config(variant_config: str = "smoke_test"):
    config = get_base_config(variant_config)
    config["visualizations"] = {
        "overfit_sanity_print": {
            "dataset": "overfit",
            "visualization": "viz.sanity_print",
        }
    }
    action_horizon = 8
    transform = ModuleSpec.create(gnm_dataset_transform, action_horizon=action_horizon)
    config["overfit_dataset"] = True
    config["action_tokenizer"] = f"action_tokenizer.bin(min_action_value=-1, max_action_value=1, action_vocab_size=128, action_horizon={action_horizon})"
    config["dataset_kwargs"]  = {
            "oxe_kwargs" : None, 
            "dataset_kwargs_list" : {
                    "atomic_forward_kwargs": {
                        "name": "atomic_forward_dataset",
                        "data_dir": "gs://cat-datasets",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },

            },
        "sample_weights": [1.0],
        "traj_transform_kwargs": {
                "window_size": 1, 
                "action_horizon": action_horizon,
        },
        "frame_transform_kwargs": {
               "image_augment_kwargs": {},
               "resize_size": {"primary": [224, 224]},
        },
        "balance_weights": True,
        "shuffle_buffer_size": 50000,
        "traj_transform_threads": 16,
        "traj_read_threads": 16,
    }

    return ConfigDict(config)
