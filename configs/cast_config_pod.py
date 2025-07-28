from octo.data.utils.data_utils import NormalizationType
from ml_collections.config_dict import placeholder, ConfigDict, FieldReference
from functools import partial
from palivla.components.model import get_default_config
from palivla.standardization_transforms import gnm_dataset_transform
from octo.utils.spec import ModuleSpec

placeholder(int)._value

def get_config():
    num_train_steps = FieldReference(300000, int)

    model_config = get_default_config()
    action_horizon = 8
    num_vms_in_pod = 4  # Number of TPU VMs in the pod, adjust as needed
    transform = ModuleSpec.create(gnm_dataset_transform, action_horizon=action_horizon)
    return ConfigDict(
        {
            "wandb_project": "vla-nav",
            "wandb_mode": "online",
            "wandb_run": "filtered_atomic_skip_norm",
            #Tokenizers
            "language_tokenizer": "google/paligemma-3b-mix-224",
            "action_tokenizer": f"action_tokenizer.bin(min_action_value=-1, max_action_value=1, action_vocab_size=128, action_horizon={action_horizon})",
            "sequence_builder": "sequence_builder.default(prompt_pad_length=100, gen_pad_length=20)",
            # Initialization
            "load_fns": [
                (
                    "load.paligemma_weights",
                    {
                        "hf_repo": "google/paligemma-3b-mix-224-jax",
                        "path": "paligemma-3b-mix-224.npz",
                    },
                )
            ],
            "resume_checkpoint_dir": "gs://cat-logs/filtered_atomic_skip_norm_2025_03_31_16_08_10",
            "resume_checkpoint_step": 55000,
            "weights_only": False,
            # Overfit
            "overfit_dataset": False,
            # Training settings
            "batch_size": 192 * num_vms_in_pod,  # Adjust batch size based on number of VMs
            "eval_batch_size": 128,
            "num_steps": num_train_steps,
            # Checkpoint settings
            "save_path": "gs://cat-logs",
            "save_interval": 5000,
            "max_to_keep": 10,
            # Multi-device settings
            "data_axis_size": 1,
            "fsdp_axis_size": -1,
            # Model
            "model_config": model_config,
            "shuffle_buffer_size": 50000,
            "num_steps": num_train_steps,
            # Logging and visualization
            "eval_interval": 100,
            "log_interval": 1,
            # Optimizer settings
            "optimizer": {
                "name": "optimizer.default_optimizer",
                "kwargs": {
                    "optimizer": "adamw",
                    "num_train_steps": num_train_steps,
                    "base_learning_rate": 1e-4,
                },
            },
            "dataset_kwargs": {
                "oxe_kwargs": None,
                "dataset_kwargs_list": {
                    # "lcbc_kwargs": {
                    #     "name": "lcbc_orig_dataset_128",
                    #     "data_dir": "gs://cat-datasets/cleaned",
                    #     "image_obs_keys": {"primary": "image"},
                    #     "proprio_obs_key": "position",
                    #     "language_key" : "language_instruction",
                    #     "action_proprio_normalization_type": NormalizationType.NORMAL,
                    #     "standardize_fn" : transform,   
                    #     "force_recompute_dataset_statistics": False,
                    # },
                    "lcbc_filtered_kwargs": {
                        "name": "lcbc_filtered_128",
                        "data_dir": "gs://cat-datasets/cleaned",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },
                    "lcbc_filtered_v2_kwargs": {
                        "name": "lcbc_filtered_v2_dataset",
                        "data_dir": "gs://cat-datasets/cleaned",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },
                    "cf_kwargs": {
                        "name": "cf_v2_dataset_128",
                        "data_dir": "gs://cat-datasets/cleaned",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },
                    "cf_v3_kwargs": {
                        "name": "cf_v3_dataset_128",
                        "data_dir": "gs://cat-datasets/cleaned",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },
                    "cf_v4_kwargs": {
                        "name": "cf_v4_dataset",
                        "data_dir": "gs://cat-datasets/cleaned",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },
                    # "outdoor_kwargs": {
                    #     "name": "outdoor_dataset_128",
                    #     "data_dir": "gs://cat-datasets/cleaned",
                    #     "image_obs_keys": {"primary": "image"},
                    #     "proprio_obs_key": "position",
                    #     "language_key" : "language_instruction",
                    #     "action_proprio_normalization_type": NormalizationType.NORMAL,
                    #     "standardize_fn" : transform,   
                    #     "force_recompute_dataset_statistics": False,
                    # },
                    "outdoor_filtered_kwargs": {
                        "name": "outdoor_filtered_dataset_128",
                        "data_dir": "gs://cat-datasets/cleaned",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },
                    "outdoor_filtered_v2_kwargs": {
                        "name": "outdoor_filtered_v2_dataset",
                        "data_dir": "gs://cat-datasets/cleaned",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },
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
                    "atomic_turn_left_kwargs": {
                        "name": "atomic_turn_left_dataset",
                        "data_dir": "gs://cat-datasets",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },
                    "atomic_turn_right_kwargs": {
                        "name": "atomic_turn_right_dataset",
                        "data_dir": "gs://cat-datasets",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },
                    "atomic_adjust_left_kwargs": {
                        "name": "atomic_adjust_left_dataset",
                        "data_dir": "gs://cat-datasets",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },
                    "atomic_adjust_right_kwargs": {
                        "name": "atomic_adjust_right_dataset",
                        "data_dir": "gs://cat-datasets",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                        "force_recompute_dataset_statistics": False,
                        "skip_norm": True,
                    },
                    "atomic_stop_kwargs": {
                        "name": "atomic_stop_dataset",
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
                "sample_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.7, 0.7, 0.7, 0.7, 0.9],
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
            },
        }
    )
