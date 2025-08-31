from octo.data.utils.data_utils import NormalizationType
from ml_collections.config_dict import placeholder, ConfigDict, FieldReference
from functools import partial
from palivla.components.model import get_default_config
from palivla.standardization_transforms import gnm_dataset_transform
from octo.utils.spec import ModuleSpec
from palivla.frodo_dataset import ActionFormat

placeholder(int)._value

def get_config():
    num_train_steps = FieldReference(100000, int)

    model_config = get_default_config()
    action_horizon = 8
    transform = ModuleSpec.create(gnm_dataset_transform, action_horizon=action_horizon)
    return ConfigDict(
        {
            "wandb_project": "vla-nav",
            "wandb_mode": "online",
            "wandb_run": "frodo_vla",
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
            "resume_checkpoint_dir": None,
            "resume_checkpoint_step": None,
            "weights_only": False,
            # Overfit
            "overfit_dataset": False,
            # Training settings
            "batch_size": 192,
            "eval_batch_size": 128,
            "num_steps": num_train_steps,
            # Checkpoint settings
            "save_path": "gs://cat-logs",
            "save_interval": 10000,
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
                "root": "gs://frodo-bucket-c2/frodobots_v2_export",
                "repo_id" : "frodobots/Berkeley-Frodobots-7K",
                "video": "gs://frodo-bucket-c2/frodobots_v2_export/videos",
                "data_dir": "gs://frodo-bucket-c2",
                "action_format": ActionFormat.WAYPOINT,
                "image_obs_keys": {"primary": "image"},
                "proprio_obs_key": "position",
                "language_key" : "language_instruction",
                "action_proprio_normalization_type": NormalizationType.NORMAL,
                "standardize_fn" : transform,   
                "force_recompute_dataset_statistics": False,
                "skip_norm": True,
                },
            "traj_transform_kwargs": {
                "window_size": 1,
                "action_horizon": action_horizon,
            },
            "frame_transform_kwargs": {
                "image_augment_kwargs": {},
                "resize_size": {"primary": [224, 224]},
            },
            "balance_weights": False,
            "shuffle_buffer_size": 50000,
            "traj_transform_threads": 16,
            "traj_read_threads": 16,
        }
    )
