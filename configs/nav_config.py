from octo.data.utils.data_utils import NormalizationType
from ml_collections.config_dict import placeholder, ConfigDict, FieldReference

from palivla.components.model import get_default_config
from octo.data.oxe.oxe_standardization_transforms import gnm_dataset_transform
from octo.utils.spec import ModuleSpec

placeholder(int)._value

def get_config():
    num_train_steps = FieldReference(100000, int)

    model_config = get_default_config()

    transform = ModuleSpec.create(gnm_dataset_transform)
    return ConfigDict(
        {
            "wandb_project": "vla-nav",
            "wandb_mode": "online",
            #Tokenizers
            "language_tokenizer": "google/paligemma-3b-mix-224",
            "action_tokenizer": "action_tokenizer.bin(min_action_value=-1, max_action_value=1, action_vocab_size=512)",
            "sequence_builder": "sequence_builder.default(prompt_pad_length=50, gen_pad_length=10)",
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
            # Overfit
            "overfit_dataset": False,
            # Training settings
            "batch_size": 192,
            "eval_batch_size": 128,
            "num_steps": num_train_steps,
            # Checkpoint settings
            "save_path": "gs://vlm-guidance-logs",
            "save_interval": 10000,
            "max_to_keep": 1,
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
                    "base_learning_rate": 1e-6,
                },
            },
            "dataset_kwargs": {
                "oxe_kwargs": None,
                "dataset_kwargs_list": {
                    "lcbc_kwargs": {
                        "name": "lcbc_dataset",
                        "data_dir": "gs://vlm-guidance-data",
                        "image_obs_keys": {"primary": "image"},
                        "proprio_obs_key": "position",
                        "language_key" : "language_instruction",
                        "force_recompute_dataset_statistics": False,
                        "action_proprio_normalization_type": NormalizationType.NORMAL,
                        "standardize_fn" : transform,   
                    },
                },
                "sample_weights": [1.0],
                "traj_transform_kwargs": {
                    "window_size": 1,
                    "action_horizon": 1,
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
