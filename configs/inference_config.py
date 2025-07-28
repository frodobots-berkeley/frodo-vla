from ml_collections.config_dict import placeholder, ConfigDict, FieldReference
from functools import partial
from palivla.components.model import get_default_config

placeholder(int)._value

def get_config():
    num_train_steps = FieldReference(100000, int)

    model_config = get_default_config()
    action_horizon = 8
    return ConfigDict(
        {
            "wandb_project": "vla-nav",
            "wandb_mode": "online",
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
            # Overfit
            "overfit_dataset": False,
            # For inference, we only load the weights
            "weights_only": True,
            # Training settings
            "batch_size": 192,
            "eval_batch_size": 128,
            "num_steps": num_train_steps,
            # Checkpoint settings
            "save_path": "",
            "save_interval": 1000,
            "max_to_keep": 1,
            # Multi-device settings
            "data_axis_size": 1,
            "fsdp_axis_size": -1,
            # Model
            "model_config": model_config,
            "shuffle_buffer_size": 50000,
            "num_steps": num_train_steps,
            # Sampling 
            "sampler": "greedy", 
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
                },
                "sample_weights": [],
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
