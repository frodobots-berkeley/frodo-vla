import os
import matplotlib.pyplot as plt
import datetime, time
import shutil

from big_vision.utils import Registry
from palivla.components.action_tokenizer import ActionTokenizer, DCTActionTokenizer
from palivla.components.model import PaliVLAModel
from palivla.components.sequence_builder import SequenceBuilder
from palivla.components.train_state import ShardingMetadata
from jax_torch_adapter import TorchToJaxDataset
from palivla.frodo_dataset import FrodbotDataset_MBRA

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
import tqdm
import pickle as pkl
from absl import app, flags
from absl import logging as absl_logging
from flax.core.frozen_dict import freeze
from ml_collections import ConfigDict, config_flags
from scalax.sharding import FSDPShardingRule, MeshShardingHelper
from transformers import AutoTokenizer

import wandb
import palivla.load_fns
from palivla.dataset import make_base_dataset
from palivla.model_components import ModelComponents
from palivla.optimizer import make_optimizer
from palivla.spec import ModuleSpec, OptimizerSpec
from palivla.utils import host_broadcast_str

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")


def make_sharding(config: ConfigDict):
    mesh = MeshShardingHelper([-1], ["fsdp"])
    sharding_metadata = ShardingMetadata(
        mesh=mesh,
        model_sharding_rule=FSDPShardingRule(
            "fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"]
        ),
    )
    return sharding_metadata


def create_model(config: ConfigDict, sharding_metadata: ShardingMetadata):
    example_batch = {
        "sensors": {
            "image_primary": jax.ShapeDtypeStruct(
                shape=(1, 224, 224, 3), dtype=jnp.uint8
            ),
            "proprio": jax.ShapeDtypeStruct(shape=(1, 7), dtype=jnp.float32),
        },
        "sensors_mask": {
            "image_primary": jax.ShapeDtypeStruct(
                shape=(1, 224, 224, 3), dtype=jnp.bool_
            ),
            "proprio": jax.ShapeDtypeStruct(shape=(1, 7), dtype=jnp.bool_),
        },
        "prompt": {
            "tokens": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.int32),
            "mask": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
            "mask_ar": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
            "mask_loss": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.float32),
        },
        "gen": {
            "tokens": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.int32),
            "mask": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
            "mask_ar": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
            "mask_loss": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.float32),
        },
    }

    language_tokenizer = AutoTokenizer.from_pretrained(config.language_tokenizer)
    action_tokenizer: ActionTokenizer = Registry.lookup(config.action_tokenizer)()
    sequence_builder: SequenceBuilder = Registry.lookup(config.sequence_builder)()
    
    if isinstance(action_tokenizer, DCTActionTokenizer):
        raise NotImplementedError(
            "DCTActionTokenizer is not yet implemented in the training script. Please use a different action tokenizer."
        )
    
    # Add the extra tokens to the language tokenizer for actions
    extra_tokens = [
        "<begin_of_action>",
    ] + [f"<act{i}>" for i in range(action_tokenizer.vocab_size)]
    language_tokenizer.add_tokens(extra_tokens)
    language_tokenizer.add_bos_token = False

    # Set up the model
    model_config = config.model_config.to_dict()
    model_config["llm_spec"]["config"]["vocab_size"] = len(language_tokenizer)
    model_spec = ModuleSpec(
        PaliVLAModel,
        freeze(model_config),
    )
    optimizer_spec = OptimizerSpec.create(
        make_optimizer,
        config.optimizer.kwargs.to_dict(),
    )

    return ModelComponents.initialize(
        model_spec=model_spec,
        optimizer_spec=optimizer_spec,
        seed=config.get("seed", 0),
        language_tokenizer=language_tokenizer,
        action_tokenizer=action_tokenizer,
        sequence_builder=sequence_builder,
        sharding_metadata=sharding_metadata,
        example_batch=(example_batch["sensors"], example_batch["sensors_mask"], example_batch["prompt"], example_batch["gen"]),
    )


def main(_):
    if flags.FLAGS.platform == "tpu":
        jax.distributed.initialize()

    # Turn off debug logs
    tf.get_logger().setLevel("WARNING")
    absl_logging.set_verbosity(absl_logging.WARNING)

    tf.random.set_seed(jax.process_index())

    config = flags.FLAGS.config

    sharding_metadata = make_sharding(config)

    if config.resume_checkpoint_dir is not None:
        # Load the model from a checkpoint
        model = ModelComponents.load_static(
            config.resume_checkpoint_dir, sharding_metadata
        )
        restore_manager = ocp.CheckpointManager(
            config.resume_checkpoint_dir, options=ocp.CheckpointManagerOptions()
        )
        model.load_state(config.resume_checkpoint_step, restore_manager)
    else:
        pass
        # Otherwise, create the model from scratch and apply any load_fns
        # model = create_model(config, sharding_metadata)
        # for load_fn, load_fn_kwargs in config.load_fns:
        #     load_fn = Registry.lookup(load_fn)
        #     load_fn(model, **load_fn_kwargs)

    print("Loading dataset...")
    # Make the basic dataset
    torch_train_ds = FrodbotDataset_MBRA(
        repo_id=config.dataset_kwargs.repo_id,
        video=config.dataset_kwargs.video,
        root=config.dataset_kwargs.root,
        split="train",
        action_format=config.dataset_kwargs.action_format,
        action_horizon=config.dataset_kwargs.action_horizon,
        action_spacing=config.dataset_kwargs.action_spacing,
        goal_horizon=config.dataset_kwargs.goal_horizon,
        context_size=config.dataset_kwargs.context_size,
        context_spacing=config.dataset_kwargs.context_spacing,
        dataset_framerate=config.dataset_kwargs.dataset_framerate,
        image_size=tuple(config.dataset_kwargs.image_size),
        image_transforms=None,  # your current transform pipeline
        sacson=getattr(config.dataset_kwargs, "sacson", False),
    )

    train_ds = TorchToJaxDataset(
        dataset=torch_train_ds,
        sampler=sampler_or_none,
        num_workers=getattr(config, "num_workers", 8),  # tune
        pin_memory=True,
        seed=getattr(config, "seed", None),
    )

    # Construct the final dataset
    # We need to do this after the model is constructed, since we need to have a tokenizer
    per_host_train_batch_size = config.batch_size // jax.process_count()

    def make_training_batch(batch):
        return batch

    train_it = train_ds.batch(
        per_host_train_batch_size,
        shard_for_pmap=True,   # or False for pjit
        prefetch_host=2,
        prefetch_device=2,
    ).iterator()

    # W&B setup
    if jax.process_index() == 0:
        wandb_kwargs = {
            "project": config.wandb_project,
            "tags": [],
            "mode": config.wandb_mode,
            "name": config.get("wandb_run", "run") + "_" + time.strftime("%Y_%m_%d_%H_%M_%S"),
        }

        wandb.init(**wandb_kwargs)
        wandb.config.update(config.to_dict())

        run_name = wandb.run.name
    else:
        run_name = None

    run_name = host_broadcast_str(run_name)

    if config.save_path is not None:
        checkpoint_save_path = tf.io.gfile.join(config.save_path, run_name)

        checkpoint_save_manager = ocp.CheckpointManager(
            checkpoint_save_path,
            options=ocp.CheckpointManagerOptions(max_to_keep=config.max_to_keep),
        )

        model.save_static(tf.io.gfile.join(checkpoint_save_path))

    wandb_logs = []

    # Main training loop
    start_step = model.train_state.step.item()

    if config.overfit_dataset:
        batch = next(train_it)
    
    if config.visualize:
        # Create a directory for saving images
        if jax.process_index() == 0:
            if os.path.exists("images"):
                shutil.rmtree("images")
            os.makedirs("images", exist_ok=True)

    with tqdm.trange(
        start_step, config.num_steps, desc="Training", dynamic_ncols=True
    ) as pbar:
        for i in pbar:
            if not config.overfit_dataset:
                batch = next(train_it)
            info = model.train_step(batch)
            info = jax.device_get(info)
            wandb_logs.append(info)
            pbar.set_postfix(
                loss=f"{info['loss']:.4f}",
            )
            
            if (i + 1) % config.eval_interval == 0:

                # Get eval info
                eval_data = model.eval_step(batch)
                eval_info = eval_data["eval_info"]
                eval_plots = eval_data["eval_data"]

                # Select random subset of the batch to visualize
                if config.visualize:
                    if jax.process_index() == 0:
                        print("Visualizing evaluation results...")
                        
                        wandb_list = []
                        idxs = np.random.choice(np.arange(eval_plots["pred_actions"].shape[0]//jax.process_count()), 5)
                        
                        # Get the ground truth and predicted actions
                        gt_viz = eval_plots["gt_actions"][idxs, ...]
                        gt_viz = np.cumsum(gt_viz, axis=1)
                        gt_viz = gt_viz - gt_viz[:, 0, :].reshape(-1, 1, model.action_tokenizer.action_dim)

                        pred_viz = eval_plots["pred_actions"][idxs, ...]
                        pred_viz = np.cumsum(pred_viz, axis=1)
                        try:
                            pred_viz = pred_viz - pred_viz[:, 0, :].reshape(-1, 1, model.action_tokenizer.action_dim)
                        except:
                            pred_viz = pred_viz - pred_viz[:, 0, :].reshape(-1, 1, 2)
                        
                        context = batch["observation"]["image_primary"][idxs, ...]
                        prompts = [model.sequence_builder.prepare_prompt(p) for p in batch["task"]["language_instruction"][idxs]]
                        for j in range(pred_viz.shape[0]):
                            fig, ax = plt.subplots(1,2)
                            ax[0].plot(gt_viz[j,:,0], gt_viz[j,:,1], 'r')
                            ax[0].plot(gt_viz[j,-1,0], gt_viz[j,-1,1], 'ro')
                            ax[0].plot(pred_viz[j,:,0], pred_viz[j,:,1], 'b')
                            ax[0].plot(gt_viz[j,-1,0], gt_viz[j,-1,1], 'ro')
                            ax[1].imshow(context[j, ...].squeeze(0))
                            ax[1].set_title(prompts[j])
                            plt.legend()
                            save_path = f"images/eval_gt_{i+1}_{j}.png"
                            plt.savefig(save_path)
                            wandb_list.append(wandb.Image(save_path))
                            plt.close()
                            
                            wandb.log({"action_prediction": wandb_list}, commit=False)
                            wandb.log(eval_info, step=i + 1, commit=False)
                            with open("batch.pkl", "wb") as f:
                                pkl.dump(batch, f)
                            wandb.save("batch.pkl")

            if (i + 1) % config.log_interval == 0:
                avg_info = jax.tree.map(
                    lambda *xs: np.mean(np.stack(xs), axis=0), *wandb_logs
                )
                if jax.process_index() == 0:
                    wandb.log(avg_info, step=i + 1)
                wandb_logs = []

            if (i + 1) % config.save_interval == 0:
                if config.save_path is not None:
                    model.save_state(i + 1, checkpoint_save_manager)

    if config.save_path is not None:
        checkpoint_save_manager.wait_until_finished()


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "configs/smoke_test.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    app.run(main)
