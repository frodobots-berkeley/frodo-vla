from ml_collections import ConfigDict
from palivla.base_config import get_config as get_base_config

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)

    config["data_dir"] = "gs://cat-datasets/bridge_release/data/tfds"

    config["action_tokenizer"] = f"action_tokenizer.dct(action_dim=2, time_horizon=8, save_path='tmp', fit=True, pretrained_path=None, default_path='gs://cat-logs/action-tokenizer-dct')"
    config["sequence_builder"] = "sequence_builder.default(prompt_pad_length=80, gen_pad_length=20)"

    config["dataset_kwargs"]["oxe_kwargs"]["data_dir"] = config["data_dir"]
    config["visualization_datasets"]["bridge"]["data_dir"] = config["data_dir"]
    config["dataset_kwargs"]["oxe_kwargs"]["dataset_statistics"] = "gs://cat-datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/bridge_statistics.json"

    return ConfigDict(config)
