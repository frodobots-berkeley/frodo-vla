from ml_collections import ConfigDict
from palivla.base_config import get_config as get_base_config

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)

    config["data_dir"] = "gs://cat-datasets/bridge_release/data/tfds"

    config["action_tokenizer"] = f"action_tokenizer.bin(min_action_value=-1, max_action_value=1, action_vocab_size=128, action_horizon=8)"
    config["sequence_builder"] = "sequence_builder.default(prompt_pad_length=80, gen_pad_length=20)",

    return ConfigDict(config)
