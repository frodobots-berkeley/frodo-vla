from os import PathLike
from typing import Any

import cloudpickle
import numpy as np
import tensorflow as tf
from einops import rearrange, EinopsError
from transformers import AutoProcessor

from big_vision.utils import Registry


class ActionTokenizer:
    def tokenize(self, data, obs=None): ...

    def detokenize(self, tokens, obs=None): ...

    def save(self, path: Any):
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "action_tokenizer.pkl"), "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, path: PathLike):
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "action_tokenizer.pkl"), "rb") as f:
            return cloudpickle.load(f)


@Registry.register("action_tokenizer.bin")
class BinActionTokenizer(ActionTokenizer):
    def __init__(
        self,
        min_action_value: np.ndarray | float,
        max_action_value: np.ndarray | float,
        action_vocab_size: int = 1000,
        action_horizon: int = 10,
    ):
        self.min_action_value = min_action_value
        self.max_action_value = max_action_value
        self.action_vocab_size = action_vocab_size
        self.action_horizon = action_horizon

    @property
    def num_tokens(self):
        return self.action_horizon * self.action_dim

    @property
    def vocab_size(self):
        return self.action_vocab_size

    def tokenize(self, data, obs=None):
        data = (data - self.min_action_value) / (
            self.max_action_value - self.min_action_value
        )
        data = rearrange(data, "... p a -> ... (p a)")
        return np.clip(
            np.round(data * (self.vocab_size - 1)).astype(np.int32),
            0,
            self.vocab_size - 1,
        )

    def detokenize(self, tokens, *, obs=None, action_dim: int):
        values = np.where(
            (tokens < 0) | (tokens >= self.vocab_size),
            np.nan,
            tokens / (self.vocab_size - 1),
        )
        data = (
            values * (self.max_action_value - self.min_action_value)
            + self.min_action_value
        )
        pred_action_dim = min(data.shape[0]//action_dim, self.action_horizon)
        data = data[:pred_action_dim*action_dim].reshape(-1, action_dim)
        return data


@Registry.register("action_tokenizer.dct")
class DCTActionTokenizer(ActionTokenizer):
    def __init__(
        self,
        max_action_value: np.ndarray | float,
        min_action_value: np.ndarray | float,
        action_dim: int,
        action_horizon: int = 10,
        action_vocab_size: int = 256,
        language_vocab_size: int = 257152,
        fast_tokenizer_path: str = "physical-intelligence/fast",
    ):
        self.max_action_value = 1
        self.min_action_value = -1
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_vocab_size = action_vocab_size
        self._fast_skip_tokens = 128
        self.language_vocab_size = language_vocab_size
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)

    @property
    def num_tokens(self):
        return self.action_horizon * self.action_dim

    @property
    def vocab_size(self):
        return self.action_vocab_size

    def tokenize(self, data, obs=None):
        data = (data - self.min_action_value) / (
            self.max_action_value - self.min_action_value
        )
        breakpoint()
        data = rearrange(data, "... p a -> ... (p a)")
        breakpoint()
        action_tokens = self._fast_tokenizer(data)[0]
        action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
        action_tokens = np.array(action_tokens).reshape(data.shape[0], -1).tolist()
        return action_tokens_in_pg

    def detokenize(self, tokens, *, obs=None):
        action_tokens = self._paligemma_tokens_to_act_tokens(tokens)
        return self._fast_tokenizer.decode([action_tokens.tolist()], time_horizon=self.action_horizon, action_dim=self.action_dim)[0]
        # values = np.where(
        #     (tokens < 0) | (tokens >= self.vocab_size),
        #     np.nan,
        #     tokens / (self.vocab_size - 1),
        # )
        # data = (
        #     values * (self.max_action_value - self.min_action_value)
        #     + self.min_action_value
        # )
        # pred_action_dim = min(data.shape[0]//action_dim, self.action_horizon)
        # data = data[:pred_action_dim*action_dim].reshape(-1, action_dim)
        # return data

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self.language_vocab_size - 1 - self._fast_skip_tokens - tokens 
