from dataclasses import dataclass
from typing import TypedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from mini_vllm.config import Config


@dataclass(frozen=True)
class ModelInfo:
    """
    Architecture specific information about the model the rest of the engine reads
    """

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    max_position: int

    def kv_bytes_per_block(self, block_size: int, dtype: torch.dtype) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * block_size * dtype.itemsize


class ModelLoaderResult(TypedDict):
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    info: ModelInfo


class ModelLoader:
    """
    Loads an HF causal langauge model and extracts model info for the engine
    """

    def __init__(self, config: Config):
        self.config = config

    def load(self) -> ModelLoaderResult:
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=self.config.dtype,
            trust_remote_code=self.config.trust_remote_code,
        ).to(self.config.device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        info = self._extract_info(model.config)

        return ModelLoaderResult(model=model, tokenizer=tokenizer, info=info)

    @staticmethod
    def _extract_info(hf_cfg) -> ModelInfo:
        num_heads = hf_cfg.num_attention_heads
        num_kv_heads = getattr(hf_cfg, "num_key_value_heads", num_heads)
        head_dim = getattr(hf_cfg, "head_dim", hf_cfg.hidden_size // num_heads)
        return ModelInfo(
            num_layers=hf_cfg.num_hidden_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hf_cfg.hidden_size,
            vocab_size=hf_cfg.vocab_size,
            max_position=hf_cfg.max_position_embeddings,
        )
