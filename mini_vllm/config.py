from dataclasses import dataclass

import torch


@dataclass
class Config:
    model_path: str
    dtype: torch.dtype = torch.float16
    trust_remote_code: bool = False

    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096

    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    num_gpu_blocks: int | None = None

    eos_token_id: int | None = None
    seed: int = 0

    device: str = "cuda"

    def validate(self) -> "Config":
        if self.block_size <= 0 or (self.block_size & (self.block_size - 1)):
            raise ValueError(f"block_size must be a positive power of 2, got {self.block_size}")
        if self.max_num_batched_tokens < self.block_size:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens})"
                f"must be >= block size ({self.block_size})"
            )
        if self.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {self.max_model_len}")
        if not (0 < self.gpu_memory_utilization < 1):
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1), got {self.gpu_memory_utilization}"
            )
        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {self.max_num_seqs}")
        return self
