from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mini_vllm.sampling.params import SamplingParams


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    PREEMPTED = "preempted"


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    block_size: int = 16

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def num_logical_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def get_last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def is_prefill_complete(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    def num_uncomputed_prompt_tokens(self) -> int:
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)
