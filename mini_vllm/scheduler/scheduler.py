from collections import deque
from dataclasses import dataclass, field

from ..block_manager.manager import BlockManager
from ..engine.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    """
    The output of the scheduler containing
    - scheduled sequences
    - number of scheduled tokens
    - number of prefill tokens
    - the number of decode tokens
    - the number of blocks to swap in
    - the number of blocks to swap out
    - the finished sequences ids
    """

    scheduled_seqs: list[Sequence] = field(default_factory=list)
    num_scheduled_tokens: list[int] = field(default_factory=list)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    blocks_to_swap_in: list[int] = field(default_factory=list)
    blocks_to_swap_out: list[int] = field(default_factory=list)
    finished_seq_ids: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_seqs) == 0

    @property
    def total_num_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens


class Scheduler:
    """
    The scheduler is responsible for scheduling the sequences for execution.
    It takes the current sequences and the block manager as input and outputs the scheduled sequences and the blocks to swap in and out.
    The scheduler can be implemented in different ways, such as round-robin, priority-based, etc.
    """

    def __init__(self, block_manager: BlockManager, max_num_seqs: int, max_num_batched_tokens: int):
        self.block_manager = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add_request(self, seq: Sequence) -> None:
        assert seq.status == SequenceStatus.WAITING, (
            f"add request expects WAITING, but got {seq.status}"
        )
        self.waiting.append(seq)

    def has_unfinished_requests(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        token_budget = self.max_num_batched_tokens
        seq_budget = self.max_num_seqs

        # decode first
        for seq in list(self.running):
            if seq_budget <= 0 or token_budget <= 0:
                break
            if seq.is_finished():
                continue
            if not seq.is_prefill_complete():
                continue
            self.block_manager.append_slots(seq, n_tokens=1)
            out.scheduled_seqs.append(seq)
            out.num_scheduled_tokens.append(1)
            out.num_decode_tokens += 1
            token_budget -= 1
            seq_budget -= 1

        # continue partial prefills already in running
        for seq in list(self.running):
            if seq_budget <= 0 or token_budget <= 0:
                break
            if seq.is_finished():
                continue
            if seq.is_prefill_complete():
                continue
            remaining = seq.num_uncomputed_prompt_tokens()
            chunk_size = min(remaining, token_budget)
            if not self.block_manager.can_allocate_for_chunk(seq, chunk_size):
                continue
            self.block_manager.allocate_for_chunk(seq, chunk_size)
            out.scheduled_seqs.append(seq)
            out.num_scheduled_tokens.append(chunk_size)
            out.num_prefill_tokens += chunk_size
            seq.num_computed_tokens += chunk_size
            token_budget -= chunk_size
            seq_budget -= 1

        # prefill next
        while self.waiting and seq_budget > 0 and token_budget > 0:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate_for_sequence(seq)
            remaining = seq.num_uncomputed_prompt_tokens()
            chunk_size = min(remaining, token_budget)
            seq.status = SequenceStatus.RUNNING
            seq.num_computed_tokens += chunk_size
            self.waiting.popleft()
            self.running.append(seq)
            out.scheduled_seqs.append(seq)
            out.num_scheduled_tokens.append(chunk_size)
            out.num_prefill_tokens += chunk_size
            token_budget -= chunk_size
            seq_budget -= 1

        return out

    def free_finished(self) -> list[int]:
        still_running = []
        finished_ids: list[int] = []
        for seq in self.running:
            if seq.is_finished():
                self.block_manager.free_sequence(seq)
                finished_ids.append(seq.seq_id)
            else:
                still_running.append(seq)
        self.running = still_running
        return finished_ids
