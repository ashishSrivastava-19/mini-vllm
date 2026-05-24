from ..engine.sequence import Sequence
from .block import KVCacheBlock
from .free_queue import FreeKVCacheBlockQueue


class BlockManager:
    """
    Manages and owns the global KVCache block pool
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks: list[KVCacheBlock] = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
        self.free_queue = FreeKVCacheBlockQueue(self.blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_queue)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def _allocate_one(self) -> KVCacheBlock:
        if len(self.free_queue) == 0:
            raise RuntimeError("BlockManager: out of free KVCache blocks")
        block = self.free_queue.popleft()
        assert block.ref_count == 0, (
            f"block {block.block_id} popped from free queue with ref count {block.ref_count}, invariant violated"
        )
        block.incr_ref()
        return block

    def allocate(self, num_blocks: int) -> list[KVCacheBlock]:
        if num_blocks > len(self.free_queue):
            raise RuntimeError(
                f"BlockManager: requested {num_blocks} blocks, only {len(self.free_queue)} free blocks available"
            )
        return [self._allocate_one() for _ in range(num_blocks)]

    def free(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            block.decr_ref()
            if block.ref_count == 0:
                self.free_queue.append(block)

    def can_allocate(self, seq: Sequence) -> bool:
        return seq.num_logical_blocks() <= len(self.free_queue)

    def allocate_for_sequence(self, seq: Sequence) -> None:
        assert seq.block_table == [], f"sequence {seq.seq_id} already has blocks; double admit?"
        n = seq.num_logical_blocks()
        blocks = self.allocate(n)
        seq.block_table = [b.block_id for b in blocks]

    def append_slots(self, seq: Sequence, n_tokens: int) -> KVCacheBlock | None:
        new_total_tokens = seq.num_tokens + n_tokens
        new_logical_blocks = (new_total_tokens + self.block_size - 1) // self.block_size
        current_blocks = len(seq.block_table)

        if new_logical_blocks <= current_blocks:
            return None

        assert new_logical_blocks == current_blocks + 1, (
            f"v0 append slots expected to grow by 1 block, got {new_logical_blocks - current_blocks}"
        )
        new_block = self._allocate_one()
        seq.block_table.append(new_block.block_id)
        return new_block

    def free_sequence(self, seq: Sequence) -> None:
        if not seq.block_table:
            return
        blocks = [self.blocks[i] for i in seq.block_table]
        self.free(blocks)
        seq.block_table = []

    def can_allocate_for_chunk(self, seq: Sequence, chunk_size: int) -> bool:
        new_total = seq.num_tokens + chunk_size
        new_blocks_needed = (new_total + self.block_size - 1) // self.block_size
        delta = new_blocks_needed - len(seq.block_table)
        return delta <= self.num_free_blocks

    def allocate_for_chunk(self, seq: Sequence, chunk_size: int) -> None:
        new_total = seq.num_tokens + chunk_size
        new_blocks_needed = (new_total + self.block_size - 1) // self.block_size
        for _ in range(new_blocks_needed - len(seq.block_table)):
            new_block = self._allocate_one()
            seq.block_table.append(new_block.block_id)
