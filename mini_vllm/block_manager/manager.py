from ..engine.sequence import Sequence
from .block import KVCacheBlock
from .free_queue import FreeKVCacheBlockQueue
from .hashing import ROOT_HASH, hash_block


class BlockManager:
    """
    Manages and owns the global KVCache block pool
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks: list[KVCacheBlock] = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
        self.free_queue = FreeKVCacheBlockQueue(self.blocks)
        self.hash_to_block: dict[int, KVCacheBlock] = {}
        self.cache_hit_tokens: int = 0
        self.total_prompt_tokens: int = 0

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
        if block.block_hash is not None:
            if self.hash_to_block.get(block.block_hash) is block:
                del self.hash_to_block[block.block_hash]
            block.reset_hash()
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
        n_fresh, cached_in_queue = self._lookup_prefix(seq.prompt_token_ids, dry_run = True)
        return n_fresh <= len(self.free_queue) - cached_in_queue

    def allocate_for_sequence(self, seq: Sequence) -> int:
        assert seq.block_table == [], f"sequence {seq.seq_id} already has blocks; double admit?"
        cached_blocks, _ = self._lookup_prefix(seq.prompt_token_ids, dry_run = False)
        cache_hit_blocks = len(cached_blocks)
        cache_hit_tokens = cache_hit_blocks * self.block_size

        n_logical = (len(seq.prompt_token_ids) + self.block_size - 1) // self.block_size
        num_fresh = n_logical - cache_hit_blocks

        for b in cached_blocks:
            if b.ref_count == 0:
                self.free_queue.remove(b)
            b.incr_ref()
            seq.block_table.append(b.block_id)

        for _ in range(num_fresh):
            fresh = self._allocate_one()
            seq.block_table.append(fresh.block_id)

        seq.num_computed_tokens = cache_hit_tokens
        self.cache_hit_tokens += cache_hit_tokens
        self.total_prompt_tokens += len(seq.prompt_token_ids)
        return cache_hit_tokens
    
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
        new_pos = seq.num_computed_tokens + chunk_size
        new_blocks_needed = (new_pos + self.block_size - 1) // self.block_size
        delta = max(0, new_blocks_needed - len(seq.block_table))
        return delta <= self.num_free_blocks

    def allocate_for_chunk(self, seq: Sequence, chunk_size: int) -> None:
        new_pos = seq.num_computed_tokens + chunk_size
        new_blocks_needed = (new_pos + self.block_size - 1) // self.block_size
        for _ in range(max(0, new_blocks_needed - len(seq.block_table))):
            new_block = self._allocate_one()
            seq.block_table.append(new_block.block_id)

    def maybe_register_full_block(self, seq: Sequence, logical_block_idx: int) -> None:
        """
        Compute and store the hash of a logical block if it's full, and register the hash to block mapping for potential future reuse.
        """
        if logical_block_idx >= len(seq.block_table):
            return
        block = self.blocks[seq.block_table[logical_block_idx]]

        if block.block_hash is not None:
            return

        start = logical_block_idx * self.block_size
        end = start + self.block_size
        if end > seq.num_tokens:
            return
        block_tokens = tuple(seq.all_token_ids[start:end])
        if logical_block_idx == 0:
            parent_hash = ROOT_HASH
        else:
            parent_block = self.blocks[seq.block_table[logical_block_idx - 1]]
            if parent_block.block_hash is None:
                self.maybe_register_full_block(seq, logical_block_idx - 1)
            parent_hash = parent_block.block_hash

        block.block_hash = hash_block(parent_hash, block_tokens)
        self.hash_to_block.setdefault(block.block_hash, block)

    def _lookup_prefix(self, prompt: list[int], dry_run: bool) -> tuple[list[KVCacheBlock] | int, int]:
        n_full = len(prompt) // self.block_size
        n_logical = (len(prompt) + self.block_size - 1) // self.block_size

        max_cacheable = n_full - 1 if n_full == n_logical and n_full > 0 else n_full

        cached_blocks: list[KVCacheBlock] = []
        cached_in_queue = 0
        parent_hash = ROOT_HASH
        for i in range(max_cacheable):
            start = i * self.block_size
            tokens = tuple(prompt[start : start + self.block_size])
            h = hash_block(parent_hash, tokens)
            cached = self.hash_to_block.get(h)
            if cached is None:
                break
            cached_blocks.append(cached)
            if cached.ref_count == 0:
                cached_in_queue += 1
            parent_hash = h
        
        if dry_run:
            return (n_logical - len(cached_blocks), cached_in_queue)
        return (cached_blocks, cached_in_queue)
    
    @property
    def cache_hit_rate(self) -> float:
        if self.total_prompt_tokens == 0:
            return 0.0
        return self.cache_hit_tokens / self.total_prompt_tokens