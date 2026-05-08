from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(eq=False, repr=False)
class KVCacheBlock:
    """A single physical KV-cache block.

    Owned by the BlockManager. Sequences only reference these by id.
    Carries free-list pointers so the FreeKVCacheBlockQueue can keep
    pop / append / remove all O(1).
    """

    block_id: int
    ref_count: int = 0
    block_hash: int | None = None
    prev_free_block: KVCacheBlock | None = field(default=None, repr=False, compare=False)
    next_free_block: KVCacheBlock | None = field(default=None, repr=False, compare=False)

    def incr_ref(self) -> None:
        self.ref_count += 1

    def decr_ref(self) -> None:
        assert self.ref_count > 0, (
            f"decr_ref on block {self.block_id} with ref_count {self.ref_count}"
        )
        self.ref_count -= 1

    def reset_hash(self) -> None:
        self.block_hash = None

    def __repr__(self) -> str:
        # Custom repr - never recurse into prev/next pointers.
        return f"KVCacheBlock(id={self.block_id}, ref={self.ref_count}, hash={self.block_hash})"
