from collections.abc import Iterable, Iterator

from .block import KVCacheBlock


class FreeKVCacheBlockQueue:
    """
    Doubly linked free list with sentinel head/tail nodes
    Order is least recently freed to most recently freed.
    """

    def __init__(self, blocks: Iterable[KVCacheBlock]):
        self._head = KVCacheBlock(block_id=-1)
        self._tail = KVCacheBlock(block_id=-1)
        self._head.next_free_block = self._tail
        self._tail.prev_free_block = self._head
        self._size = 0
        for b in blocks:
            self.append(b)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[KVCacheBlock]:
        cur = self._head.next_free_block
        while cur is not self._tail:
            yield cur
            cur = cur.next_free_block

    def popleft(self) -> KVCacheBlock:
        if self._size == 0:
            raise IndexError("FreeKVCacheBlockQueue is empty")
        block = self._head.next_free_block
        self.remove(block)
        return block

    def append(self, block: KVCacheBlock) -> None:
        assert block.prev_free_block is None and block.next_free_block is None, (
            f"block {block.block_id} is already in a free queue"
        )
        prev = self._tail.prev_free_block
        prev.next_free_block = block
        block.prev_free_block = prev
        block.next_free_block = self._tail
        self._tail.prev_free_block = block
        self._size += 1

    def remove(self, block: KVCacheBlock) -> None:
        assert block.prev_free_block is not None and block.next_free_block is not None, (
            f"block {block.block_id} is not in a free queue"
        )
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = None
        block.next_free_block = None
        self._size -= 1
