import random

import pytest

from mini_vllm.block_manager.block import KVCacheBlock
from mini_vllm.block_manager.free_queue import FreeKVCacheBlockQueue
from mini_vllm.block_manager.manager import BlockManager
from mini_vllm.engine.sequence import Sequence


def make_pool(n: int) -> list[KVCacheBlock]:
    return [KVCacheBlock(block_id=i) for i in range(n)]


# ---------- KVCacheBlock ----------


def test_block_ref_count_basic():
    b = KVCacheBlock(block_id=0)
    assert b.ref_count == 0
    b.incr_ref()
    b.incr_ref()
    assert b.ref_count == 2
    b.decr_ref()
    assert b.ref_count == 1


def test_block_decr_below_zero_asserts():
    b = KVCacheBlock(block_id=0)
    with pytest.raises(AssertionError):
        b.decr_ref()


def test_block_reset_hash():
    b = KVCacheBlock(block_id=0, block_hash=12345)
    b.reset_hash()
    assert b.block_hash is None


def test_block_repr_does_not_recurse():
    # Regression test: with default dataclass repr the linked-list pointers
    # would cause infinite recursion. Build a 2-node cycle and repr it.
    a = KVCacheBlock(block_id=1)
    b = KVCacheBlock(block_id=2)
    a.next_free_block = b
    b.prev_free_block = a
    a.prev_free_block = b
    b.next_free_block = a
    s = repr(a)
    assert "id=1" in s
    assert "ref=0" in s


def test_block_equality_is_identity():
    # eq=False -> identity-based equality, not structural.
    # Two distinct instances with the same block_id must NOT compare equal,
    # otherwise the BlockManager's "is this the same physical block?" checks break.
    a = KVCacheBlock(block_id=5)
    b = KVCacheBlock(block_id=5)
    assert a != b


# ---------- FreeKVCacheBlockQueue ----------


def test_init_size():
    q = FreeKVCacheBlockQueue(make_pool(10))
    assert len(q) == 10


def test_init_empty():
    q = FreeKVCacheBlockQueue([])
    assert len(q) == 0


def test_popleft_order_is_fifo():
    blocks = make_pool(5)
    q = FreeKVCacheBlockQueue(blocks)
    popped = [q.popleft().block_id for _ in range(5)]
    assert popped == [0, 1, 2, 3, 4]
    assert len(q) == 0


def test_pop_from_empty_raises():
    q = FreeKVCacheBlockQueue([])
    with pytest.raises(IndexError):
        q.popleft()


def test_append_after_drain_works():
    blocks = make_pool(3)
    q = FreeKVCacheBlockQueue(blocks)
    for _ in range(3):
        q.popleft()
    q.append(blocks[1])
    q.append(blocks[0])
    assert q.popleft().block_id == 1
    assert q.popleft().block_id == 0


def test_remove_from_middle():
    blocks = make_pool(5)
    q = FreeKVCacheBlockQueue(blocks)
    q.remove(blocks[2])
    assert len(q) == 4
    order = [q.popleft().block_id for _ in range(4)]
    assert order == [0, 1, 3, 4]


def test_remove_head():
    blocks = make_pool(3)
    q = FreeKVCacheBlockQueue(blocks)
    q.remove(blocks[0])
    assert len(q) == 2
    assert q.popleft().block_id == 1


def test_remove_tail():
    blocks = make_pool(3)
    q = FreeKVCacheBlockQueue(blocks)
    q.remove(blocks[2])
    assert len(q) == 2
    order = [q.popleft().block_id for _ in range(2)]
    assert order == [0, 1]


def test_remove_then_reappend_goes_to_mru_end():
    # Simulates: cached block is reused (ref 0->1, removed from free queue),
    # then released (ref->0, returned to MRU end of queue).
    blocks = make_pool(4)
    q = FreeKVCacheBlockQueue(blocks)
    q.remove(blocks[1])
    q.append(blocks[1])
    order = [q.popleft().block_id for _ in range(4)]
    assert order == [0, 2, 3, 1]


def test_append_already_in_queue_asserts():
    blocks = make_pool(3)
    q = FreeKVCacheBlockQueue(blocks)
    with pytest.raises(AssertionError):
        q.append(blocks[0])


def test_remove_not_in_queue_asserts():
    blocks = make_pool(3)
    q = FreeKVCacheBlockQueue(blocks)
    b = q.popleft()
    with pytest.raises(AssertionError):
        q.remove(b)


def test_iter_walks_in_order():
    blocks = make_pool(4)
    q = FreeKVCacheBlockQueue(blocks)
    q.remove(blocks[1])
    assert [b.block_id for b in q] == [0, 2, 3]


def test_iter_empty():
    q = FreeKVCacheBlockQueue([])
    assert list(q) == []


def test_links_consistent_after_ops():
    # Walk forward and backward and check both directions match.
    blocks = make_pool(6)
    q = FreeKVCacheBlockQueue(blocks)
    q.remove(blocks[2])
    q.remove(blocks[4])
    q.append(blocks[2])

    forward = [b.block_id for b in q]
    backward = []
    cur = q._tail.prev_free_block
    while cur is not q._head:
        backward.append(cur.block_id)
        cur = cur.prev_free_block
    assert forward == list(reversed(backward))


@pytest.mark.parametrize("seed", [0, 1, 2, 17, 42])
def test_random_stress(seed):
    blocks = make_pool(100)
    q = FreeKVCacheBlockQueue(blocks)
    rng = random.Random(seed)
    out_of_queue: list[KVCacheBlock] = []

    for _ in range(10_000):
        op = rng.choice(["pop", "append", "remove"])
        if op == "pop" and len(q) > 0:
            out_of_queue.append(q.popleft())
        elif op == "append" and out_of_queue:
            b = out_of_queue.pop(rng.randrange(len(out_of_queue)))
            q.append(b)
        elif op == "remove" and len(q) > 0:
            n = rng.randrange(len(q))
            cur = q._head.next_free_block
            for _ in range(n):
                cur = cur.next_free_block
            q.remove(cur)
            out_of_queue.append(cur)

    # Invariant: every block accounted for exactly once.
    in_queue_ids = [b.block_id for b in q]
    assert len(in_queue_ids) == len(q)
    in_set = set(in_queue_ids)
    out_set = {b.block_id for b in out_of_queue}
    assert len(in_set) == len(in_queue_ids)  # no in-queue duplicates
    assert len(out_set) == len(out_of_queue)  # no out-of-queue duplicates
    assert in_set.isdisjoint(out_set)
    assert in_set | out_set == set(range(100))


# ---------- BlockManager ----------


def make_seq(seq_id: int, prompt_len: int, block_size: int = 16) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=None,
        block_size=block_size,
    )


def test_initial_pool_state():
    bm = BlockManager(num_blocks=10, block_size=16)
    assert bm.num_free_blocks == 10
    assert bm.num_used_blocks == 0
    assert bm.num_blocks == 10
    assert bm.block_size == 16


def test_allocate_and_free_roundtrip():
    bm = BlockManager(num_blocks=10, block_size=16)
    blocks = bm.allocate(4)
    assert bm.num_free_blocks == 6
    assert bm.num_used_blocks == 4
    assert all(b.ref_count == 1 for b in blocks)
    bm.free(blocks)
    assert bm.num_free_blocks == 10
    assert all(b.ref_count == 0 for b in blocks)


def test_allocate_zero_is_noop():
    bm = BlockManager(num_blocks=4, block_size=16)
    blocks = bm.allocate(0)
    assert blocks == []
    assert bm.num_free_blocks == 4


def test_allocate_too_many_raises():
    bm = BlockManager(num_blocks=4, block_size=16)
    with pytest.raises(RuntimeError):
        bm.allocate(5)
    assert bm.num_free_blocks == 4


def test_allocate_one_until_empty_then_raises():
    bm = BlockManager(num_blocks=3, block_size=16)
    bm.allocate(3)
    assert bm.num_free_blocks == 0
    with pytest.raises(RuntimeError):
        bm._allocate_one()


def test_allocate_for_sequence_writes_block_table():
    bm = BlockManager(num_blocks=10, block_size=16)
    seq = make_seq(seq_id=1, prompt_len=50)
    assert bm.can_allocate(seq)
    bm.allocate_for_sequence(seq)
    assert len(seq.block_table) == 4
    assert bm.num_free_blocks == 6
    assert all(isinstance(bid, int) for bid in seq.block_table)


def test_allocate_for_sequence_double_admit_asserts():
    bm = BlockManager(num_blocks=10, block_size=16)
    seq = make_seq(seq_id=1, prompt_len=50)
    bm.allocate_for_sequence(seq)
    with pytest.raises(AssertionError):
        bm.allocate_for_sequence(seq)


def test_can_allocate_returns_false_when_short():
    bm = BlockManager(num_blocks=2, block_size=16)
    seq = make_seq(seq_id=1, prompt_len=50)
    assert not bm.can_allocate(seq)


def test_can_allocate_exact_fit():
    bm = BlockManager(num_blocks=4, block_size=16)
    seq = make_seq(seq_id=1, prompt_len=64)
    assert bm.can_allocate(seq)
    bm.allocate_for_sequence(seq)
    assert bm.num_free_blocks == 0


def test_append_slots_no_new_block_when_room_left():
    bm = BlockManager(num_blocks=10, block_size=16)
    seq = make_seq(seq_id=1, prompt_len=10)
    bm.allocate_for_sequence(seq)
    seq.append_token(999)
    new = bm.append_slots(seq, n_tokens=1)
    assert new is None
    assert len(seq.block_table) == 1
    assert bm.num_free_blocks == 9


def test_append_slots_grows_block_table():
    bm = BlockManager(num_blocks=10, block_size=16)
    seq = make_seq(seq_id=1, prompt_len=16)
    bm.allocate_for_sequence(seq)
    seq.append_token(999)
    new = bm.append_slots(seq, n_tokens=1)
    assert new is not None
    assert new.ref_count == 1
    assert len(seq.block_table) == 2
    assert seq.block_table[-1] == new.block_id
    assert bm.num_free_blocks == 8


def test_append_slots_oom_raises():
    bm = BlockManager(num_blocks=1, block_size=16)
    seq = make_seq(seq_id=1, prompt_len=16)
    bm.allocate_for_sequence(seq)
    seq.append_token(999)
    with pytest.raises(RuntimeError):
        bm.append_slots(seq, n_tokens=1)


def test_free_sequence_returns_all_blocks():
    bm = BlockManager(num_blocks=10, block_size=16)
    seq = make_seq(seq_id=1, prompt_len=50)
    bm.allocate_for_sequence(seq)
    bm.free_sequence(seq)
    assert bm.num_free_blocks == 10
    assert seq.block_table == []


def test_free_sequence_on_empty_is_noop():
    bm = BlockManager(num_blocks=10, block_size=16)
    seq = make_seq(seq_id=1, prompt_len=50)
    bm.free_sequence(seq)
    assert bm.num_free_blocks == 10
    assert seq.block_table == []


def test_no_leaks_under_churn():
    bm = BlockManager(num_blocks=20, block_size=16)
    seqs = [make_seq(seq_id=i, prompt_len=16) for i in range(20)]
    for s in seqs:
        bm.allocate_for_sequence(s)
    assert bm.num_free_blocks == 0

    for s in seqs[:10]:
        bm.free_sequence(s)
    assert bm.num_free_blocks == 10

    new_seqs = [make_seq(seq_id=100 + i, prompt_len=16) for i in range(10)]
    for s in new_seqs:
        bm.allocate_for_sequence(s)
    assert bm.num_free_blocks == 0

    for s in seqs[10:] + new_seqs:
        bm.free_sequence(s)
    assert bm.num_free_blocks == 20

    all_ids = []
    cur = bm.free_queue._head.next_free_block
    while cur is not bm.free_queue._tail:
        all_ids.append(cur.block_id)
        cur = cur.next_free_block
    assert sorted(all_ids) == list(range(20))
    assert len(all_ids) == len(set(all_ids))
    assert all(bm.blocks[i].ref_count == 0 for i in range(20))
