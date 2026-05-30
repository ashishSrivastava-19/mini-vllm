import pytest

from mini_vllm.block_manager.manager import BlockManager
from mini_vllm.engine.sequence import Sequence
from mini_vllm.sampling.params import SamplingParams


def _seq(seq_id: int, prompt: list[int], block_size: int = 4) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(temperature=0.0),
        block_size=block_size,
    )


def _lifecycle(bm: BlockManager, seq: Sequence) -> None:
    """Admit seq, hash its full prompt blocks, free it. Leaves cached blocks
    in the queue with hashes still registered."""
    bm.allocate_for_sequence(seq)
    for i in range(len(seq.prompt_token_ids) // bm.block_size):
        bm.maybe_register_full_block(seq, i)
    bm.free_sequence(seq)


def test_first_sequence_has_zero_cache_hits():
    bm = BlockManager(num_blocks=20, block_size=4)
    hits = bm.allocate_for_sequence(_seq(1, list(range(16))))
    assert hits == 0
    assert bm.cache_hit_tokens == 0


def test_allocate_for_sequence_returns_full_block_table():
    """All logical blocks must be present in seq.block_table (cached + fresh).
    The runner indexes seq.block_table[pos // block_size]; gaps cause IndexError."""
    bm = BlockManager(num_blocks=20, block_size=4)
    seq = _seq(1, list(range(10)))
    bm.allocate_for_sequence(seq)
    assert len(seq.block_table) == 3


def test_shared_prefix_hits_cache():
    bm = BlockManager(num_blocks=20, block_size=4)
    _lifecycle(bm, _seq(1, list(range(16))))

    seq_b = _seq(2, [*range(12), 99, 99, 99, 99])
    hits = bm.allocate_for_sequence(seq_b)
    assert hits == 12
    assert seq_b.num_computed_tokens == 12
    assert len(seq_b.block_table) == 4


def test_full_cache_hit_leaves_last_block_for_recompute():
    bm = BlockManager(num_blocks=20, block_size=4)
    _lifecycle(bm, _seq(1, list(range(16))))

    seq_b = _seq(2, list(range(16)))
    hits = bm.allocate_for_sequence(seq_b)
    assert hits == 12
    assert seq_b.num_uncomputed_prompt_tokens() == 4


def test_chain_breaks_on_first_miss():
    bm = BlockManager(num_blocks=20, block_size=4)
    _lifecycle(bm, _seq(1, list(range(12))))

    seq_b = _seq(2, [99, 99, 99, 99, *range(4, 12)])
    hits = bm.allocate_for_sequence(seq_b)
    assert hits == 0


def test_ref_count_shared_across_sequences():
    bm = BlockManager(num_blocks=20, block_size=4)
    _lifecycle(bm, _seq(1, list(range(8))))

    seq_b = _seq(2, [*range(8), 99, 99, 99, 99])
    seq_c = _seq(3, [*range(8), 88, 88, 88, 88])
    bm.allocate_for_sequence(seq_b)
    bm.allocate_for_sequence(seq_c)

    cached_id = seq_b.block_table[0]
    assert bm.blocks[cached_id].ref_count == 2

    bm.free_sequence(seq_b)
    assert bm.blocks[cached_id].ref_count == 1
    bm.free_sequence(seq_c)
    assert bm.blocks[cached_id].ref_count == 0


def test_metrics_track_cumulative_hit_rate():
    bm = BlockManager(num_blocks=20, block_size=4)
    _lifecycle(bm, _seq(1, list(range(8))))
    bm.allocate_for_sequence(_seq(2, [*range(8), 2, 2, 2, 2]))
    bm.allocate_for_sequence(_seq(3, [*range(8), 3, 3, 3, 3]))
    assert bm.total_prompt_tokens == 32
    assert bm.cache_hit_tokens == 16
    assert bm.cache_hit_rate == pytest.approx(0.5)


def test_stale_hash_cleared_on_fresh_pop():
    """When a cached block gets popped for a fresh allocation, its hash entry
    must be removed so future lookups don't 'hit' overwritten data."""
    bm = BlockManager(num_blocks=2, block_size=4)
    _lifecycle(bm, _seq(1, list(range(8))))
    assert len(bm.hash_to_block) == 2

    bm.allocate_for_sequence(_seq(2, [99, 99, 99, 99, 88, 88, 88, 88]))
    assert len(bm.hash_to_block) == 0


def test_can_allocate_accounts_for_cached_blocks():
    """A cached block sitting in the free queue doesn't consume a fresh slot."""
    bm = BlockManager(num_blocks=2, block_size=4)
    _lifecycle(bm, _seq(1, list(range(8))))
    assert bm.can_allocate(_seq(2, list(range(8)))) is True


def test_can_allocate_false_when_truly_oversubscribed():
    bm = BlockManager(num_blocks=2, block_size=4)
    _lifecycle(bm, _seq(1, list(range(8))))
    big = _seq(2, [50] * 16)
    assert bm.can_allocate(big) is False
