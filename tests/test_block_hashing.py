import pytest

from mini_vllm.block_manager.hashing import (
    ROOT_HASH,
    hash_block,
    hash_blocks_for_tokens,
)
from mini_vllm.block_manager.manager import BlockManager
from mini_vllm.engine.sequence import Sequence
from mini_vllm.sampling.params import SamplingParams


def _seq(prompt_token_ids: list[int], block_size: int = 4) -> Sequence:
    return Sequence(
        seq_id=0,
        prompt_token_ids=list(prompt_token_ids),
        sampling_params=SamplingParams(temperature=0.0),
        block_size=block_size,
    )


# ---------- hash_block primitive ----------


def test_hash_is_deterministic_across_calls():
    assert hash_block(ROOT_HASH, (1, 2, 3, 4)) == hash_block(ROOT_HASH, (1, 2, 3, 4))


def test_hash_depends_on_token_order():
    assert hash_block(ROOT_HASH, (1, 2, 3, 4)) != hash_block(ROOT_HASH, (4, 3, 2, 1))


def test_hash_depends_on_parent():
    assert hash_block(ROOT_HASH, (10, 20, 30)) != hash_block(42, (10, 20, 30))


def test_different_blocks_produce_different_hashes():
    assert hash_block(ROOT_HASH, (1, 2, 3, 4)) != hash_block(ROOT_HASH, (1, 2, 3, 5))


def test_list_input_rejected():
    with pytest.raises(AssertionError):
        hash_block(ROOT_HASH, [1, 2, 3])


# ---------- hash_blocks_for_tokens helper ----------


def test_hashes_for_aligned_token_sequence():
    assert len(hash_blocks_for_tokens(list(range(50)), block_size=16)) == 3


def test_no_hashes_for_partial_block():
    assert hash_blocks_for_tokens(list(range(10)), block_size=16) == []


def test_chain_property_holds():
    ha = hash_blocks_for_tokens(list(range(32)), block_size=16)
    hb = hash_blocks_for_tokens([999, *range(1, 32)], block_size=16)
    assert ha[0] != hb[0]
    assert ha[1] != hb[1]  # divergence in block 0 propagates downstream


def test_shared_prefix_produces_shared_block_hashes():
    ha = hash_blocks_for_tokens([*range(32), 100, 101], block_size=16)
    hb = hash_blocks_for_tokens([*range(32), 200, 201], block_size=16)
    assert ha[:2] == hb[:2]  # identical first two blocks -> identical hashes


# ---------- BlockManager integration ----------


def test_maybe_register_records_hash():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq([1, 2, 3, 4, 5, 6, 7, 8])  # exactly 2 full blocks
    bm.allocate_for_sequence(seq)
    assert seq.block_table != []

    bm.maybe_register_full_block(seq, 0)
    block_0 = bm.blocks[seq.block_table[0]]
    assert block_0.block_hash is not None
    assert block_0.block_hash in bm.hash_to_block

    bm.maybe_register_full_block(seq, 1)
    block_1 = bm.blocks[seq.block_table[1]]
    assert block_1.block_hash is not None
    assert block_1.block_hash != block_0.block_hash


def test_maybe_register_is_idempotent():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq([1, 2, 3, 4])
    bm.allocate_for_sequence(seq)
    bm.maybe_register_full_block(seq, 0)
    h_before = bm.blocks[seq.block_table[0]].block_hash
    bm.maybe_register_full_block(seq, 0)  # second call should no-op
    assert bm.blocks[seq.block_table[0]].block_hash == h_before


def test_partial_block_not_registered():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq([1, 2, 3])  # 3 tokens, block isn't full
    bm.allocate_for_sequence(seq)
    bm.maybe_register_full_block(seq, 0)
    assert bm.blocks[seq.block_table[0]].block_hash is None


def test_block_chain_matches_helper():
    """The hashes stored on blocks by maybe_register_full_block must equal the
    hashes hash_blocks_for_tokens computes for the same tokens — i.e. the engine
    path and the lookup path agree (load-bearing for Thursday)."""
    tokens = [10, 11, 12, 13, 20, 21, 22, 23]  # 2 full blocks of size 4
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq(tokens)
    bm.allocate_for_sequence(seq)
    bm.maybe_register_full_block(seq, 0)
    bm.maybe_register_full_block(seq, 1)

    expected = hash_blocks_for_tokens(tokens, block_size=4)
    got = [bm.blocks[seq.block_table[i]].block_hash for i in range(2)]
    assert got == expected
