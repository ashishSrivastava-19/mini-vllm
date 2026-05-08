from mini_vllm.engine.sequence import Sequence, SequenceStatus


def make_seq(prompt_len: int = 50, block_size: int = 16) -> Sequence:
    return Sequence(
        seq_id=0,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=None,  # stub until Week 2
        block_size=block_size,
    )


def test_initial_status_is_waiting():
    s = make_seq()
    assert s.status == SequenceStatus.WAITING
    assert not s.is_finished()


def test_num_logical_blocks_prompt_only():
    s = make_seq(prompt_len=50, block_size=16)
    # ceil(50 / 16) == 4
    assert s.num_logical_blocks() == 4


def test_num_logical_blocks_after_decode():
    s = make_seq(prompt_len=50, block_size=16)
    for tok in range(20):
        s.append_token(tok)
    # ceil(70 / 16) == 5
    assert s.num_logical_blocks() == 5
    assert s.num_tokens == 70
    assert s.num_output_tokens == 20


def test_num_logical_blocks_exact_boundary():
    s = make_seq(prompt_len=32, block_size=16)
    assert s.num_logical_blocks() == 2  # exactly 2 blocks, no half block


def test_num_logical_blocks_single_token():
    s = make_seq(prompt_len=1, block_size=16)
    assert s.num_logical_blocks() == 1


def test_get_last_token_id_prompt_then_decode():
    s = make_seq(prompt_len=5)
    assert s.get_last_token_id() == 4  # last prompt token
    s.append_token(99)
    assert s.get_last_token_id() == 99


def test_num_uncomputed_tokens_tracks_chunked_prefill():
    s = make_seq(prompt_len=100, block_size=16)
    assert s.num_uncomputed_tokens() == 100
    s.num_computed_tokens = 64
    assert s.num_uncomputed_tokens() == 36
    s.num_computed_tokens = 100
    assert s.num_uncomputed_tokens() == 0


def test_all_token_ids_concatenates():
    s = make_seq(prompt_len=3)
    s.append_token(100)
    s.append_token(101)
    assert s.all_token_ids == [0, 1, 2, 100, 101]


def test_finished_flag():
    s = make_seq()
    assert not s.is_finished()
    s.status = SequenceStatus.FINISHED
    assert s.is_finished()


def test_block_table_is_ints_not_objects():
    # Guardrail-as-test: block_table must stay a list[int].
    # The BlockManager owns block objects; the Sequence only references ids.
    s = make_seq()
    assert s.block_table == []
    s.block_table.extend([3, 7, 12])
    assert all(isinstance(b, int) for b in s.block_table)
