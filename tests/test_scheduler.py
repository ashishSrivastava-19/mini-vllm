from mini_vllm.block_manager.manager import BlockManager
from mini_vllm.engine.sequence import Sequence, SequenceStatus
from mini_vllm.scheduler.scheduler import Scheduler


def make_seq(seq_id: int, prompt_len: int, block_size: int = 16) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=None,
        block_size=block_size,
    )


def make_scheduler(num_blocks=20, max_seqs=8, max_tokens=256, block_size=16):
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    sched = Scheduler(
        block_manager=bm,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max_tokens,
    )
    return sched, bm


# ---------- waiting / running queue mechanics ----------


def test_empty_scheduler_returns_empty_output():
    sched, _ = make_scheduler()
    out = sched.schedule()
    assert out.is_empty
    assert out.num_prefill_tokens == 0
    assert out.num_decode_tokens == 0
    assert not sched.has_unfinished_requests()


def test_add_request_lands_in_waiting():
    sched, _ = make_scheduler()
    seq = make_seq(1, 32)
    sched.add_request(seq)
    assert len(sched.waiting) == 1
    assert sched.has_unfinished_requests()


# ---------- prefill admission ----------


def test_single_prefill_admits_to_running():
    sched, bm = make_scheduler()
    seq = make_seq(1, 32)  # 2 blocks
    sched.add_request(seq)
    out = sched.schedule()

    assert len(out.scheduled_seqs) == 1
    assert out.scheduled_seqs[0] is seq
    assert out.num_scheduled_tokens == [32]
    assert out.num_prefill_tokens == 32
    assert out.num_decode_tokens == 0
    assert seq.status == SequenceStatus.RUNNING
    assert seq.num_computed_tokens == 32
    assert len(seq.block_table) == 2
    assert bm.num_free_blocks == 18
    assert seq in sched.running
    assert len(sched.waiting) == 0


def test_multiple_prefills_admitted_in_one_step():
    sched, _ = make_scheduler(max_tokens=256)
    for i in range(3):
        sched.add_request(make_seq(i, 32))
    out = sched.schedule()
    assert len(out.scheduled_seqs) == 3
    assert out.num_prefill_tokens == 96


# ---------- decode-first ordering ----------


def test_decode_scheduled_before_new_prefills():
    sched, _ = make_scheduler(max_tokens=64)

    # admit one prefill in iter 1
    seq_a = make_seq(1, 16)
    sched.add_request(seq_a)
    sched.schedule()

    # iter 2: a new prefill arrives. seq_a is in running — should decode first.
    seq_b = make_seq(2, 32)
    sched.add_request(seq_b)
    out = sched.schedule()

    # seq_a comes first (decode), seq_b second (prefill)
    assert out.scheduled_seqs[0] is seq_a
    assert out.scheduled_seqs[1] is seq_b
    assert out.num_scheduled_tokens == [1, 32]
    assert out.num_decode_tokens == 1
    assert out.num_prefill_tokens == 32


# ---------- token budget ----------


def test_oversized_prompt_chunks_instead_of_deferring():
    """Under chunked prefill, a prompt larger than the per-iter budget no
    longer defers — it admits with a partial-prefill chunk."""
    sched, _ = make_scheduler(max_tokens=20)
    seq = make_seq(1, 32)
    sched.add_request(seq)
    out = sched.schedule()
    assert len(out.scheduled_seqs) == 1
    assert out.num_scheduled_tokens == [20]
    assert out.num_prefill_tokens == 20
    assert seq.status == SequenceStatus.RUNNING
    assert seq.num_computed_tokens == 20
    assert not seq.is_prefill_complete()
    assert len(sched.waiting) == 0


def test_budget_exhaustion_stops_admission_mid_loop():
    sched, _ = make_scheduler(max_tokens=64)
    # 3 prompts of 32 tokens each — only 2 fit in budget of 64
    for i in range(3):
        sched.add_request(make_seq(i, 32))
    out = sched.schedule()
    assert len(out.scheduled_seqs) == 2
    assert out.num_prefill_tokens == 64
    assert len(sched.waiting) == 1


def test_seq_budget_caps_admission():
    sched, _ = make_scheduler(max_seqs=2, max_tokens=10_000)
    for i in range(4):
        sched.add_request(make_seq(i, 16))
    out = sched.schedule()
    assert len(out.scheduled_seqs) == 2
    assert len(sched.waiting) == 2


# ---------- memory pressure ----------


def test_oom_defers_prefill_no_preemption():
    # only 2 blocks total; first prompt eats both
    sched, bm = make_scheduler(num_blocks=2, max_tokens=10_000)
    seq_a = make_seq(1, 32)  # 2 blocks
    seq_b = make_seq(2, 32)  # 2 more — won't fit
    sched.add_request(seq_a)
    sched.add_request(seq_b)
    out = sched.schedule()

    # seq_a admitted, seq_b deferred
    assert len(out.scheduled_seqs) == 1
    assert out.scheduled_seqs[0] is seq_a
    assert seq_b.status == SequenceStatus.WAITING
    assert seq_b in sched.waiting
    assert bm.num_free_blocks == 0


# ---------- finishing ----------


def test_free_finished_returns_blocks_and_ids():
    sched, bm = make_scheduler()
    seq = make_seq(1, 32)
    sched.add_request(seq)
    sched.schedule()
    assert bm.num_free_blocks == 18

    # simulate engine marking the seq finished
    seq.status = SequenceStatus.FINISHED
    finished = sched.free_finished()

    assert finished == [1]
    assert bm.num_free_blocks == 20
    assert seq not in sched.running


def test_unfinished_decodes_persist_across_steps():
    sched, _ = make_scheduler()
    seq = make_seq(1, 16)
    sched.add_request(seq)

    # iter 1: prefill
    out1 = sched.schedule()
    assert out1.num_prefill_tokens == 16

    # engine appends a token (simulated)
    seq.append_token(42)

    # iter 2: decode
    out2 = sched.schedule()
    assert out2.num_decode_tokens == 1
    assert out2.scheduled_seqs[0] is seq
