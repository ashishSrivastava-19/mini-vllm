"""Sarathi-Serve style chunked prefill: long prompts split across iterations,
and decodes are scheduled first so they're never stalled by a prefill."""

from mini_vllm.block_manager.manager import BlockManager
from mini_vllm.engine.sequence import Sequence
from mini_vllm.sampling.params import SamplingParams
from mini_vllm.scheduler.scheduler import Scheduler


def make_seq(seq_id: int, prompt_len: int, block_size: int = 16) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
        block_size=block_size,
    )


def make_scheduler(num_blocks=50, max_seqs=8, max_tokens=64, block_size=16):
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    sched = Scheduler(
        block_manager=bm,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max_tokens,
    )
    return sched, bm


def test_prompt_larger_than_budget_chunks_across_iterations():
    """200-token prompt, 64-token budget -> 4 iters: 64 + 64 + 64 + 8."""
    sched, _ = make_scheduler(max_tokens=64)
    seq = make_seq(1, 200)
    sched.add_request(seq)

    chunk_sizes = []
    while not seq.is_prefill_complete():
        out = sched.schedule()
        assert len(out.scheduled_seqs) == 1
        chunk_sizes.append(out.num_scheduled_tokens[0])
    assert chunk_sizes == [64, 64, 64, 8]
    assert seq.num_computed_tokens == 200


def test_decodes_take_priority_over_partial_prefill_chunk_size():
    """A long-prompt admission alongside 4 active decodes gets (budget - 4) tokens.
    Decodes are scheduled first; the prefill chunk shrinks to fit the remainder."""
    sched, _ = make_scheduler(max_tokens=64)

    for i in range(4):
        s = make_seq(i, 4)
        sched.add_request(s)
        sched.schedule()  # prefill completes in one iter (4 <= 64)
        s.append_token(99)  # simulate engine emitting a token -> decode mode

    long_seq = make_seq(100, 100)
    sched.add_request(long_seq)

    out = sched.schedule()
    assert out.num_decode_tokens == 4
    assert out.num_prefill_tokens == 60
    assert long_seq.num_computed_tokens == 60
    assert not long_seq.is_prefill_complete()


def test_partial_prefill_continues_next_iteration():
    """A prompt whose first chunk doesn't cover it continues on the next call."""
    sched, _ = make_scheduler(max_tokens=32)
    seq = make_seq(1, 50)
    sched.add_request(seq)

    out1 = sched.schedule()
    assert out1.num_scheduled_tokens == [32]
    assert seq.num_computed_tokens == 32
    assert not seq.is_prefill_complete()

    out2 = sched.schedule()
    assert out2.num_scheduled_tokens == [18]
    assert seq.num_computed_tokens == 50
    assert seq.is_prefill_complete()


def test_long_prompt_monopolizes_budget_short_prompt_defers():
    """With a greedy chunk policy (min(prompt_len, budget)), the first admitted
    prompt eats the whole budget; later waiting prompts wait one iter."""
    sched, _ = make_scheduler(max_tokens=64)
    long_seq = make_seq(1, 200)
    short_seq = make_seq(2, 4)
    sched.add_request(long_seq)
    sched.add_request(short_seq)

    out = sched.schedule()
    assert out.scheduled_seqs == [long_seq]
    assert out.num_scheduled_tokens == [64]
    assert short_seq in sched.waiting


def test_short_prompt_admitted_when_budget_allows():
    """If the long-prompt chunk leaves room, a second prompt fits the same iter."""
    sched, _ = make_scheduler(max_tokens=64)
    long_seq = make_seq(1, 50)
    short_seq = make_seq(2, 10)
    sched.add_request(long_seq)
    sched.add_request(short_seq)

    out = sched.schedule()
    assert long_seq in out.scheduled_seqs
    assert short_seq in out.scheduled_seqs
    assert sum(out.num_scheduled_tokens) == 60
    assert long_seq.is_prefill_complete()
    assert short_seq.is_prefill_complete()


def test_no_output_token_during_partial_prefill_then_first_token_when_complete():
    """Scheduler-only view: num_computed_tokens advances; outputs are the
    engine's concern and stay empty until prefill completes."""
    sched, _ = make_scheduler(max_tokens=32)
    seq = make_seq(1, 50)
    sched.add_request(seq)

    sched.schedule()
    assert not seq.is_prefill_complete()
    assert seq.output_token_ids == []

    sched.schedule()
    assert seq.is_prefill_complete()
    assert seq.output_token_ids == []  # engine.step() emits the first token next


def test_full_prompt_fits_budget_completes_in_one_iter():
    """Sanity: small prompts still admit in one shot, no chunking artifacts."""
    sched, _ = make_scheduler(max_tokens=64)
    seq = make_seq(1, 32)
    sched.add_request(seq)
    out = sched.schedule()
    assert out.num_scheduled_tokens == [32]
    assert seq.is_prefill_complete()


def test_partial_prefill_continuation_runs_before_new_admissions():
    """A continuing partial prefill consumes budget before any new prompt
    is admitted in the same iteration."""
    sched, _ = make_scheduler(max_tokens=64)
    long_seq = make_seq(1, 100)
    sched.add_request(long_seq)
    sched.schedule()  # iter 1: long_seq admitted, 64 tokens computed

    new_seq = make_seq(2, 20)
    sched.add_request(new_seq)
    out = sched.schedule()  # iter 2

    # pass 2 continues long_seq (36 remaining); pass 3 admits new_seq with what's left
    assert out.scheduled_seqs[0] is long_seq
    assert out.num_scheduled_tokens[0] == 36
    assert long_seq.is_prefill_complete()
    assert new_seq in out.scheduled_seqs
    assert new_seq.num_computed_tokens == 20
