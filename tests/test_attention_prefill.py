import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_vllm.config import Config
from mini_vllm.engine.sequence import Sequence
from mini_vllm.sampling.params import SamplingParams
from mini_vllm.scheduler.scheduler import SchedulerOutput

flash_attn = pytest.importorskip("flash_attn")
from mini_vllm.layers.attention_patch import patch_model  # noqa: E402
from mini_vllm.model_runner.runner import ModelRunner  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
BLOCK_SIZE = 16
NUM_BLOCKS = 64
MAX_BLOCKS_PER_SEQ = 8  # leaves room for prompts up to 128 tokens


def _seq(seq_id: int, token_ids: list[int]) -> Sequence:
    """Stand in for what the scheduler would produce just before runner.execute():
    num_computed_tokens advanced to the full prompt, block_table populated with
    enough distinct physical blocks to cover the prompt."""
    s = Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(token_ids),
        sampling_params=SamplingParams(temperature=0.0),
        block_size=BLOCK_SIZE,
    )
    n_blocks = (len(token_ids) + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert n_blocks <= MAX_BLOCKS_PER_SEQ, "test prompt too long for fixture"
    start = seq_id * MAX_BLOCKS_PER_SEQ
    s.block_table = list(range(start, start + n_blocks))
    s.num_computed_tokens = len(s.prompt_token_ids)
    return s


def _sched_out(seqs: list[Sequence]) -> SchedulerOutput:
    lens = [len(s.prompt_token_ids) for s in seqs]
    return SchedulerOutput(
        scheduled_seqs=seqs,
        num_scheduled_tokens=lens,
        num_prefill_tokens=sum(lens),
    )


def _build_runners(tok):
    cfg = Config(
        model_path=MODEL, dtype=torch.float16, device="cuda", block_size=BLOCK_SIZE
    ).validate()

    m_ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    runner_ref = ModelRunner(m_ref, cfg, pad_token_id=tok.pad_token_id)

    m_flash = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    patch_model(m_flash)
    runner_flash = ModelRunner(m_flash, cfg, pad_token_id=tok.pad_token_id)
    runner_flash.init_kv_cache(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    return runner_ref, runner_flash


def _assert_logits_agree(logits_ref: torch.Tensor, logits_flash: torch.Tensor) -> None:
    """Load-bearing invariant: same argmax and same top-5. Raw fp16 abs-diff
    across 24 layers + kernel swap can reach a few percent — that's drift, not
    a correctness failure. Argmax / top-K equality is what generation depends on."""
    assert logits_ref.shape == logits_flash.shape
    argmax_ref = logits_ref.argmax(dim=-1)
    argmax_flash = logits_flash.argmax(dim=-1)
    assert torch.equal(argmax_ref, argmax_flash), (
        f"argmax mismatch:\n  ref:   {argmax_ref.tolist()}\n  flash: {argmax_flash.tolist()}"
    )
    top5_ref = logits_ref.topk(5, dim=-1).indices
    top5_flash = logits_flash.topk(5, dim=-1).indices
    # Top-5 sets should match (order within may differ from tie-breaking).
    for i in range(logits_ref.shape[0]):
        assert set(top5_ref[i].tolist()) == set(top5_flash[i].tolist()), (
            f"row {i} top-5 mismatch:\n  ref:   {top5_ref[i].tolist()}\n  flash: {top5_flash[i].tolist()}"
        )
    max_diff = (logits_ref.float() - logits_flash.float()).abs().max().item()
    # Soft bound — well above expected fp16 drift but tight enough to catch a real bug.
    assert max_diff < 5e-1, f"max abs diff {max_diff:.4e} is unreasonably large; likely a bug"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flash_prefill_matches_sdpa():
    """Two-prompt batch through SDPA vs flash-attn varlen: argmax + top-5 must match."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    runner_ref, runner_flash = _build_runners(tok)

    prompts = ["Hello, my name is", "The capital of France is"]
    ref_seqs = [_seq(i, tok.encode(p)) for i, p in enumerate(prompts)]
    flash_seqs = [_seq(i, tok.encode(p)) for i, p in enumerate(prompts)]

    logits_ref = runner_ref.execute_sdpa(_sched_out(ref_seqs))
    logits_flash = runner_flash.execute(_sched_out(flash_seqs))

    _assert_logits_agree(logits_ref, logits_flash)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flash_prefill_single_sequence():
    """One-sequence batch — exercises cu_seqlens of length 2."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    runner_ref, runner_flash = _build_runners(tok)

    ids = tok.encode("The quick brown fox jumps over the lazy dog")
    logits_ref = runner_ref.execute_sdpa(_sched_out([_seq(0, ids)]))
    logits_flash = runner_flash.execute(_sched_out([_seq(0, ids)]))

    _assert_logits_agree(logits_ref, logits_flash)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunked_prefill_raises_not_implemented():
    """Tuesday's runner forbids chunked prefill (a continuation chunk where the
    prompt is split across iters). One-shot prefill must still work; only
    mid-prompt continuation should raise."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    _, runner_flash = _build_runners(tok)

    ids = tok.encode("The capital of France is")
    s = _seq(0, ids)
    # Simulate a continuation chunk: scheduler advanced num_computed_tokens past
    # the start of the prompt but not all the way to the end.
    half = max(1, len(ids) // 2)
    s.num_computed_tokens = half + 1  # so start = 1, end = half + 1 — neither full nor first
    sched = SchedulerOutput(
        scheduled_seqs=[s],
        num_scheduled_tokens=[half],
        num_prefill_tokens=half,
    )
    with pytest.raises(NotImplementedError):
        runner_flash.execute(sched)
