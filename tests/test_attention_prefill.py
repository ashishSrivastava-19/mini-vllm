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


def _seq(seq_id: int, token_ids: list[int]) -> Sequence:
    s = Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(token_ids),
        sampling_params=SamplingParams(temperature=0.0),
        block_size=16,
    )
    # Scheduler would have advanced num_computed_tokens to chunk size before runner runs.
    s.num_computed_tokens = len(s.prompt_token_ids)
    return s


def _sched_out(seqs: list[Sequence]) -> SchedulerOutput:
    lens = [len(s.prompt_token_ids) for s in seqs]
    return SchedulerOutput(
        scheduled_seqs=seqs,
        num_scheduled_tokens=lens,
        num_prefill_tokens=sum(lens),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flash_prefill_matches_sdpa():
    """Same prompts through SDPA path and flash-attn varlen path should produce
    near-identical last-token logits (fp16 tolerance)."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    cfg = Config(model_path=MODEL, dtype=torch.float16, device="cuda").validate()

    # SDPA reference
    m_ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    runner_ref = ModelRunner(m_ref, cfg, pad_token_id=tok.pad_token_id)

    # Flash-attn varlen
    m_flash = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    runner_flash = ModelRunner(m_flash, cfg, pad_token_id=tok.pad_token_id)
    patch_model(runner_flash.model)

    prompts = ["Hello, my name is", "The capital of France is"]
    ref_seqs = [_seq(i, tok.encode(p)) for i, p in enumerate(prompts)]
    flash_seqs = [_seq(i, tok.encode(p)) for i, p in enumerate(prompts)]

    logits_ref = runner_ref.execute_sdpa(_sched_out(ref_seqs))
    logits_flash = runner_flash.execute(_sched_out(flash_seqs))

    assert logits_ref.shape == logits_flash.shape
    max_diff = (logits_ref.float() - logits_flash.float()).abs().max().item()
    assert max_diff < 1e-2, f"max abs diff {max_diff:.4e} exceeds 1e-2"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flash_prefill_single_sequence():
    """One-sequence batch should still work — cu_seqlens has length 2."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    cfg = Config(model_path=MODEL, dtype=torch.float16, device="cuda").validate()
    m_ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    runner_ref = ModelRunner(m_ref, cfg, pad_token_id=tok.pad_token_id)
    m_flash = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    runner_flash = ModelRunner(m_flash, cfg, pad_token_id=tok.pad_token_id)
    patch_model(runner_flash.model)

    ids = tok.encode("The quick brown fox jumps over the lazy dog")
    logits_ref = runner_ref.execute_sdpa(_sched_out([_seq(0, ids)]))
    logits_flash = runner_flash.execute(_sched_out([_seq(0, ids)]))

    max_diff = (logits_ref.float() - logits_flash.float()).abs().max().item()
    assert max_diff < 1e-2, f"max abs diff {max_diff:.4e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flash_decode_raises_not_implemented():
    """Monday limitation: a fully-prefilled sequence should raise on execute()."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    cfg = Config(model_path=MODEL, dtype=torch.float16, device="cuda").validate()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    runner = ModelRunner(m, cfg, pad_token_id=tok.pad_token_id)
    patch_model(runner.model)

    s = _seq(0, tok.encode("hi"))
    s.num_computed_tokens = len(s.prompt_token_ids)
    s.output_token_ids.append(42)  # one decoded token already -> is_prefill_complete()

    sched = SchedulerOutput(
        scheduled_seqs=[s],
        num_scheduled_tokens=[1],
        num_decode_tokens=1,
    )
    with pytest.raises(NotImplementedError):
        runner.execute(sched)
