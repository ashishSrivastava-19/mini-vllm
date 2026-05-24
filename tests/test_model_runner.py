"""Correctness tests for ModelRunner v0.

These tests load a small real HF model. The first run downloads it; subsequent
runs are cached. If the model can't be loaded (no network, no disk), the whole
module is skipped — these tests are not part of the fast unit suite.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from mini_vllm.config import Config  # noqa: E402
from mini_vllm.engine.sequence import Sequence  # noqa: E402
from mini_vllm.model_runner.runner import ModelRunner  # noqa: E402
from mini_vllm.sampling.params import SamplingParams  # noqa: E402
from mini_vllm.scheduler.scheduler import SchedulerOutput  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model_and_tokenizer(device):
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
        m.to(device)
        m.eval()
    except Exception as e:
        pytest.skip(f"could not load {MODEL}: {e}")
    return m, tok


@pytest.fixture(scope="module")
def runner(model_and_tokenizer, device):
    m, tok = model_and_tokenizer
    cfg = Config(model_path=MODEL, dtype=torch.float32, device=device)
    return ModelRunner(model=m, config=cfg, pad_token_id=tok.pad_token_id)


def make_seq(seq_id, token_ids):
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(token_ids),
        sampling_params=SamplingParams(temperature=0.0),
        block_size=16,
    )


def test_logits_shape(runner):
    seq = make_seq(0, [1, 2, 3, 4, 5])
    sched_out = SchedulerOutput(scheduled_seqs=[seq], num_scheduled_tokens=[5])
    logits = runner.execute(sched_out)
    assert logits.shape == (1, runner.vocab_size)


def test_batch_logits_shape(runner):
    seqs = [make_seq(i, list(range(1, 4 + i))) for i in range(3)]
    sched_out = SchedulerOutput(
        scheduled_seqs=seqs,
        num_scheduled_tokens=[len(s.prompt_token_ids) for s in seqs],
    )
    logits = runner.execute(sched_out)
    assert logits.shape == (3, runner.vocab_size)


def test_argmax_matches_hf_for_single_sequence(runner, model_and_tokenizer, device):
    """The runner's top-1 token for a prompt must match HF's greedy choice."""
    m, tok = model_and_tokenizer
    prompt = "The capital of France is"
    ids = tok.encode(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        hf_out = m(input_ids=ids, use_cache=False)
        hf_next = hf_out.logits[0, -1].argmax().item()

    seq = make_seq(0, ids[0].tolist())
    sched_out = SchedulerOutput(
        scheduled_seqs=[seq],
        num_scheduled_tokens=[len(seq.prompt_token_ids)],
    )
    runner_logits = runner.execute(sched_out)
    runner_next = runner_logits[0].argmax().item()

    assert runner_next == hf_next


def test_padding_does_not_change_logits_of_padded_row(runner, model_and_tokenizer):
    """Load-bearing: A's last-token logits must be invariant to whether A is
    batched alone or alongside a longer sequence B. If this fails, left-pad +
    attention_mask + RoPE position derivation is broken."""
    _, tok = model_and_tokenizer
    ids_a = tok.encode("The cat sat on the")
    ids_b = tok.encode("Once upon a time in a faraway kingdom there")

    seq_a_alone = make_seq(0, ids_a)
    sched_alone = SchedulerOutput(
        scheduled_seqs=[seq_a_alone],
        num_scheduled_tokens=[len(ids_a)],
    )
    logits_alone = runner.execute(sched_alone)[0]

    seq_a_batched = make_seq(0, ids_a)
    seq_b = make_seq(1, ids_b)
    sched_batched = SchedulerOutput(
        scheduled_seqs=[seq_a_batched, seq_b],
        num_scheduled_tokens=[len(ids_a), len(ids_b)],
    )
    logits_batched = runner.execute(sched_batched)[0]

    torch.testing.assert_close(logits_alone, logits_batched, rtol=1e-4, atol=1e-4)


def test_end_to_end_single_step_via_engine(runner, model_and_tokenizer):
    """Smoke test: full engine + real runner, one step, one token out."""
    from mini_vllm.block_manager.manager import BlockManager
    from mini_vllm.engine.llm_engine import LLMEngine
    from mini_vllm.scheduler.scheduler import Scheduler

    _, tok = model_and_tokenizer
    bm = BlockManager(num_blocks=64, block_size=16)
    sched = Scheduler(block_manager=bm, max_num_seqs=8, max_num_batched_tokens=512)
    eng = LLMEngine(
        model_runner=runner,
        block_manager=bm,
        scheduler=sched,
        eos_token_id=tok.eos_token_id,
        max_model_len=128,
        tokenizer=tok,
    )
    prompt = "Hello, my name is"
    ids = tok.encode(prompt)
    eng.add_request(prompt, ids, SamplingParams(temperature=0.0, max_tokens=1))
    outputs = []
    while eng.has_unfinished_requests():
        outputs.extend(eng.step())
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 1
    assert outputs[0].outputs[0].finish_reason == "length"
