import torch

from mini_vllm.block_manager.manager import BlockManager
from mini_vllm.engine.llm_engine import LLMEngine
from mini_vllm.model_runner.stub import StubModelRunner
from mini_vllm.sampling.params import SamplingParams
from mini_vllm.scheduler.scheduler import Scheduler

VOCAB = 100
EOS = 0


def greedy(max_tokens: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def make_engine(
    num_blocks=20,
    max_seqs=8,
    max_tokens_budget=256,
    block_size=16,
    eos=EOS,
    max_model_len=128,
    seed=0,
):
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    sched = Scheduler(
        block_manager=bm,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max_tokens_budget,
    )
    runner = StubModelRunner(vocab_size=VOCAB, seed=seed)
    eng = LLMEngine(
        model_runner=runner,
        block_manager=bm,
        scheduler=sched,
        eos_token_id=eos,
        max_model_len=max_model_len,
    )
    return eng, bm, sched


def test_step_on_empty_engine_is_noop():
    eng, _, _ = make_engine()
    assert eng.step() == []


def test_single_request_finishes_at_max_tokens():
    eng, bm, _ = make_engine()
    eng.add_request("hi", [1, 2, 3], greedy(4))
    outputs: list = []
    while eng.has_unfinished_requests():
        outputs.extend(eng.step())
    assert len(outputs) == 1
    out = outputs[0]
    assert out.finished
    assert out.outputs[0].finish_reason == "length"
    assert len(out.outputs[0].token_ids) == 4
    assert bm.num_free_blocks == 20


def test_eos_terminates_early():
    eng, bm, _ = make_engine()
    # Force argmax to land on EOS every step.
    eng.model_runner.execute = lambda sched_out: torch.full(
        (len(sched_out.scheduled_seqs), VOCAB), -1e9
    ).index_fill_(1, torch.tensor([EOS]), 1e9)

    eng.add_request("hi", [1, 2, 3], greedy(100))
    outputs: list = []
    while eng.has_unfinished_requests():
        outputs.extend(eng.step())
    assert outputs[0].outputs[0].finish_reason == "stop"
    assert outputs[0].outputs[0].token_ids == [EOS]
    assert bm.num_free_blocks == 20


def test_multiple_concurrent_requests_all_finish():
    eng, bm, _ = make_engine()
    for i in range(4):
        eng.add_request(f"p{i}", list(range(8 + i)), greedy(3))
    outputs: list = []
    while eng.has_unfinished_requests():
        outputs.extend(eng.step())
    assert len(outputs) == 4
    assert all(o.outputs[0].finish_reason == "length" for o in outputs)
    assert all(len(o.outputs[0].token_ids) == 3 for o in outputs)
    assert bm.num_free_blocks == 20


def test_output_order_matches_request_order():
    eng, _, _ = make_engine()
    prompts = ["zero", "one", "two", "three"]
    out = eng.generate(
        prompts=prompts,
        prompt_token_ids=[list(range(4 + i)) for i in range(4)],
        sampling_params=greedy(2),
    )
    assert [o.prompt for o in out] == prompts


def test_generate_handles_mixed_prompt_lengths():
    eng, bm, _ = make_engine(num_blocks=40, max_tokens_budget=512)
    outs = eng.generate(
        prompts=["short", "med", "long"],
        prompt_token_ids=[list(range(4)), list(range(20)), list(range(50))],
        sampling_params=greedy(3),
    )
    assert len(outs) == 3
    assert all(o.finished for o in outs)
    assert bm.num_free_blocks == 40


def test_max_model_len_truncates():
    # Prompt has 8 tokens, max_model_len=10 -> only 2 output tokens allowed
    # before num_tokens (prompt + output) hits the cap.
    eng, _, _ = make_engine(max_model_len=10)
    eng.add_request("hi", list(range(8)), greedy(100))
    outputs: list = []
    while eng.has_unfinished_requests():
        outputs.extend(eng.step())
    assert outputs[0].outputs[0].finish_reason == "max_model_len"
    assert len(outputs[0].outputs[0].token_ids) == 2
