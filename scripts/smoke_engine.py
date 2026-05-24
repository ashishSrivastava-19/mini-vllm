"""REPL-friendly smoke test for LLMEngine + Sampler.

Run with:
    uv run python scripts/smoke_engine.py
"""

import torch

from mini_vllm.block_manager.manager import BlockManager
from mini_vllm.engine.llm_engine import LLMEngine
from mini_vllm.model_runner.stub import StubModelRunner
from mini_vllm.sampling.params import SamplingParams
from mini_vllm.scheduler.scheduler import Scheduler

VOCAB = 128
EOS = 0


def build_engine(seed: int = 0) -> LLMEngine:
    bm = BlockManager(num_blocks=32, block_size=16)
    sched = Scheduler(block_manager=bm, max_num_seqs=8, max_num_batched_tokens=512)
    runner = StubModelRunner(vocab_size=VOCAB, seed=seed)
    return LLMEngine(
        model_runner=runner,
        block_manager=bm,
        scheduler=sched,
        eos_token_id=EOS,
        max_model_len=128,
    )


def demo_greedy():
    print("\n=== greedy (temperature=0, max_tokens=5) ===")
    eng = build_engine(seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=5)
    out = eng.generate(["hello"], [[1, 2, 3, 4]], sp)
    print("token_ids:", out[0].outputs[0].token_ids)
    print("finish_reason:", out[0].outputs[0].finish_reason)


def demo_stochastic():
    print("\n=== stochastic (T=0.8, top_p=0.95, top_k=50, max_tokens=10) ===")
    torch.manual_seed(0)
    eng = build_engine(seed=0)
    sp = SamplingParams(temperature=0.8, top_p=0.95, top_k=50, max_tokens=10)
    out = eng.generate(["hello"], [[1, 2, 3, 4]], sp)
    print("token_ids:", out[0].outputs[0].token_ids)
    print("finish_reason:", out[0].outputs[0].finish_reason)


def demo_mixed_batch():
    print("\n=== mixed batch: greedy + stochastic ===")
    torch.manual_seed(0)
    eng = build_engine(seed=0)
    params = [
        SamplingParams(temperature=0.0, max_tokens=4),
        SamplingParams(temperature=1.0, top_k=10, max_tokens=4),
    ]
    out = eng.generate(
        prompts=["greedy", "stochastic"],
        prompt_token_ids=[[1, 2, 3], [4, 5, 6, 7]],
        sampling_params=params,
    )
    for o in out:
        print(f"{o.prompt!r}: tokens={o.outputs[0].token_ids} reason={o.outputs[0].finish_reason}")


def demo_stop_token_ids():
    print("\n=== stop_token_ids = [42], forced argmax to 42 ===")
    eng = build_engine(seed=0)
    eng.model_runner.execute = lambda sched_out: torch.full(
        (len(sched_out.scheduled_seqs), VOCAB), -1e9
    ).index_fill_(1, torch.tensor([42]), 1e9)
    sp = SamplingParams(temperature=0.0, max_tokens=100, stop_token_ids=[42])
    out = eng.generate(["hi"], [[1, 2, 3]], sp)
    print("token_ids:", out[0].outputs[0].token_ids)
    print("finish_reason:", out[0].outputs[0].finish_reason)


if __name__ == "__main__":
    demo_greedy()
    demo_stochastic()
    demo_mixed_batch()
    demo_stop_token_ids()
