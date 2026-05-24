"""REPL-friendly smoke test for the real ModelRunner via build_engine.

Run with:
    uv run python scripts/smoke_real_model.py
"""

import torch

from mini_vllm.config import Config
from mini_vllm.engine.builder import build_engine
from mini_vllm.sampling.params import SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B"


def make_config() -> Config:
    return Config(
        model_path=MODEL,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_num_seqs=4,
        max_num_batched_tokens=512,
        max_model_len=256,
        block_size=16,
    )


def demo_greedy_single():
    print("\n=== greedy, 1 prompt, max_tokens=16 ===")
    cfg = make_config()
    eng, tok = build_engine(cfg, num_blocks=64)
    prompt = "The capital of France is"
    ids = tok.encode(prompt)
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    out = eng.generate([prompt], [ids], sp)
    print(f"prompt: {prompt!r}")
    print(f"tokens: {out[0].outputs[0].token_ids}")
    print(f"text:   {out[0].outputs[0].text!r}")
    print(f"reason: {out[0].outputs[0].finish_reason}")


def demo_batched_greedy():
    print("\n=== greedy, 3 prompts, max_tokens=8 ===")
    cfg = make_config()
    eng, tok = build_engine(cfg, num_blocks=128)
    prompts = [
        "The capital of France is",
        "Once upon a time",
        "In a hole in the ground there lived",
    ]
    ids = [tok.encode(p) for p in prompts]
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    out = eng.generate(prompts, ids, sp)
    for o in out:
        print(f"  {o.prompt!r:50s} -> {o.outputs[0].text!r}")


def demo_stochastic():
    print("\n=== stochastic (T=0.8, top_p=0.95, top_k=50, max_tokens=16) ===")
    torch.manual_seed(0)
    cfg = make_config()
    eng, tok = build_engine(cfg, num_blocks=64)
    prompt = "Once upon a time"
    ids = tok.encode(prompt)
    sp = SamplingParams(temperature=0.8, top_p=0.95, top_k=50, max_tokens=16)
    out = eng.generate([prompt], [ids], sp)
    print(f"prompt: {prompt!r}")
    print(f"text:   {out[0].outputs[0].text!r}")


def demo_chunked_prefill():
    """Prompt larger than max_num_batched_tokens — exercises Sarathi-Serve chunking.
    Output tokens should be identical to what a single-iter prefill would produce."""
    print("\n=== chunked prefill (prompt >> per-iter budget) ===")
    cfg = make_config()
    cfg.max_num_batched_tokens = 64  # force chunking
    cfg.max_model_len = 1024
    eng, tok = build_engine(cfg, num_blocks=128)
    long_prompt = "The history of the Roman Empire began in 27 BC with " * 20
    ids = tok.encode(long_prompt)
    print(f"prompt length: {len(ids)} tokens, budget: {cfg.max_num_batched_tokens}/iter")
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    out = eng.generate([long_prompt], [ids], sp)
    print(f"finish: {out[0].outputs[0].finish_reason}")
    print(f"text:   {out[0].outputs[0].text!r}")


if __name__ == "__main__":
    print(f"loading {MODEL} on {'cuda' if torch.cuda.is_available() else 'cpu'}...")
    demo_greedy_single()
    demo_batched_greedy()
    demo_stochastic()
    demo_chunked_prefill()
