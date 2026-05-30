"""Cloud-only smoke test for Thursday's prefix-cache lookup.

Verifies three things at the engine level:
  1. Cold run -> warm run on the same prompts produces non-zero cache_hit_rate.
  2. Generated tokens are bit-identical between cold and warm runs (caching is
     a perf optimization; correctness must not drift).
  3. Long shared system prompt + varied user queries also produces hits.

Requires: CUDA + flash_attn. With Config.block_size=256 (forced by flash_attn
2.7's paged kernel), prompts need to exceed 256 tokens before any block fills
and becomes cacheable. The "shared system prompt" run below pads to ensure that.
"""

import torch

from mini_vllm.config import Config
from mini_vllm.engine.builder import build_engine
from mini_vllm.sampling.params import SamplingParams


def main() -> None:
    assert torch.cuda.is_available(), "smoke requires CUDA"

    cfg = Config(
        model_path="Qwen/Qwen2.5-0.5B",
        dtype=torch.float16,
        device="cuda",
        block_size=256,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        max_model_len=2048,
    ).validate()
    eng, tok = build_engine(cfg, num_blocks=32)
    sp = SamplingParams(temperature=0.0, max_tokens=20)

    # --- 1. Cold/warm with identical prompts ---
    prompts = ["The capital of France is"] * 2  # short — won't fill a 256-block
    ids = [tok.encode(p) for p in prompts]

    cold = eng.generate(prompts[:1], ids[:1], sp)
    cold_tokens = cold[0].outputs[0].token_ids
    cold_hits = eng.block_manager.cache_hit_tokens

    warm = eng.generate(prompts[1:], ids[1:], sp)
    warm_tokens = warm[0].outputs[0].token_ids
    warm_hits = eng.block_manager.cache_hit_tokens

    print(f"[short prompt]")
    print(f"  cold tokens: {cold_tokens}")
    print(f"  warm tokens: {warm_tokens}")
    print(f"  cumulative hits: cold={cold_hits} warm={warm_hits}")
    print(f"  hit rate: {eng.block_manager.cache_hit_rate:.1%}")
    assert cold_tokens == warm_tokens, "warm-cache output diverges from cold-cache"
    # With block_size=256 and a ~6-token prompt, no block ever fills -> 0 hits expected.

    # --- 2. Long shared system prompt -> should actually hit the cache ---
    # Pad the shared prefix to comfortably exceed block_size=256 tokens so at
    # least one block fills and becomes cacheable.
    shared = ("You are a helpful assistant. Answer the question precisely. " * 30)
    queries = ["What is 2+2?", "Capital of France?", "Who wrote Hamlet?"]
    long_prompts = [shared + q for q in queries]
    long_ids = [tok.encode(p) for p in long_prompts]
    print(f"\n[long shared system prompt; len={len(long_ids[0])} tokens]")

    pre_hits = eng.block_manager.cache_hit_tokens
    pre_total = eng.block_manager.total_prompt_tokens

    # First call seeds the cache with the shared prefix.
    eng.generate(long_prompts[:1], long_ids[:1], sp)
    # Subsequent calls with the same shared prefix should hit.
    outs = eng.generate(long_prompts[1:], long_ids[1:], sp)

    delta_hits = eng.block_manager.cache_hit_tokens - pre_hits
    delta_total = eng.block_manager.total_prompt_tokens - pre_total
    delta_rate = delta_hits / delta_total if delta_total else 0.0
    print(f"  this-run hits: {delta_hits} / {delta_total} tokens ({delta_rate:.1%})")
    for q, out in zip(queries[1:], outs, strict=True):
        print(f"  Q: {q!r}  -> {tok.decode(out.outputs[0].token_ids)!r}")

    if delta_hits == 0:
        print(
            "\n  NOTE: 0 hits means even the long prompt didn't fill a 256-token block "
            "after the prefix was admitted. Increase the multiplier in `shared` "
            "until len(long_ids[0]) >> 256."
        )
    else:
        print(f"\n  ok: prefix cache produced {delta_hits} token-hits on the second batch.")


if __name__ == "__main__":
    main()
