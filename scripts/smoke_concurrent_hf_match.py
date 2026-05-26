"""Smoke test: 4+ concurrent mixed-length prompts in a single engine.generate()
call must produce the exact same greedy tokens as HF's .generate() per prompt.

Run with:
    uv run python scripts/smoke_concurrent_hf_match.py
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_vllm.config import Config
from mini_vllm.engine.builder import build_engine
from mini_vllm.sampling.params import SamplingParams


MODEL = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 20

PROMPTS = [
    "Hi.",  # very short
    "The capital of France is",  # short
    "Once upon a time in a kingdom far, far away there lived",  # medium
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return ",  # code, medium
    (
        "The Roman Empire began in 27 BC with the reign of Augustus. "
        "Over the next four centuries it expanded across the Mediterranean basin, "
        "covering parts of three continents. Its decline began in the third century, "
        "and the western half fell in"
    ),  # long
]


def main() -> int:
    torch.manual_seed(0)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"loading {MODEL} on {device} ({dtype})...")

    # --- HF reference ---
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()

    # --- engine ---
    cfg = Config(
        model_path=MODEL,
        dtype=dtype,
        device=device,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        max_model_len=512,
    )
    eng, eng_tok = build_engine(cfg, num_blocks=256)

    # --- HF: one prompt at a time, greedy ---
    hf_outputs: list[list[int]] = []
    for prompt in PROMPTS:
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = hf.generate(
                input_ids=ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
            )
        hf_outputs.append(out[0, ids.shape[1] :].tolist())

    # --- engine: all prompts in one .generate() call (concurrent batch) ---
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    eng_ids = [eng_tok.encode(p) for p in PROMPTS]
    print(f"\nbatching {len(PROMPTS)} concurrent prompts "
          f"(lengths {[len(i) for i in eng_ids]})")
    eng_outputs_raw = eng.generate(PROMPTS, eng_ids, sp)

    # --- compare token-for-token ---
    n_pass = 0
    n_fail = 0
    for prompt, hf_t, out in zip(PROMPTS, hf_outputs, eng_outputs_raw, strict=True):
        eng_t = out.outputs[0].token_ids
        ok = eng_t == hf_t
        n_pass += int(ok)
        n_fail += int(not ok)
        status = "OK" if ok else "FAIL"
        print(f"\n[{status}] prompt[:40]={prompt[:40]!r}")
        print(f"  prompt_tokens : {len(out.prompt_token_ids)}")
        print(f"  finish_reason : {out.outputs[0].finish_reason}")
        if ok:
            print(f"  text          : {out.outputs[0].text!r}")
        else:
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(hf_t, eng_t)) if a != b), None
            )
            print(f"  HF tokens     : {hf_t}")
            print(f"  engine tokens : {eng_t}")
            print(f"  first diff @  : {first_diff if first_diff is not None else 'length'}")

    print(f"\n=== summary: {n_pass}/{len(PROMPTS)} matched HF; {n_fail} mismatched ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
