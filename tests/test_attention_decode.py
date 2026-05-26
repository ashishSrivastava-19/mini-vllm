import pytest
import torch

flash_attn = pytest.importorskip("flash_attn")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from mini_vllm.config import Config  # noqa: E402
from mini_vllm.engine.builder import build_engine  # noqa: E402
from mini_vllm.sampling.params import SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Once upon a time, there was a",
    "def fibonacci(n):",
]


@pytest.fixture(scope="module")
def hf():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    return m, tok


@pytest.fixture(scope="module")
def engine():
    cfg = Config(
        model_path=MODEL,
        dtype=torch.float16,
        device="cuda",
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        max_model_len=512,
        block_size=256,  # flash_attn 2.7 paged kv kernel requires page_block_size % 256 == 0
    ).validate()
    eng, tok = build_engine(cfg, num_blocks=32)
    return eng, tok


def _hf_greedy(model, tok, prompt: str, n: int) -> list[int]:
    ids = tok.encode(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            max_new_tokens=n,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
        )
    return out[0, ids.shape[1] :].tolist()


def _eng_greedy(eng, tok, prompt: str, n: int) -> list[int]:
    ids = tok.encode(prompt)
    sp = SamplingParams(temperature=0.0, max_tokens=n)
    out = eng.generate([prompt], [ids], sp)
    return out[0].outputs[0].token_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("prompt", PROMPTS)
def test_paged_decode_matches_hf_single(prompt, engine, hf):
    """End-to-end: paged-attention engine should generate the same tokens as
    HF model.generate() under greedy decoding."""
    eng, eng_tok = engine
    hf_m, hf_tok = hf
    hf_tokens = _hf_greedy(hf_m, hf_tok, prompt, 20)
    eng_tokens = _eng_greedy(eng, eng_tok, prompt, 20)
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(hf_tokens, eng_tokens)) if a != b), None
    )
    assert eng_tokens == hf_tokens, (
        f"\nprompt: {prompt!r}\nHF:  {hf_tokens}\nENG: {eng_tokens}\nfirst diff @ {first_diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_paged_decode_matches_hf_batch(engine, hf):
    """Same as above but as a multi-sequence batch — exercises mixed and pure-decode iters."""
    eng, eng_tok = engine
    hf_m, hf_tok = hf
    hf_outs = [_hf_greedy(hf_m, hf_tok, p, 20) for p in PROMPTS]
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    ids = [eng_tok.encode(p) for p in PROMPTS]
    eng_outs = eng.generate(PROMPTS, ids, sp)
    for p, hf_t, out in zip(PROMPTS, hf_outs, eng_outs, strict=True):
        eng_t = out.outputs[0].token_ids
        assert eng_t == hf_t, f"\nprompt: {p!r}\nHF:  {hf_t}\nENG: {eng_t}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_paged_decode_terminates_on_eos(engine, hf):
    """Greedy generation should terminate at the same EOS as HF."""
    eng, eng_tok = engine
    hf_m, hf_tok = hf
    prompt = "Q: what is 9x8? A:"
    hf_tokens = _hf_greedy(hf_m, hf_tok, prompt, 50)
    eng_tokens = _eng_greedy(eng, eng_tok, prompt, 50)
    assert eng_tokens == hf_tokens
