import os
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_vllm.engine.builder import build_engine
from mini_vllm.config import Config
from mini_vllm.sampling.params import SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

torch.manual_seed(42)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(scope="module")
def hf_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE)
    model.to(DEVICE)
    model.eval()
    return model, tok


@pytest.fixture(scope="module")
def engine_and_tokenizer():
    cfg = Config(
        model_path=MODEL,
        dtype=DTYPE,
        device=DEVICE,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        max_model_len=512,
    )
    engine, tokenizer = build_engine(cfg, num_blocks=256)
    return engine, tokenizer


def hf_greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return out[0, ids.shape[1] :].tolist()


def engine_greedy_generate(engine, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    ids = tokenizer.encode(prompt)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    out = engine.generate([prompt], [ids], sp)
    return out[0].outputs[0].token_ids


PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Once upon a time, there was a",
    "In the year 2024, the world will",
    "def fibonacci(n):",
]


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_hf_single_prompt(prompt, engine_and_tokenizer, hf_model_and_tokenizer):
    eng, eng_tok = engine_and_tokenizer
    hf, hf_tok = hf_model_and_tokenizer

    hf_tokens = hf_greedy_generate(hf, hf_tok, prompt, max_new_tokens=20)
    eng_tokens = engine_greedy_generate(eng, eng_tok, prompt, max_new_tokens=20)

    first_diff = next(
        (i for i, (a, b) in enumerate(zip(hf_tokens, eng_tokens)) if a != b),
        None,
    )
    assert eng_tokens == hf_tokens, (
        f"Mismatch on prompt '{prompt}'\n"
        f"  HF:     {hf_tokens}\n"
        f"  engine: {eng_tokens}\n"
        f"  first diff at position: {first_diff if first_diff is not None else 'length differs'}"
    )


def test_greedy_matches_hf_batch(engine_and_tokenizer, hf_model_and_tokenizer):
    eng, eng_tok = engine_and_tokenizer
    hf, hf_tok = hf_model_and_tokenizer

    hf_outputs = [hf_greedy_generate(hf, hf_tok, prompt, max_new_tokens=20) for prompt in PROMPTS]

    sp = SamplingParams(temperature=0.0, max_tokens=20)
    ids = [eng_tok.encode(prompt) for prompt in PROMPTS]
    eng_outputs = eng.generate(PROMPTS, ids, sp)

    for prompt, hf_t, out in zip(PROMPTS, hf_outputs, eng_outputs, strict=True):
        eng_t = out.outputs[0].token_ids
        assert eng_t == hf_t, (
            f"Mismatch on prompt '{prompt}': engine tokens {eng_t} vs hf tokens {hf_t}"
        )


def test_greedy_terminates_on_eos(engine_and_tokenizer, hf_model_and_tokenizer):
    eng, eng_tok = engine_and_tokenizer
    hf, hf_tok = hf_model_and_tokenizer

    prompt = "Q: what is 9x8? A:"
    hf_tokens = hf_greedy_generate(hf, hf_tok, prompt, max_new_tokens=50)
    eng_tokens = engine_greedy_generate(eng, eng_tok, prompt, max_new_tokens=50)
    assert eng_tokens == hf_tokens


def debug_single_step_diff(engine, hf_model, tokenizer, prompt):
    from mini_vllm.engine.sequence import Sequence
    from mini_vllm.scheduler.scheduler import SchedulerOutput

    ids = tokenizer.encode(prompt)
    hf_ids = torch.tensor([ids], device=hf_model.device)

    with torch.inference_mode():
        hf_logits = hf_model(input_ids=hf_ids, use_cache=False).logits[0, -1]

    seq = Sequence(
        seq_id=0,
        prompt_token_ids=ids,
        sampling_params=SamplingParams(temperature=0.0),
        block_size=16,
    )
    sched_out = SchedulerOutput(scheduled_seqs=[seq], num_scheduled_tokens=[len(ids)])
    eng_logits = engine.model_runner.execute(sched_out)[0]

    print(f"HF TOP 5: {hf_logits.topk(5).indices.tolist()}")
    print(f"ENG TOP 5: {eng_logits.topk(5).indices.tolist()}")
    print(f"max abs diff: {(hf_logits.float() - eng_logits.float()).abs().max().item():.4e}")
