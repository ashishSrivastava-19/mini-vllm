import pytest
import torch

from mini_vllm import Config, ModelInfo, ModelLoader

# --- Pure-CPU tests (run by default) ---


def test_model_info_is_frozen():
    info = ModelInfo(
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        hidden_size=32,
        vocab_size=100,
        max_position=128,
    )
    with pytest.raises((AttributeError, Exception)):
        info.num_layers = 99  # type: ignore[misc]


def test_kv_bytes_per_block_qwen_like_fp16():
    # Qwen2.5-0.5B-ish: 24 layers, 2 KV heads, head_dim 64
    info = ModelInfo(
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=896,
        vocab_size=151936,
        max_position=32768,
    )
    # 2 (K and V) * 24 * 2 * 64 * 16 (block_size) * 2 (fp16 itemsize)
    assert info.kv_bytes_per_block(block_size=16, dtype=torch.float16) == 196_608


def test_kv_bytes_per_block_scales_with_block_size():
    info = ModelInfo(
        num_layers=4,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        hidden_size=512,
        vocab_size=1000,
        max_position=2048,
    )
    one = info.kv_bytes_per_block(block_size=1, dtype=torch.float16)
    sixteen = info.kv_bytes_per_block(block_size=16, dtype=torch.float16)
    assert sixteen == one * 16


def test_kv_bytes_per_block_scales_with_dtype():
    info = ModelInfo(
        num_layers=4,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        hidden_size=512,
        vocab_size=1000,
        max_position=2048,
    )
    fp16 = info.kv_bytes_per_block(block_size=16, dtype=torch.float16)
    fp32 = info.kv_bytes_per_block(block_size=16, dtype=torch.float32)
    assert fp32 == fp16 * 2


def test_extract_info_uses_num_key_value_heads_when_present():
    """GQA models (Qwen2.5, Llama3) report a smaller num_key_value_heads."""

    class FakeHFConfig:
        num_hidden_layers = 24
        num_attention_heads = 14
        num_key_value_heads = 2
        hidden_size = 896
        vocab_size = 151936
        max_position_embeddings = 32768

    info = ModelLoader._extract_info(FakeHFConfig())
    assert info.num_heads == 14
    assert info.num_kv_heads == 2
    assert info.head_dim == 896 // 14  # falls back to hidden_size // num_heads


def test_extract_info_falls_back_when_kv_heads_absent():
    """Pure-MHA models (older GPT2-style) lack num_key_value_heads."""

    class FakeHFConfig:
        num_hidden_layers = 12
        num_attention_heads = 12
        hidden_size = 768
        vocab_size = 50257
        max_position_embeddings = 1024

    info = ModelLoader._extract_info(FakeHFConfig())
    assert info.num_kv_heads == info.num_heads == 12


def test_extract_info_prefers_explicit_head_dim_when_set():
    """Newer Llama/Qwen variants set head_dim independently of hidden_size."""

    class FakeHFConfig:
        num_hidden_layers = 32
        num_attention_heads = 32
        num_key_value_heads = 8
        head_dim = 128  # not equal to hidden_size // num_attention_heads
        hidden_size = 4096
        vocab_size = 128256
        max_position_embeddings = 8192

    info = ModelLoader._extract_info(FakeHFConfig())
    assert info.head_dim == 128


# --- GPU integration test (opt-in) ---


@pytest.mark.gpu
@pytest.mark.slow
def test_load_qwen_smoke():
    """End-to-end: download Qwen2.5-0.5B, check shapes, run one forward pass."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = Config(model_path="Qwen/Qwen2.5-0.5B").validate()
    result = ModelLoader(cfg).load()
    model, tok, info = result["model"], result["tokenizer"], result["info"]

    assert info.num_layers > 0
    assert info.num_kv_heads <= info.num_heads
    assert info.vocab_size > 0

    ids = tok("hello world", return_tensors="pt").input_ids.to(cfg.device)
    with torch.no_grad():
        out = model(ids)
    assert out.logits.shape[0] == 1
    assert out.logits.shape[-1] == info.vocab_size
