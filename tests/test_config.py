import pytest
import torch

from mini_vllm import Config


def test_defaults_validate():
    cfg = Config(model_path="dummy").validate()
    assert cfg.dtype == torch.float16
    assert cfg.block_size == 16
    assert cfg.max_num_seqs == 16
    assert cfg.max_num_batched_tokens == 2048
    assert cfg.num_gpu_blocks is None


def test_validate_returns_self_for_chaining():
    cfg = Config(model_path="dummy")
    assert cfg.validate() is cfg


def test_block_size_must_be_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        Config(model_path="dummy", block_size=15).validate()


def test_block_size_must_be_positive():
    with pytest.raises(ValueError, match="power of 2"):
        Config(model_path="dummy", block_size=0).validate()


def test_batched_tokens_must_be_at_least_block_size():
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        Config(model_path="dummy", block_size=32, max_num_batched_tokens=16).validate()


def test_max_model_len_must_be_positive():
    with pytest.raises(ValueError, match="max_model_len"):
        Config(model_path="dummy", max_model_len=0).validate()


def test_gpu_memory_utilization_upper_bound():
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        Config(model_path="dummy", gpu_memory_utilization=1.5).validate()


def test_gpu_memory_utilization_lower_bound():
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        Config(model_path="dummy", gpu_memory_utilization=0.0).validate()


def test_max_num_seqs_must_be_positive():
    with pytest.raises(ValueError, match="max_num_seqs"):
        Config(model_path="dummy", max_num_seqs=0).validate()
