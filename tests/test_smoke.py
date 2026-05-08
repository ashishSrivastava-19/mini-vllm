import mini_vllm


def test_version():
    assert mini_vllm.__version__ == "0.1.0"


def test_subpackages_importable():
    from mini_vllm import (  # noqa: F401, PLC0415
        block_manager,
        engine,
        layers,
        model_runner,
        sampling,
        scheduler,
        utils,
    )
