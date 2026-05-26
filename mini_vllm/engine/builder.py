from transformers import PreTrainedTokenizerBase

from ..block_manager.manager import BlockManager
from ..config import Config
from ..layers.attention_patch import patch_model
from ..model_runner.loader import ModelLoader
from ..model_runner.runner import ModelRunner
from ..scheduler.scheduler import Scheduler
from .llm_engine import LLMEngine


def build_engine(
    config: Config, num_blocks: int = 512
) -> tuple[LLMEngine, PreTrainedTokenizerBase]:
    loaded = ModelLoader(config).load()
    tokenizer = loaded["tokenizer"]
    model = loaded["model"]
    info = loaded["info"]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    patch_model(model)
    runner = ModelRunner(model=model, config=config, pad_token_id=tokenizer.pad_token_id, info=info)
    runner.init_kv_cache(num_blocks=num_blocks, block_size=config.block_size)
    bm = BlockManager(num_blocks=num_blocks, block_size=config.block_size)
    sched = Scheduler(
        block_manager=bm,
        max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
    )
    eos = config.eos_token_id if config.eos_token_id is not None else tokenizer.eos_token_id
    engine = LLMEngine(
        model_runner=runner,
        block_manager=bm,
        scheduler=sched,
        eos_token_id=eos,
        max_model_len=config.max_model_len,
        tokenizer=tokenizer,
    )
    return engine, tokenizer
