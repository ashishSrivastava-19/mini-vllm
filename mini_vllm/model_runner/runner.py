import torch

from ..config import Config
from ..scheduler.scheduler import SchedulerOutput
from .loader import ModelInfo


class ModelRunner:
    """
    v0 model runner with HF model, left padded forward pass and no KV cache
    """

    def __init__(
        self,
        model,
        config: Config,
        pad_token_id: int,
        info: ModelInfo | None = None,
    ):
        self.model = model
        self.config = config
        self.pad_token_id = pad_token_id
        self.info = info

        self.model.to(device=self.config.device, dtype=self.config.dtype)
        self.model.eval()

        if info is not None:
            self.vocab_size = info.vocab_size
            self.num_layers = info.num_layers
            self.num_heads = info.num_heads
            self.num_kv_heads = info.num_kv_heads
            self.head_dim = info.head_dim
        else:
            cfg = self.model.config
            self.vocab_size = cfg.vocab_size
            self.num_layers = cfg.num_hidden_layers
            self.num_heads = cfg.num_attention_heads
            self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
            self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    @torch.inference_mode()
    def execute(self, sched_out: SchedulerOutput) -> torch.Tensor:
        seqs = sched_out.scheduled_seqs
        B = len(seqs)
        token_lists = [s.all_token_ids for s in seqs]
        seq_lens = [len(t) for t in token_lists]
        max_len = max(seq_lens)

        input_ids = torch.full(
            (B, max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.config.device,
        )
        attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=self.config.device)

        for i, (tokens, n) in enumerate(zip(token_lists, seq_lens, strict=True)):
            input_ids[i, max_len - n :] = torch.tensor(
                tokens, dtype=torch.long, device=self.config.device
            )
            attn_mask[i, max_len - n :] = 1

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits[:, -1, :]
