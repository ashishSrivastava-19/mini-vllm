import torch

from ..config import Config
from ..layers.attention_patch import AttentionMetadata, set_metadata
from ..scheduler.scheduler import SchedulerOutput
from .kv_cache import allocate_kv_cache
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

        self.block_size: int | None = None
        self.k_caches: list[torch.Tensor] = []
        self.v_caches: list[torch.Tensor] = []

    def init_kv_cache(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.k_caches, self.v_caches = allocate_kv_cache(
            num_layers = self.num_layers,
            num_blocks = num_blocks,
            block_size = block_size,
            num_kv_heads = self.num_kv_heads,
            head_dim = self.head_dim,
            dtype = self.config.dtype,
            device = torch.device(self.config.device),
        )

    @torch.inference_mode()
    def execute_sdpa(self, sched_out: SchedulerOutput) -> torch.Tensor:
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
    
    @torch.inference_mode()
    def execute(self, sched_out: SchedulerOutput) -> torch.Tensor:
        assert self.block_size is not None, "KV cache not initialized; call init_kv_cache() first"
        seqs = sched_out.scheduled_seqs
        nsched = sched_out.num_scheduled_tokens
        block_size = self.block_size

        token_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        prefill_query_indices: list[int] = []
        decode_query_indices: list[int] = []
        prefill_seq_lens: list[int] = []
        decode_cache_seqlens: list[int] = []
        decode_block_tables: list[list[int]] = []
        last_logit_indices: list[int] = []

        flat = 0

        for seq, n in zip(seqs, nsched, strict = True):
            if seq.output_token_ids:
                assert n == 1, "Only one token should be scheduled for decode step"
                tok = seq.output_token_ids[-1]
                pos = seq.num_tokens - 1
                block = seq.block_table[pos // block_size]
                slots.append(block*block_size + pos % block_size)
                token_ids.append(tok)
                positions.append(pos)
                decode_query_indices.append(flat)
                decode_cache_seqlens.append(seq.num_tokens)
                decode_block_tables.append(list(seq.block_table))
                last_logit_indices.append(flat)
                flat+=1
            else:
                start = seq.num_computed_tokens - n
                end = seq.num_computed_tokens
                if start != 0 or end != seq.num_prompt_tokens:
                    raise NotImplementedError("Only support scheduling the entire prompt as prefill for now")
                for offset in range(n):
                    pos = start + offset
                    block = seq.block_table[pos // block_size]
                    slots.append(block*block_size + pos % block_size)
                    token_ids.append(seq.prompt_token_ids[pos])
                    positions.append(pos)
                    prefill_query_indices.append(flat)
                    flat+=1
                prefill_seq_lens.append(n)
                last_logit_indices.append(flat - 1)
        
        device = self.config.device
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = torch.tensor(positions, dtype=torch.long, device=device).unsqueeze(0)
        slot_mapping = torch.tensor(slots, dtype=torch.long, device=device)
        prefill_query_indices_t = torch.tensor(prefill_query_indices, dtype=torch.long, device=device)
        decode_query_indices_t = torch.tensor(decode_query_indices, dtype=torch.long, device=device)

        has_prefill = len(prefill_query_indices) > 0
        has_decode = len(decode_query_indices) > 0

        if has_prefill:
            prefill_lens = torch.tensor(prefill_seq_lens, dtype=torch.int32, device=device)
            cu_prefill = torch.zeros(len(prefill_seq_lens) + 1, dtype=torch.int32, device=device)
            cu_prefill[1:] = torch.cumsum(prefill_lens, dim=0)
            max_prefill_len = int(max(prefill_seq_lens))
        else:
            cu_prefill = torch.empty(0, dtype=torch.int32, device=device)
            max_prefill_len = 0

        if has_decode:
            max_blocks = max(len(bt) for bt in decode_block_tables)
            block_table = torch.zeros((len(decode_block_tables), max_blocks), dtype=torch.int32, device=device)
            for i, bt in enumerate(decode_block_tables):
                block_table[i, : len(bt)] = torch.tensor(bt, dtype=torch.int32, device=device)
            cache_seqlens = torch.tensor(decode_cache_seqlens, dtype=torch.int32, device=device)
        else:
            block_table = torch.empty((0, 0), dtype=torch.int32, device=device)
            cache_seqlens = torch.empty(0, dtype=torch.int32, device=device)
        
        metadata = AttentionMetadata(
            k_caches=self.k_caches,
            v_caches=self.v_caches,
            slot_mapping=slot_mapping,
            prefill_query_indices=prefill_query_indices_t,
            decode_query_indices=decode_query_indices_t,
            has_prefill=has_prefill,
            cu_seqlens_prefill=cu_prefill,
            max_seqlen_prefill=max_prefill_len,
            has_decode=has_decode,
            decode_cache_seqlens=cache_seqlens,
            decode_block_table=block_table,
        )
        set_metadata(metadata)

        try:
            out = self.model(
                input_ids = input_ids,
                position_ids = position_ids,
                use_cache = False,
                return_dict = True,
            )
        finally:
            set_metadata(None)
        
        last_t = torch.tensor(last_logit_indices, dtype=torch.long, device=device)
        return out.logits[0, last_t, :]

