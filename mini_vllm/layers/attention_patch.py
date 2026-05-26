from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from .attention import mini_attention, mini_decode_attention, write_kv_to_cache

@dataclass
class AttentionMetadata:
    k_caches: list[torch.Tensor]
    v_caches: list[torch.Tensor]

    slot_mapping: torch.Tensor

    prefill_query_indices: torch.Tensor
    decode_query_indices: torch.Tensor

    has_prefill: bool
    cu_seqlens_prefill: torch.Tensor
    max_seqlen_prefill: int

    has_decode: bool
    decode_cache_seqlens: torch.Tensor
    decode_block_table: torch.Tensor


_current_metadata: Optional[AttentionMetadata] = None

def set_metadata(m: AttentionMetadata | None) -> None:
    global _current_metadata
    _current_metadata = m

def get_metadata() -> AttentionMetadata:
    assert _current_metadata is not None, "Attention metadata is not set"
    return _current_metadata

def _patched_forward_factory():
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask = None, past_key_values = None, **kwargs):
        metadata = get_metadata()
        cfg = self.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        head_dim = self.head_dim
        layer_idx = self.layer_idx

        B, T, _ = hidden_states.shape  # B == 1
        # [B, T, H, D] -> [B, H, T, D] so apply_rotary_pos_emb (unsqueeze_dim=1) broadcasts.
        q = self.q_proj(hidden_states).view(B, T, num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, num_kv_heads, head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Flatten to [total_tokens, H, D] for flash_attn_varlen_func.
        q = q.transpose(1, 2).reshape(B * T, num_heads, head_dim).contiguous()
        k = k.transpose(1, 2).reshape(B * T, num_kv_heads, head_dim).contiguous()
        v = v.transpose(1, 2).reshape(B * T, num_kv_heads, head_dim).contiguous()

        k_cache = metadata.k_caches[layer_idx]
        v_cache = metadata.v_caches[layer_idx]

        write_kv_to_cache(k, v, k_cache, v_cache, metadata.slot_mapping)

        attn_out = torch.empty_like(q)

        if metadata.has_prefill:
            pidx = metadata.prefill_query_indices
            attn_out[pidx] = mini_attention(
                q = q[pidx],
                k = k[pidx],
                v = v[pidx],
                cu_seqlens_q = metadata.cu_seqlens_prefill,
                cu_seqlens_k = metadata.cu_seqlens_prefill,
                max_seqlen_q = metadata.max_seqlen_prefill,
                max_seqlen_k = metadata.max_seqlen_prefill,
                softmax_scale = self.scaling,
                causal = True,
            )
        if metadata.has_decode:
            didx = metadata.decode_query_indices
            attn_out[didx] = mini_decode_attention(
                q = q[didx],
                k_cache = k_cache,
                v_cache = v_cache,
                cache_seqlens = metadata.decode_cache_seqlens,
                block_table = metadata.decode_block_table,
                softmax_scale = self.scaling,
            )
        attn_out = attn_out.reshape(B, T, num_heads*head_dim)
        return self.o_proj(attn_out), None
    return forward

def patch_model(model: nn.Module) -> None:
    forward = _patched_forward_factory()
    patched = 0
    for module in model.modules():
        if type(module).__name__ in {"Qwen2Attention", "LlamaAttention", "MistralAttention"}:
            module.forward = forward.__get__(module, type(module))
            patched += 1
    assert patched > 0, "No attention layers found"
