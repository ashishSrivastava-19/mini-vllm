from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from .attention import mini_attention

@dataclass
class AttentionMetadata:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int

_current_metadata: Optional[AttentionMetadata] = None

def set_metadata(m: AttentionMetadata | None) -> None:
    global _current_metadata
    _current_metadata = m

def get_metadata() -> Optional[AttentionMetadata]:
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

        attn = mini_attention(
            q = q,
            k = k,
            v = v,
            cu_seqlens_q = metadata.cu_seqlens_q,
            cu_seqlens_k = metadata.cu_seqlens_k,
            max_seqlen_q = metadata.max_seqlen_q,
            max_seqlen_k = metadata.max_seqlen_k,
            softmax_scale = self.scaling,
            causal = True,
        )  # [total_tokens, num_heads, head_dim]
        attn = attn.reshape(B, T, num_heads * head_dim)
        out = self.o_proj(attn)
        return out, None

    return forward

def patch_model(model: nn.Module) -> None:
    forward = _patched_forward_factory()
    patched = 0
    for module in model.modules():
        if type(module).__name__ in {"Qwen2Attention", "LlamaAttention", "MistralAttention"}:
            module.forward = forward.__get__(module, type(module))
            patched += 1
    assert patched > 0, "No attention layers found"
