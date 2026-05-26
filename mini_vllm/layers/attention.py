import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

def mini_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        causal: bool = True,
) -> torch.Tensor:
    return flash_attn_varlen_func(
        q = q, 
        k = k,
        v = v,
        cu_seqlens_q = cu_seqlens_q,
        cu_seqlens_k = cu_seqlens_k,
        max_seqlen_q = max_seqlen_q,
        max_seqlen_k = max_seqlen_k,
        softmax_scale = softmax_scale,
        causal = causal,
    )

def mini_decode_attention(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        softmax_scale: float | None = None,
) -> torch.Tensor:
    out = flash_attn_with_kvcache(
        q = q.unsqueeze(1),
        k_cache = k_cache,
        v_cache = v_cache,
        cache_seqlens = cache_seqlens,
        block_table = block_table,
        softmax_scale = softmax_scale,
        causal = True,
    )
    return out.squeeze(1)

def write_kv_to_cache(
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
) -> None:
    num_blocks, block_size, n_kv, d = k_cache.shape
    k_flat = k_cache.view(num_blocks * block_size, n_kv, d)
    v_flat = v_cache.view(num_blocks * block_size, n_kv, d)
    idx = slot_mapping.long()
    k_flat[idx] = k_new
    v_flat[idx] = v_new