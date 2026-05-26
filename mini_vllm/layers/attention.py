import torch
from flash_attn import flash_attn_varlen_func

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