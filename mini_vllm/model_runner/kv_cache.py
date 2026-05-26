import torch

def allocate_kv_cache(
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    shape = (num_blocks, block_size, num_kv_heads, head_dim)
    k_caches = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]
    v_caches = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]
    return k_caches, v_caches