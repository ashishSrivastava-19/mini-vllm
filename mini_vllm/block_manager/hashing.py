import xxhash

ROOT_HASH = 0


def hash_block(parent_hash: int, token_ids: tuple[int, ...]) -> int:
    assert isinstance(token_ids, tuple), f"token_ids must be a tuple, but got {type(token_ids)}"
    h = xxhash.xxh64()
    h.update(parent_hash.to_bytes(8, "little", signed=False))
    for token_id in token_ids:
        h.update(token_id.to_bytes(4, "little", signed=False))
    return h.intdigest()


def hash_blocks_for_tokens(token_ids: list[int], block_size: int) -> list[int]:
    n_full = len(token_ids) // block_size
    hashes: list[int] = []
    parent_hash = ROOT_HASH
    for i in range(n_full):
        start = i * block_size
        block_token_ids = tuple(token_ids[start : start + block_size])
        block_hash = hash_block(parent_hash, block_token_ids)
        hashes.append(block_hash)
        parent_hash = block_hash
    return hashes
