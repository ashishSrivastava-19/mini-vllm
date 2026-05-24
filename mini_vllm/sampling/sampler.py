import torch

from ..engine.sequence import Sequence


class Sampler:
    """
    Sampler converts logits to token ids. It also updates the sequence status and returns the finished sequences.
    """

    @torch.inference_mode()
    def sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        assert logits.dim() == 2
        assert logits.size(0) == len(seqs), (
            f"logits batch size {logits.size(0)} does not match number of sequences {len(seqs)}"
        )

        logits = logits.float().clone()
        logits = self._apply_repetition_penalty(logits, seqs)
        is_greedy_mask = torch.tensor(
            [s.sampling_params.is_greedy for s in seqs], device=logits.device, dtype=torch.bool
        )
        temps = torch.tensor(
            [1.0 if s.sampling_params.is_greedy else s.sampling_params.temperature for s in seqs],
            device=logits.device,
            dtype=torch.float32,
        ).unsqueeze(1)
        logits = logits / temps
        logits = self._apply_top_k(logits, seqs)
        logits = self._apply_top_p(logits, seqs)

        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
        greedy = torch.argmax(logits, dim=-1)
        out = torch.where(is_greedy_mask, greedy, sampled)
        return out.tolist()

    @staticmethod
    def _apply_repetition_penalty(logits, seqs):
        if all(s.sampling_params.repetition_penalty == 1.0 for s in seqs):
            return logits
        for i, seq in enumerate(seqs):
            p = seq.sampling_params.repetition_penalty
            if p == 1.0:
                continue
            seen = set(seq.prompt_token_ids) | set(seq.output_token_ids)
            if not seen:
                continue
            seen_idx = torch.tensor(list(seen), device=logits.device, dtype=torch.long)
            row = logits[i, seen_idx]
            row = torch.where(row > 0, row / p, row * p)
            logits[i, seen_idx] = row
        return logits

    @staticmethod
    def _apply_top_k(logits, seqs):
        ks = [s.sampling_params.top_k for s in seqs]
        if all(k == -1 for k in ks):
            return logits
        V = logits.size(-1)
        max_k = max(k if k > 0 else V for k in ks)
        topk_vals, _ = torch.topk(logits, k=max_k, dim=-1)
        for i, k in enumerate(ks):
            if k == -1:
                continue
            threshold = topk_vals[i, k - 1]
            logits[i] = torch.where(
                logits[i] < threshold, torch.full_like(logits[i], float("-inf")), logits[i]
            )
        return logits

    @staticmethod
    def _apply_top_p(logits, seqs):
        ps = [s.sampling_params.top_p for s in seqs]
        if all(p == 1.0 for p in ps):
            return logits
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        ps_t = torch.tensor(ps, device=logits.device, dtype=torch.float32).unsqueeze(1)
        drop = cumprobs > ps_t
        drop[:, 1:] = drop[:, :-1].clone()
        drop[:, 0] = False
        drop_mask = torch.zeros_like(drop)
        drop_mask.scatter_(1, sorted_idx, drop)
        return logits.masked_fill(drop_mask, float("-inf"))
