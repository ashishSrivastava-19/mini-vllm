import torch

from ..scheduler.scheduler import SchedulerOutput


class StubModelRunner:
    """
    Random logits in [B, V] where B is the batch size and V is the vocab size.
    One row per scheduled seq, regardless of prefill vs decode step.
    """

    def __init__(self, vocab_size: int, device: str = "cpu", seed: int = 0):
        self.vocab_size = vocab_size
        self.device = device
        self._gen = torch.Generator(device=device).manual_seed(seed)

    def execute(self, sched_out: SchedulerOutput) -> torch.Tensor:
        B = len(sched_out.scheduled_seqs)
        return torch.randn(B, self.vocab_size, generator=self._gen, device=self.device)
