from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 16
    stop_token_ids: list[int] = field(default_factory=list)
    repetition_penalty: float = 1.0

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be non-negative"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.top_k == -1 or self.top_k > 0, "top_k must be -1 or positive"
        assert self.max_tokens > 0, "max_tokens must be > 0"
        assert self.repetition_penalty > 0.0, "repetition_penalty must be > 0"

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0
