from dataclasses import dataclass, field


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass
class RequestOutput:
    request_id: int
    prompt: str
    prompt_token_ids: list[int] = field(default_factory=list)
    outputs: list[CompletionOutput] = field(default_factory=list)
    finished: bool = False
