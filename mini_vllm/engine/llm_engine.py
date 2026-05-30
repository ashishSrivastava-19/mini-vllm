from ..block_manager.manager import BlockManager
from ..sampling.params import SamplingParams
from ..sampling.sampler import Sampler
from ..scheduler.scheduler import Scheduler
from .output import CompletionOutput, RequestOutput
from .sequence import Sequence, SequenceStatus


class LLMEngine:
    def __init__(
        self,
        model_runner,
        block_manager: BlockManager,
        scheduler: Scheduler,
        eos_token_id: int,
        max_model_len: int = 2048,
        tokenizer=None,
    ):
        self.model_runner = model_runner
        self.block_manager = block_manager
        self.scheduler = scheduler
        self.eos_token_id = eos_token_id
        self.max_model_len = max_model_len
        self.tokenizer = tokenizer
        self.sampler = Sampler()
        self._next_seq_id = 0
        self._all_seqs: dict[int, Sequence] = {}
        self._prompt_text: dict[int, str] = {}  # seq_id -> prompt text

    def add_request(
        self, prompt: str, prompt_token_ids: list[int], sampling_params: SamplingParams
    ) -> int:
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        seq = Sequence(
            seq_id=seq_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            block_size=self.block_manager.block_size,
        )
        self.scheduler.add_request(seq)
        self._all_seqs[seq_id] = seq
        self._prompt_text[seq_id] = prompt
        return seq_id

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests()

    def step(self) -> list[RequestOutput]:
        sched_out = self.scheduler.schedule()
        if sched_out.is_empty:
            return []

        logits = self.model_runner.execute(sched_out)
        next_token_ids = self.sampler.sample(logits, sched_out.scheduled_seqs)

        bm = self.block_manager
        finished_now: list[RequestOutput] = []
        for seq, new_token_id in zip(sched_out.scheduled_seqs, next_token_ids, strict=True):
            if not seq.is_prefill_complete():
                for idx in range(seq.num_computed_tokens // bm.block_size):
                    bm.maybe_register_full_block(seq, idx)
                continue
            tok = int(new_token_id)
            seq.append_token(tok)

            n_kv = seq.num_tokens - 1
            for idx in range(n_kv // bm.block_size):
                bm.maybe_register_full_block(seq, idx)

            sp = seq.sampling_params

            finish_reason: str | None = None
            if tok == self.eos_token_id or tok in sp.stop_token_ids:
                finish_reason = "stop"
            elif len(seq.output_token_ids) >= sp.max_tokens:
                finish_reason = "length"
            elif seq.num_tokens >= self.max_model_len:
                finish_reason = "max_model_len"

            if finish_reason is not None:
                seq.status = SequenceStatus.FINISHED
                finished_now.append(self._make_output(seq, finish_reason))

        self.scheduler.free_finished()
        return finished_now

    def generate(
        self, prompts: list[str], prompt_token_ids: list[list[int]], sampling_params
    ) -> list[RequestOutput]:
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        assert len(prompts) == len(prompt_token_ids) == len(sampling_params)

        ids = [
            self.add_request(p, t, sp)
            for p, t, sp in zip(prompts, prompt_token_ids, sampling_params, strict=True)
        ]
        outputs_by_id: dict[int, RequestOutput] = {}
        while self.has_unfinished_requests():
            for finished in self.step():
                outputs_by_id[finished.request_id] = finished
        return [outputs_by_id[i] for i in ids]

    def _make_output(self, seq: Sequence, finish_reason: str) -> RequestOutput:
        if self.tokenizer is not None:
            text = self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
        else:
            text = ""
        return RequestOutput(
            request_id=seq.seq_id,
            prompt=self._prompt_text[seq.seq_id],
            prompt_token_ids=seq.prompt_token_ids,
            outputs=[
                CompletionOutput(
                    index=0, text=text, token_ids=seq.output_token_ids, finish_reason=finish_reason
                )
            ],
            finished=True,
        )
