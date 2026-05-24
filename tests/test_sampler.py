import torch

from mini_vllm.engine.sequence import Sequence
from mini_vllm.sampling.params import SamplingParams
from mini_vllm.sampling.sampler import Sampler


def make_seq(seq_id, sp, prompt=(1, 2, 3), outputs=()):
    s = Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(prompt),
        sampling_params=sp,
        block_size=16,
    )
    for t in outputs:
        s.append_token(t)
    return s


def test_greedy_picks_argmax():
    s = Sampler()
    logits = torch.tensor([[0.1, 5.0, 0.2, -1.0]])
    seqs = [make_seq(0, SamplingParams(temperature=0.0))]
    assert s.sample(logits, seqs) == [1]


def test_temperature_does_not_affect_greedy_choice():
    s = Sampler()
    logits = torch.tensor([[0.1, 5.0, 0.2, -1.0]])
    seqs = [make_seq(0, SamplingParams(temperature=0.0))]
    assert s.sample(logits, seqs) == [1]


def test_sampling_distribution_matches_softmax():
    s = Sampler()
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=1)
    expected = torch.softmax(logits, dim=-1).squeeze().tolist()

    counts = [0, 0, 0, 0]
    N = 4000
    for _ in range(N):
        seqs = [make_seq(0, sp)]
        tid = s.sample(logits.clone(), seqs)[0]
        counts[tid] += 1
    empirical = [c / N for c in counts]
    for e, p in zip(empirical, expected, strict=True):
        assert abs(e - p) < 0.03, f"expected {p:.3f}, got {e:.3f}"


def test_top_k_keeps_only_top_k_tokens():
    s = Sampler()
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    sp = SamplingParams(temperature=1.0, top_k=2, max_tokens=1)

    survivors = set()
    for _ in range(200):
        seqs = [make_seq(0, sp)]
        survivors.add(s.sample(logits.clone(), seqs)[0])
    assert survivors <= {3, 4}
    assert {3, 4} <= survivors


def test_top_p_always_keeps_most_likely_token():
    s = Sampler()
    torch.manual_seed(0)
    logits = torch.tensor([[0.1, 0.1, 10.0, 0.1]])
    sp = SamplingParams(temperature=1.0, top_p=0.001, max_tokens=1)
    for _ in range(50):
        seqs = [make_seq(0, sp)]
        assert s.sample(logits.clone(), seqs)[0] == 2


def test_repetition_penalty_demotes_seen_tokens():
    s = Sampler()
    logits = torch.tensor([[1.0, 2.0, 5.0, 1.5]])
    sp_no = SamplingParams(temperature=0.0)
    sp_yes = SamplingParams(temperature=0.0, repetition_penalty=4.0)

    seqs_no = [make_seq(0, sp_no, prompt=(2,))]
    seqs_yes = [make_seq(0, sp_yes, prompt=(2,))]

    assert s.sample(logits.clone(), seqs_no) == [2]
    assert s.sample(logits.clone(), seqs_yes) == [1]


def test_mixed_greedy_and_sampling_in_one_batch():
    s = Sampler()
    torch.manual_seed(0)
    logits = torch.tensor([[0.1, 5.0, 0.2, -1.0], [1.0, 2.0, 3.0, 4.0]])
    seqs = [
        make_seq(0, SamplingParams(temperature=0.0)),
        make_seq(1, SamplingParams(temperature=1.0)),
    ]
    out = s.sample(logits, seqs)
    assert out[0] == 1
    assert out[1] in {0, 1, 2, 3}


def test_per_row_top_k_handled_independently():
    s = Sampler()
    torch.manual_seed(42)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    seqs = [
        make_seq(0, SamplingParams(temperature=1.0, top_k=1)),
        make_seq(1, SamplingParams(temperature=1.0, top_k=1)),
    ]
    out = s.sample(logits, seqs)
    assert out == [4, 0]
