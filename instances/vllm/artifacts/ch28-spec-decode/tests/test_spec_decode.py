"""复现真实 vLLM v1 投机解码可观察行为的测试（pin f3fef123）。

测试对象是精简版，但断言锚定在 dossier 记录的真实 vLLM 行为：
  - calc_spec_decode_metadata 的三组 index 与 gpu_model_runner.py:L2601-L2664
    注释里的具体数字逐位一致；
  - ngram proposer 的 KMP/LPS 草稿生成；
  - rejection_sample 两个 triton kernel 的接受/拒绝/bonus 语义与分布等价性。

triton/CUDA 测试需要 GPU；无 GPU 时自动 skip（不掩盖纯 CPU 测试）。
"""

import numpy as np
import pytest
import torch

from metadata import SpecDecodeMetadata, calc_spec_decode_metadata
from ngram_proposer import (
    NgramProposer,
    _find_longest_matched_ngram_and_propose_tokens,
)
from rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    RejectionSampler,
    generate_uniform_probs,
    rejection_sample,
)
from sampling_metadata import SamplingMetadata

CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="需要 CUDA + triton")


# ---------------------------------------------------------------------------
# 1. index 间接：calc_spec_decode_metadata 复现 gpu_model_runner 注释里的数字
# ---------------------------------------------------------------------------
def test_calc_spec_decode_metadata_matches_source_comment():
    # 来自 vllm/v1/worker/gpu_model_runner.py:L2601-L2609 的具体示例。
    num_draft_tokens = np.array([3, 0, 2, 0, 1], dtype=np.int32)
    cu_num_scheduled_tokens = np.array([4, 104, 107, 207, 209], dtype=np.int32)
    # 草稿 token 实际被填进 input 流、错位一格取出，这里造一个足够长的 input_ids，
    # 使 draft_token_ids == logits_indices+1 处的值 = 该位置本身（便于断言）。
    input_ids = torch.arange(300, dtype=torch.int32)

    md = calc_spec_decode_metadata(
        num_draft_tokens, cu_num_scheduled_tokens, input_ids
    )

    assert md.cu_num_draft_tokens.tolist() == [3, 3, 5, 5, 6]
    assert md.cu_num_sampled_tokens.tolist() == [4, 5, 8, 9, 11]
    assert md.logits_indices.tolist() == [
        0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208,
    ]
    assert md.target_logits_indices.tolist() == [0, 1, 2, 5, 6, 9]
    assert md.bonus_logits_indices.tolist() == [3, 4, 7, 8, 10]
    assert md.max_spec_len == 3
    # 草稿数为 0 的请求(req1,req3)在 target_logits_indices 里没有任何条目，
    # 但在 bonus_logits_indices 里各占一格（只有 bonus 位）。
    # draft_token_indices = input[logits_indices][target_logits_indices+1]
    #                     = [1, 2, 3, 105, 106, 208]
    assert md.draft_token_ids.tolist() == [1, 2, 3, 105, 106, 208]


def test_metadata_post_init_and_make_dummy():
    md = SpecDecodeMetadata.make_dummy([[10, 11, 12], [], [20]], torch.device("cpu"))
    assert md.num_draft_tokens == [3, 0, 1]
    assert md.max_spec_len == 3
    assert md.cu_num_draft_tokens.tolist() == [3, 3, 4]
    assert md.cu_num_sampled_tokens.tolist() == [4, 5, 7]
    assert md.draft_token_ids.tolist() == [10, 11, 12, 20]


# ---------------------------------------------------------------------------
# 2. ngram proposer：KMP/LPS 最长匹配后缀 + 复制后续 k token
# ---------------------------------------------------------------------------
def test_ngram_find_longest_matched():
    # 序列 ... 1 2 3 [4 5] ... 1 2 3 ；后缀 (1,2,3) 在前面出现过，其后是 4,5。
    tokens = np.array([1, 2, 3, 4, 5, 9, 1, 2, 3], dtype=np.int32)
    out = _find_longest_matched_ngram_and_propose_tokens(
        tokens, min_ngram=2, max_ngram=3, max_model_len=100, k=2
    )
    assert out.tolist() == [4, 5]


def test_ngram_no_match_returns_empty():
    tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    out = _find_longest_matched_ngram_and_propose_tokens(
        tokens, min_ngram=2, max_ngram=3, max_model_len=100, k=3
    )
    assert out.shape[0] == 0


def test_ngram_proposer_batch():
    prop = NgramProposer(
        prompt_lookup_min=2,
        prompt_lookup_max=3,
        num_speculative_tokens=2,
        max_model_len=100,
    )
    seqs = [[1, 2, 3, 4, 5, 9, 1, 2, 3], [7, 7, 7]]
    max_len = max(len(s) for s in seqs)
    token_ids_cpu = np.zeros((2, max_len), dtype=np.int32)
    num_tokens = np.zeros(2, dtype=np.int32)
    for i, s in enumerate(seqs):
        token_ids_cpu[i, : len(s)] = s
        num_tokens[i] = len(s)
    # sampled_token_ids 非空表示该请求参与投机。
    drafts = prop.propose([[0], [0]], num_tokens, token_ids_cpu)
    assert drafts[0] == [4, 5]
    # 第二请求 (7,7,7)：后缀 (7,7) 在前面出现、其后是 7 -> 草稿 [7]（仅 1 个）。
    assert drafts[1] == [7]


# ---------------------------------------------------------------------------
# 3. generate_uniform_probs：float64、零草稿请求不消耗随机数
# ---------------------------------------------------------------------------
def test_generate_uniform_probs_dtype_and_range():
    u = generate_uniform_probs(5, [3, 0, 2], {}, torch.device("cpu"))
    assert u.dtype == torch.float64
    assert u.shape == (5,)
    assert (u >= 0).all() and (u < 1).all()


# ---------------------------------------------------------------------------
# 4. rejection_sample greedy 路径（triton + CUDA）
# ---------------------------------------------------------------------------
def _greedy_meta(temperature):
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
    )


@gpu
def test_greedy_all_accept_appends_bonus():
    device = torch.device("cuda")
    vocab = 8
    # 单请求，2 个草稿，目标 argmax 恰好等于草稿 -> 全接受 + bonus。
    draft = torch.tensor([5, 3], dtype=torch.int32, device=device)
    num_draft = [2]
    cu = torch.tensor([2], dtype=torch.int32, device=device)
    target_logits = torch.full((2, vocab), -10.0, device=device)
    target_logits[0, 5] = 10.0  # argmax=5 == draft[0]
    target_logits[1, 3] = 10.0  # argmax=3 == draft[1]
    bonus = torch.tensor([[7]], dtype=torch.int32, device=device)
    out = rejection_sample(
        draft, num_draft, 2, cu, None, target_logits, bonus,
        _greedy_meta(torch.zeros(1, device=device)),
    )
    # [5, 3, 7]  (两个 accept + bonus)
    assert out[0, :3].tolist() == [5, 3, 7]


@gpu
def test_greedy_reject_truncates():
    device = torch.device("cuda")
    vocab = 8
    draft = torch.tensor([5, 3], dtype=torch.int32, device=device)
    cu = torch.tensor([2], dtype=torch.int32, device=device)
    target_logits = torch.full((2, vocab), -10.0, device=device)
    target_logits[0, 5] = 10.0  # accept draft[0]=5
    target_logits[1, 6] = 10.0  # argmax=6 != draft[1]=3 -> reject, output 6, stop
    bonus = torch.tensor([[7]], dtype=torch.int32, device=device)
    out = rejection_sample(
        draft, [2], 2, cu, None, target_logits, bonus,
        _greedy_meta(torch.zeros(1, device=device)),
    )
    assert out[0, 0].item() == 5
    assert out[0, 1].item() == 6  # recovered = greedy target token
    assert out[0, 2].item() == PLACEHOLDER_TOKEN_ID  # truncated, no bonus


@gpu
def test_parse_output_filters_placeholder():
    device = torch.device("cuda")
    out = torch.tensor(
        [[5, 6, PLACEHOLDER_TOKEN_ID], [5, 3, 7]],
        dtype=torch.int32, device=device,
    )
    parsed = RejectionSampler.parse_output(out, vocab_size=8)
    assert parsed == [[5, 6], [5, 3, 7]]


# ---------------------------------------------------------------------------
# 5. rejection_sample random 路径分布等价性（NO_DRAFT_PROBS / ngram）
#    草稿 token x 以 min(1, p_target(x)) 概率被接受；统计接受率 ≈ p_target(x)。
# ---------------------------------------------------------------------------
@gpu
def test_random_acceptance_rate_matches_target_prob():
    device = torch.device("cuda")
    vocab = 4
    N = 20000
    # 每个 trial 一个请求、1 个草稿，草稿 token = 0。
    draft = torch.zeros(N, dtype=torch.int32, device=device)
    num_draft = [1] * N
    cu = torch.arange(1, N + 1, dtype=torch.int32, device=device)
    # 目标分布对 token 0 的概率约 0.5。构造 logits 使 softmax[...,0]=0.5。
    target_logits = torch.zeros((N, vocab), device=device)
    # logits = [a, b, b, b] with softmax[0] = e^a/(e^a+3e^b). 取 a=ln3+b -> 0.5
    target_logits[:, 0] = np.log(3.0)
    bonus = torch.zeros((N, 1), dtype=torch.int32, device=device)
    meta = SamplingMetadata(
        temperature=torch.ones(N, device=device),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
    )
    out = rejection_sample(
        draft, num_draft, 1, cu, None, target_logits, bonus, meta,
    )
    # 接受 => out[:,0] == draft(0)；拒绝 => recovered(!=0, 从残差采)。
    accepted = (out[:, 0] == 0).float().mean().item()
    assert abs(accepted - 0.5) < 0.03, accepted
