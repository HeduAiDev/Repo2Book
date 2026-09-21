# NgramProposer 零模型 drafter（vllm/v1/spec_decode/ngram_proposer.py:L12-L293）。
# 行为基准 = 真实 v0.27.1 的 KMP/LPS 实现：
#   - 翻转序列后找『匹配后缀的最长 n-gram（长度∈[min,max]）』（L230-L233 注释）
#   - 取翻转后最靠右的匹配 = 原序列最早出现（L256-L263 注释 position 更新条件）
#   - 从匹配 n-gram 结束处复制 k 个 token 当草稿（L291-L293）
#   - 无匹配/上下文短于 min_ngram/k 被截到 0 → 空草稿（L220-L228、L283-L285）
import numpy as np
import torch

from vllm.config import (
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.spec_decode.ngram_proposer import (
    NgramProposer,
    _find_longest_matched_ngram_and_propose_tokens,
)


def make_vllm_config(min_n=1, max_n=4, k=3, max_model_len=64, max_num_seqs=8):
    return VllmConfig(
        model_config=ModelConfig(max_model_len=max_model_len),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs),
        parallel_config=ParallelConfig(),
        speculative_config=SpeculativeConfig(
            prompt_lookup_min=min_n,
            prompt_lookup_max=max_n,
            num_speculative_tokens=k,
        ),
    )


def test_longest_match_and_copy_followers():
    # 尾部 [1,2,3,4] 在位置 0 处完整出现（匹配的是序列后缀）→ 从位置 4 复制 3 个 token
    tokens = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32)
    out = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens, min_ngram=2, max_ngram=4, max_model_len=64, k=3
    )
    assert out.tolist() == [1, 2, 3]


def test_earliest_occurrence_wins():
    # [1,2] 在位置 0/3/5 三处出现，后缀匹配取原序列最早（翻转后最靠右）——
    # 真实 L256-L263 注释：`position` 在并列时更新为更大 i（翻转域）= 原域最早。
    tokens = np.array([1, 2, 9, 1, 2, 8, 1, 2], dtype=np.int32)
    out = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens, min_ngram=2, max_ngram=3, max_model_len=64, k=2
    )
    assert out.tolist() == [9, 1]


def test_no_match_returns_empty():
    tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    out = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens, min_ngram=2, max_ngram=3, max_model_len=64, k=3
    )
    assert out.size == 0


def test_context_shorter_than_min_ngram():
    tokens = np.array([7, 7], dtype=np.int32)
    out = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens, min_ngram=3, max_ngram=4, max_model_len=64, k=2
    )
    assert out.size == 0


def test_k_capped_by_max_model_len():
    # total_token=8, max_model_len=10 → k=min(3, 2)=2
    tokens = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32)
    out = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens, min_ngram=2, max_ngram=4, max_model_len=10, k=3
    )
    assert out.tolist() == [1, 2]


def test_lps_capped_at_max_ngram():
    # 全同 token：匹配长度被 max_ngram 截住（lps 只存前 max_ngram 个前缀，L239-L241）
    tokens = np.array([7] * 10, dtype=np.int32)
    out = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens, min_ngram=2, max_ngram=3, max_model_len=64, k=2
    )
    assert out.tolist() == [7, 7]


def test_propose_end_to_end_skips_invalid_requests():
    cfg = make_vllm_config(min_n=1, max_n=4, k=2, max_model_len=16, max_num_seqs=4)
    proposer = NgramProposer(cfg)
    # req0：无 sampled ids（如 chunked prefill 尾拍）→ 跳过；
    # req1：历史 [1,2,3,1,2,3]（含 prompt），后缀 [1,2,3] 匹配位置 0 → 草稿 [1,2]；
    # req2：num_tokens_no_spec 已达 max_model_len → 跳过。
    sampled_token_ids = [[], [3], [3]]
    num_tokens_no_spec = np.array([3, 6, 16], dtype=np.int32)
    token_ids_cpu = np.zeros((3, 16), dtype=np.int32)
    token_ids_cpu[0, :3] = [9, 9, 9]
    token_ids_cpu[1, :6] = [1, 2, 3, 1, 2, 3]
    token_ids_cpu[2, :16] = 4

    drafts = proposer.propose(2, sampled_token_ids, num_tokens_no_spec, token_ids_cpu)
    assert drafts[0] == []
    assert drafts[1] == [1, 2]
    assert drafts[2] == []


def test_batch_propose_no_valid_requests():
    cfg = make_vllm_config()
    proposer = NgramProposer(cfg)
    # 空 valid 列表：不进 numba 批函数（真实 L93-L96 注释：空列表 fingerprint 报错）
    drafts = proposer.batch_propose(
        2, [], np.array([5, 5], dtype=np.int32),
        np.zeros((2, 16), dtype=np.int32), proposer.k,
    )
    assert drafts == [[], []]


def test_propose_asserts_k_within_capacity():
    cfg = make_vllm_config(k=2)
    proposer = NgramProposer(cfg)
    try:
        proposer.propose(3, [[1]], np.array([4], dtype=np.int32),
                         np.zeros((1, 16), dtype=np.int32))
        raised = False
    except AssertionError:
        raised = True
    assert raised  # 真实 L145：assert num_speculative_tokens <= self.k（动态 K 下调合法、上调不合法）


def test_numba_batch_matches_single_thread_reference():
    # numba 批路径与逐请求直接调用算法体产出一致（线程自适应已删、顺序等价）
    cfg = make_vllm_config(min_n=1, max_n=3, k=2, max_model_len=32, max_num_seqs=4)
    proposer = NgramProposer(cfg)
    histories = [
        [5, 6, 7, 5, 6, 7],
        [1, 2, 3, 4],
        [8, 8, 8, 8, 8],
    ]
    token_ids_cpu = np.zeros((3, 32), dtype=np.int32)
    for i, h in enumerate(histories):
        token_ids_cpu[i, : len(h)] = h
    num_tokens = np.array([len(h) for h in histories], dtype=np.int32)
    drafts = proposer.propose(2, [[1]] * 3, num_tokens, token_ids_cpu)
    for i, h in enumerate(histories):
        expect = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=token_ids_cpu[i, : num_tokens[i]],
            min_ngram=1, max_ngram=3, max_model_len=32, k=2,
        ).tolist()
        assert drafts[i] == expect


def test_load_model_noop():
    # 零模型 drafter：load_model 是 no-op（真实 L172-L174）
    NgramProposer(make_vllm_config()).load_model()
