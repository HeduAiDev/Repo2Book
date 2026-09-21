# SOURCE: vllm/v1/spec_decode/ngram_proposer.py —— 本章主角文件（只做减法）
# 零模型 drafter：numba 在 CPU 历史 token 里批量找匹配后缀的最长 n-gram
# （KMP/LPS），复制其后续 k 个 token 当草稿；不产概率分布（rejection 侧走
# NO_DRAFT_PROBS 分支）。减法 = dossier subtraction_plan.delete[3]：
# numba 线程自适应（阈值/线程数计算与 set_num_threads）与 __init__ 尾部的
# JIT 预热调用——纯性能调优，单线程顺序跑产出逐字节等价的草稿。
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch
from numba import jit, njit, prange

from vllm.config import VllmConfig

# SUBTRACTED: vllm/v1/spec_decode/ngram_proposer.py:L3 `import os` 与 L7 的
#   `get_num_threads, set_num_threads` 两名 —— delete[3] 连带 import 修剪
#   （os.cpu_count 只被线程数计算消费；get/set_num_threads 只被已删的
#   线程切换逻辑消费；@njit/@jit/prange 按「只删批准项」原样保留）。

# SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L12-L62 NgramProposer.__init__
#   —— 配置派生与 numba batch 预分配 buffer 逐字保留
# SUBTRACTED: vllm/v1/spec_decode/ngram_proposer.py:L34-L53 —— numba 线程数
#   自适应（num_tokens_threshold=8192 / cpu_count // 2 / cap 8 / ÷tp_size /
#   set_num_threads 尚未调用只是计算）—— delete[3] 纯性能调优。
# SUBTRACTED: vllm/v1/spec_decode/ngram_proposer.py:L55-L62 —— __init__ 尾部的
#   JIT 预热调用（self.propose(self.k, [[]] * 1024, ...) 触发 numba 编译，
#   注释原话 "Trigger Numba JIT compilation ... usually takes less than 1
#   second"）—— delete[3]；预热只搬编译时间、不改草稿结果。
class NgramProposer:
    def __init__(self, vllm_config: VllmConfig):
        # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L13-L62 NgramProposer.__init__（配置派生+numba buffer 预分配）
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # Minimum length of the n-gram to match.
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the n-gram to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # Number of tokens follow the match. If there are less than k
        # tokens follow the match, we will return the maximum amount
        # of tokens until the end.
        self.k = vllm_config.speculative_config.num_speculative_tokens
        # Maximum length of the model.
        self.max_model_len = vllm_config.model_config.max_model_len

        # Pre-allocate buffers for numba batch propose.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs), dtype=np.int32)

    # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L64-L133 batch_propose ——
    #   批量入口逐字保留（空列表卫兵/结果回读）
    # SUBTRACTED: vllm/v1/spec_decode/ngram_proposer.py:L97-L108 与 L122-L123 ——
    #   线程数选择/恢复（original_num_numba_threads=get_num_threads() /
    #   total_tokens 阈值判断 / set_num_threads(final_num_threads|1) /
    #   Restore original number of threads）—— delete[3]；单线程顺序调用
    #   batch_propose_numba 得到逐字节等价的草稿。
    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        k: int,
    ) -> list[list[int]]:
        # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L64-L133 batch_propose（批量入口）
        """Batch version of ngram proposer using numba for acceleration.

        Args:
            valid_ngram_requests:
                Set of indices of requests that need ngram proposals.
            num_tokens_no_spec:
                Numpy array of shape (batch_size,) representing the number
                of tokens without speculative tokens for each request.
            token_ids_cpu:
                Numpy array of shape (batch_size, max_model_len)
                representing the token IDs for each request.
            k:
                Number of speculative tokens to propose.

        Returns:
            list[list[int]]:
                A list where each element is a list of proposed
                token IDs for the corresponding request.
        """
        draft_token_ids: list[list[int]] = []

        # Only run batch propose if there are requests needing ngram proposals.
        # avoid calling numba function with empty list which causes error
        # ValueError: cannot compute fingerprint of empty list
        if num_ngram_requests := len(valid_ngram_requests):
            batch_propose_numba(
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                self.min_n,
                self.max_n,
                self.max_model_len,
                k,
                self.valid_ngram_draft,
                self.valid_ngram_num_drafts,
            )

        for i in range(num_requests):
            if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                )
            else:
                draft_token_ids.append([])

        return draft_token_ids

    # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L135-L170 propose —— 逐字
    #   （有效性过滤：无 sampled ids / 已达 max_model_len 的请求跳过）
    def propose(
        self,
        num_speculative_tokens: int,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L135-L170 propose（有效性过滤+分流 batch_propose）
        assert num_speculative_tokens <= self.k

        # find which requests need ngram proposals
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                continue

            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                continue

            valid_ngram_requests.append(i)

        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
            num_speculative_tokens,
        )

        return draft_token_ids

    # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L172-L174 load_model —— 逐字
    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


# SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L177-L203 batch_propose_numba
#   —— 逐字（@njit(parallel=True) 与 prange 按「只删批准项」保留；
#   host 无需 GPU，numba 线程数恒为默认 1，结果与并行版逐字节一致）
@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L177-L203 batch_propose_numba（numba 批函数）
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        context_token_ids = token_ids_cpu[idx, :num_tokens]
        drafter_output = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
        )

        valid_ngram_num_drafts[idx] = drafter_output.shape[0]
        if len(drafter_output):
            valid_ngram_draft[idx, : drafter_output.shape[0]] = drafter_output


# SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L206-L293
#   _find_longest_matched_ngram_and_propose_tokens —— KMP/LPS 算法体逐字
#   （翻转序列找匹配后缀的最长 n-gram；lps 只存前 max_ngram 个前缀省内存；
#   取翻转后最靠右的匹配 = 原序列最早出现；从匹配结束处复制 k 个 token）
@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L206-L293 _find_longest_matched_ngram_and_propose_tokens（KMP/LPS 算法体）
    """
    Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, we will extract k right after the matched ngram.
    """
    # Do not generate draft tokens is context is shorter than minimum n-gram
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Do not generate draft tokens beyond the max model length.
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip tokens, and the goal become to find longest ngram
    # on the rightmost position which matches the prefix with
    # length [min_n, max_n] (inclusive).
    tokens = origin_tokens[::-1]

    # Longest prefix (not including itself) which is a suffix of
    # the current position.
    #   lps[i] = max{v, where tokens[0:v] == tokens[i+1-v:i+1]}
    #
    # As ngram is capped by max_ngram to save memory, we only need to
    # store lps for the first max_ngram prefix.
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # lps[0] always equal to 0, we start with index 1
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] is the longest prefix as a suffix of tokens[:i]
        if tokens[prev_lps] == tokens[i]:
            # Token match: tokens[:prev_lps+1] is the longest prefix as
            # a suffix of tokens[:i+1]
            prev_lps += 1
            # Check if we found a longer valid ngram.
            #
            # Update position when longest_ngram matched prev_lps,
            # as we want to get the target n-gram of the earliest position
            # in the original tokens (i.e.
            # latest position in the reversed tokens)
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                # Store LPS for the first max_ngram prefix
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # When prev_lps reached max_ngram, update prev_lps
                # to lps[max_ngram-1] to avoid matching ngram
                # longer than max_ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            # Token mismatch: try the second-longest prefix
            # among all suffix of tokens[:i],
            # which is the longest prefix of tokens[:prev_lps]
            prev_lps = lps[prev_lps - 1]
        else:
            # Token mismatch, and no more prefix (except empty string)
            # as a suffix of tokens[:i]
            i += 1

    if longest_ngram < min_ngram:
        # No valid ngram is found
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip the position back, so in origin_tokens,
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # is the matched ngram, so we should start drafting tokens from
    # total_token-1-position+longest_ngram
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]
