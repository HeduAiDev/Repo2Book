# SUBTRACTED: SPDX 版权头（vllm/v1/spec_decode/ngram_proposer.py:L1-L2）。
# SUBTRACTED: `import os` 与 `from numba import ...`（L3, L7）—— 仅服务被减掉的
#             numba 线程数自适应与 @njit/@jit 加速；精简版纯 Python/numpy 顺序跑，
#             草稿结果完全等价。
# SUBTRACTED: `from vllm.config import VllmConfig`（L9）—— 精简版 __init__ 直接收
#             min_n/max_n/k/max_model_len 标量，不依赖 VllmConfig。
import numpy as np


# SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L12-L166
class NgramProposer:
    # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L13-L61
    # SUBTRACTED: 原 __init__ 从 vllm_config 取 prompt_lookup_min/max、
    #             num_speculative_tokens、max_model_len（L13-L27）；精简版直接收标量。
    # SUBTRACTED: numba batch 预分配 buffer self.valid_ngram_draft/num_drafts（L29-L32）。
    # SUBTRACTED: numba 线程数自适应 num_tokens_threshold / num_numba_thread_available
    #             (L34-L53) —— 纯性能调优。
    # SUBTRACTED: JIT 预热 self.propose([[]]*1024, ...)（L55-L61）—— 触发 numba 编译用。
    def __init__(
        self,
        prompt_lookup_min: int,
        prompt_lookup_max: int,
        num_speculative_tokens: int,
        max_model_len: int,
    ):
        # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L13-L61
        # Minimum length of the n-gram to match.
        self.min_n = prompt_lookup_min
        # Maximum length of the n-gram to match.
        self.max_n = prompt_lookup_max
        # Number of tokens follow the match. If there are less than k tokens
        # follow the match, we return the maximum amount of tokens until the end.
        self.k = num_speculative_tokens
        # Maximum length of the model.
        self.max_model_len = max_model_len

    # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L63-L129 (batch_propose)
    # SUBTRACTED: numba 线程切换（get_num_threads/set_num_threads/total_tokens 阈值，
    #             L92-L119）—— 纯性能；精简版顺序调用 numba 内核体 batch_propose_numba。
    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """对每个有效请求找最长匹配 n-gram、产 k 个草稿。返回 list[list[int]]。"""
        # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L63-L129
        draft_token_ids: list[list[int]] = []
        # 预分配每请求最多 k 个草稿的缓冲（对应真实 valid_ngram_draft）。
        valid_ngram_draft = np.zeros((num_requests, self.k), dtype=np.int32)
        valid_ngram_num_drafts = np.zeros((num_requests,), dtype=np.int32)

        if len(valid_ngram_requests):
            batch_propose_numba(
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                self.min_n,
                self.max_n,
                self.max_model_len,
                self.k,
                valid_ngram_draft,
                valid_ngram_num_drafts,
            )

        for i in range(num_requests):
            if i in valid_ngram_requests and valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    valid_ngram_draft[i, : valid_ngram_num_drafts[i]].tolist()
                )
            else:
                draft_token_ids.append([])

        return draft_token_ids

    # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L131-L162
    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings=None,  # unused
    ) -> list[list[int]]:
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
        )

        return draft_token_ids

    # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L164-L166
    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


# SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L169-L195 (batch_propose_numba)
# SUBTRACTED: @njit(parallel=True) 装饰器与 prange —— 仅 numba 并行加速；精简版用
#             普通 range 顺序遍历，结果等价。
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
    # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L169-L195
    for i in range(len(valid_ngram_requests)):
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


# SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L198-L285
# SUBTRACTED: @jit(nopython=True) 装饰器 —— 仅 numba 加速；函数体逐字保留，纯 numpy
#             可直接运行，产出的草稿与编译版完全一致。
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """
    Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, we will extract k right after the matched ngram.
    """
    # SOURCE: vllm/v1/spec_decode/ngram_proposer.py:L198-L285
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
