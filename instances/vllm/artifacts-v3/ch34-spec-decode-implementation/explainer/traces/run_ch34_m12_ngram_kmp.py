# ch34 m12 worked example driver — ngram 零模型 drafter（KMP/LPS）。
# 全程驱动精简版 NgramProposer.propose 真跑（numba @njit(parallel=True) 批路径）；
# KMP 内部量（lps 数组演化 / longest_ngram / position / start_position）由
# 算法体 L242-L286 的逐字镜像记录（@jit 内部不可观测，镜像与真跑草稿互核）。
import json
import os
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np

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


def kmp_walk_mirror(origin_tokens, min_ngram, max_ngram, k):
    """ngram_proposer.py:L213-L286 主循环的逐字镜像（记录内部量）。

    与真跑 _find_longest_matched_ngram_and_propose_tokens 的输出互核
    （mirror_drafts == real_drafts）。
    """
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return {"early_exit": "total_token < min_ngram", "drafts": []}
    k_eff = min(k, max_ngram * 0 + k)  # max_model_len 视为无限（走外层截断）
    tokens = origin_tokens[::-1]
    lps = np.zeros(max_ngram, dtype=np.int32)
    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1
    events = []
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
                events.append({
                    "i": i, "match": True, "prev_lps": prev_lps,
                    "longest_ngram": longest_ngram, "position": position,
                    "lps": lps.tolist(),
                })
            else:
                events.append({
                    "i": i, "match": True, "prev_lps": prev_lps,
                    "longest_ngram": longest_ngram, "position": position,
                    "lps": lps.tolist(),
                })
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            events.append({
                "i": i, "match": False, "action": "prev_lps = lps[prev_lps-1]",
                "prev_lps_after": int(lps[prev_lps - 1]),
                "longest_ngram": longest_ngram, "position": position,
                "lps": lps.tolist(),
            })
            prev_lps = lps[prev_lps - 1]
        else:
            events.append({
                "i": i, "match": False, "action": "i += 1",
                "longest_ngram": longest_ngram, "position": position,
                "lps": lps.tolist(),
            })
            i += 1
    if longest_ngram < min_ngram:
        return {"early_exit": "longest_ngram < min_ngram",
                "longest_ngram": longest_ngram, "drafts": []}
    start_position = total_token - 1 - position + longest_ngram
    k_final = min(k, total_token - start_position)
    drafts = origin_tokens[start_position : start_position + k_final].tolist()
    return {
        "reversed_tokens": tokens.tolist(),
        "lps_final": lps.tolist(),
        "longest_ngram": longest_ngram,
        "position_reversed": position,
        "start_position": start_position,
        "matched_ngram": origin_tokens[
            total_token - 1 - position : total_token - 1 - position + longest_ngram
        ].tolist(),
        "drafts": drafts,
        "events": events,
    }


def main():
    trace = {
        "mechanism": "m12",
        "what": "ngram 零模型 drafter：翻转+KMP/LPS 找匹配后缀最长 n-gram、复制后续 k token",
        "env": {"numba": __import__("numba").__version__},
        "mirror_note": (
            "草稿全来自精简版 NgramProposer.propose 真跑（numba 批路径）；"
            "KMP 内部量（lps/longest_ngram/position/start_position）由算法体 "
            "L242-L286 逐字镜像记录，镜像草稿与真跑草稿互核一致。"
        ),
    }

    # ══ 场景 1（主例）：[1,2,3,4,1,2,3,4] —— 后缀 [1,2,3,4] 匹配位置 0 ═══
    tokens1 = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32)
    real1 = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens1, min_ngram=2, max_ngram=4, max_model_len=64, k=3
    ).tolist()
    mirror1 = kmp_walk_mirror(tokens1, 2, 4, 3)
    # 端到端：经 propose（numba 批路径）
    prop1 = NgramProposer(make_vllm_config(min_n=2, max_n=4, k=3, max_model_len=64, max_num_seqs=4))
    token_ids_cpu = np.zeros((1, 64), dtype=np.int32)
    token_ids_cpu[0, :8] = tokens1
    e2e1 = prop1.propose(3, [[1]], np.array([8], dtype=np.int32), token_ids_cpu)
    scenario1 = {
        "what": "主例：历史 [1,2,3,4,1,2,3,4]，min=2/max=4/k=3",
        "provenance": "vllm/v1/spec_decode/ngram_proposer.py:L206-L293",
        "history": tokens1.tolist(),
        "reversed_tokens": mirror1.get("reversed_tokens"),
        "lps_final": mirror1.get("lps_final"),
        "longest_ngram": mirror1.get("longest_ngram"),
        "position_reversed": mirror1.get("position_reversed"),
        "matched_ngram": mirror1.get("matched_ngram"),
        "start_position": mirror1.get("start_position"),
        "real_drafts": real1,
        "mirror_drafts": mirror1.get("drafts"),
        "mirror_matches_real": real1 == mirror1.get("drafts"),
        "e2e_propose_drafts": e2e1,
        "e2e_matches": e2e1[0] == real1,
        "kmp_events": mirror1.get("events"),
    }

    # ══ 场景 2：并列取最早出现（翻转后最靠右）══════════════════════════
    tokens2 = np.array([1, 2, 9, 1, 2, 8, 1, 2], dtype=np.int32)
    real2 = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens2, min_ngram=2, max_ngram=3, max_model_len=64, k=2
    ).tolist()
    mirror2 = kmp_walk_mirror(tokens2, 2, 3, 2)
    scenario2 = {
        "what": "并列取最早：[1,2] 在位置 0/3/5 三处出现，后缀匹配取原序列最早（翻转后最靠右）",
        "provenance": "vllm/v1/spec_decode/ngram_proposer.py:L248-L263（position 更新条件注释）",
        "history": tokens2.tolist(),
        "longest_ngram": mirror2.get("longest_ngram"),
        "position_reversed": mirror2.get("position_reversed"),
        "matched_ngram": mirror2.get("matched_ngram"),
        "start_position": mirror2.get("start_position"),
        "real_drafts": real2,
        "mirror_drafts": mirror2.get("drafts"),
        "mirror_matches_real": real2 == mirror2.get("drafts"),
        "note": "匹配 [1,2]@位置0 → 草稿=其后续 [9,1]（位置 3/5 的后续 [8]/[?] 不取）",
    }

    # ══ 场景 3：无匹配 → 空草稿 ═══════════════════════════════════════
    tokens3 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    real3 = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens3, min_ngram=2, max_ngram=3, max_model_len=64, k=3
    ).tolist()
    scenario3 = {
        "what": "无匹配：全历史无重复 2-gram → 空草稿（该请求本拍零投机）",
        "provenance": "vllm/v1/spec_decode/ngram_proposer.py:L276-L278",
        "history": tokens3.tolist(),
        "real_drafts": real3,
        "real_drafts_empty": len(real3) == 0,
    }

    # ══ 场景 4：k 被 max_model_len 截断 ═══════════════════════════════
    tokens4 = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32)
    real4 = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens4, min_ngram=2, max_ngram=4, max_model_len=10, k=3
    ).tolist()
    scenario4 = {
        "what": "k 截断：total=8、max_model_len=10 → k=min(3, 10-8)=2",
        "provenance": "vllm/v1/spec_decode/ngram_proposer.py:L219-L221",
        "history": tokens4.tolist(),
        "max_model_len": 10,
        "real_drafts": real4,
    }

    # ══ 场景 5：全同 token——lps 截在 max_ngram ════════════════════════
    tokens5 = np.array([7] * 10, dtype=np.int32)
    real5 = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens5, min_ngram=2, max_ngram=3, max_model_len=64, k=2
    ).tolist()
    mirror5 = kmp_walk_mirror(tokens5, 2, 3, 2)
    scenario5 = {
        "what": "全同 token：匹配长度被 max_ngram 截住（lps 只存前 max_ngram 个前缀）",
        "provenance": "vllm/v1/spec_decode/ngram_proposer.py:L233-L234 + L260-L264",
        "history": "全 7（10 个）",
        "longest_ngram": mirror5.get("longest_ngram"),
        "lps_final": mirror5.get("lps_final"),
        "real_drafts": real5,
        "mirror_drafts": mirror5.get("drafts"),
        "mirror_matches_real": real5 == mirror5.get("drafts"),
    }

    # ══ 复杂度账（quantified 用）═══════════════════════════════════════
    complexity = {
        "main_example": {
            "history_len": 8,
            "kmp_steps": len(mirror1.get("events", [])),
            "note": "i 每轮 +1 或 prev_lps 严格降 → 总步数 O(2n)，n=8 → 实测事件数见上",
        },
        "lps_memory": "O(max_ngram)——只存前 max_ngram 个前缀（本例 4 个 int32）",
        "gpu_forward_count": 0,
        "contrast": "EAGLE k 步草稿= k 次 draft 模型前向；ngram = 纯 CPU 字符串功夫、零 GPU 前向",
    }

    trace.update({
        "scenario1_main": scenario1,
        "scenario2_earliest": scenario2,
        "scenario3_no_match": scenario3,
        "scenario4_k_capped": scenario4,
        "scenario5_lps_capped": scenario5,
        "complexity": complexity,
    })

    def sanitize(o):
        if isinstance(o, dict):
            return {k: sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [sanitize(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    out_path = pathlib.Path(__file__).resolve().parent / "ch34_m12_ngram_kmp.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(sanitize(trace), f, ensure_ascii=False, indent=1)
    print(json.dumps(sanitize(trace), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
