# ch34 m4 worked example driver — 三组 index 摊平算术（_calc_spec_decode_metadata）。
#
# 取证方式（诚实声明）：_calc_spec_decode_metadata 在 gpu_model_runner.py
# （不在精简版范围，dossier delete[5]）。本驱动把它的 CPU numpy 主体**逐字复刻**
# （含 _get_cumsum_and_arange，gpu_model_runner.py:L1743-L1767）跑真数据：
#   例 A = 源码 docstring 自带的 5 请求数字（L2853-L2859）——跑完与注释逐项核对；
#   例 B = 3 请求小例（可心算），input_ids 给具体 token，验证二次 gather 错位一格。
import json
import os
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np


# ── _get_cumsum_and_arange（gpu_model_runner.py:L1743-L1767 逐字复刻）────
class ArangeScratch:
    """runner 的 self._arange_scratch / self.arange_np 预分配缓冲面。"""

    def __init__(self, capacity):
        self.arange_np = np.arange(capacity, dtype=np.int32)
        self.scratch = np.zeros(capacity, dtype=np.int32)


def _get_cumsum_and_arange(num_tokens, arange_out, cumsum_dtype=None):
    # [2, 5, 3] -> [2, 7, 10]，且 arange_out[:10] 写成 [0,1,0,1,2,3,4,0,1,2]
    cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
    total_num_tokens = cu_num_tokens[-1]
    cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
    np.subtract(
        arange_scratch.arange_np[:total_num_tokens],
        cumsums_offsets,
        out=arange_out[:total_num_tokens],
    )
    return cu_num_tokens


arange_scratch = None


def calc_spec_decode_metadata(num_draft_tokens, cu_num_scheduled_tokens, input_ids):
    """gpu_model_runner.py:L2851-L2924 的 CPU numpy 主体逐字复刻
    （async_tensor_h2d/GPU gather 的等价 numpy 形式；行号注释保留原位）。"""
    # Compute the logits indices.
    num_sampled_tokens = num_draft_tokens + 1

    # Step 1.
    cu_num_sampled_tokens = _get_cumsum_and_arange(
        num_sampled_tokens, arange_scratch.scratch, cumsum_dtype=np.int32
    )
    # Step 2.
    logits_indices = np.repeat(
        cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
    )
    # Step 3.
    logits_indices = logits_indices + arange_scratch.scratch[: cu_num_sampled_tokens[-1]]

    # Compute the bonus logits indices.
    bonus_logits_indices = cu_num_sampled_tokens - 1

    # Compute the draft logits indices.
    cu_num_draft_tokens = _get_cumsum_and_arange(
        num_draft_tokens, arange_scratch.scratch, cumsum_dtype=np.int32
    )
    target_logits_indices = np.repeat(
        cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
    )
    target_logits_indices = target_logits_indices + arange_scratch.scratch[
        : cu_num_draft_tokens[-1]
    ]

    # Compute the draft token ids.（真实代码为 GPU gather，此处 numpy 等价）
    draft_token_ids = input_ids[logits_indices]
    draft_token_ids = draft_token_ids[target_logits_indices + 1]

    return {
        "num_draft_tokens": num_draft_tokens,
        "num_sampled_tokens": num_sampled_tokens,
        "cu_num_draft_tokens": cu_num_draft_tokens,
        "cu_num_sampled_tokens": cu_num_sampled_tokens,
        "logits_indices": logits_indices,
        "bonus_logits_indices": bonus_logits_indices,
        "target_logits_indices": target_logits_indices,
        "target_plus_one": target_logits_indices + 1,
        "first_hop": input_ids[logits_indices],
        "draft_token_ids": draft_token_ids,
    }


def main():
    global arange_scratch

    # ── 例 A：源码 docstring 的 5 请求数字（L2853-L2859）────────────────
    arange_scratch = ArangeScratch(4096)
    cu_num_scheduled = np.array([4, 104, 107, 207, 209], dtype=np.int32)
    num_draft = np.array([3, 0, 2, 0, 1], dtype=np.int32)
    total_scheduled = int(cu_num_scheduled[-1])  # 209
    input_ids_a = np.arange(1000, 1000 + total_scheduled, dtype=np.int32)
    res_a = calc_spec_decode_metadata(num_draft, cu_num_scheduled, input_ids_a)

    docstring_expect = {
        "cu_num_draft_tokens": [3, 3, 5, 5, 6],
        "logits_indices": [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208],
        "target_logits_indices": [0, 1, 2, 5, 6, 9],
        "bonus_logits_indices": [3, 4, 7, 8, 10],
        "cu_num_sampled_tokens": [4, 5, 8, 9, 11],
        "draft_token_indices(tli+1)": [1, 2, 3, 6, 7, 10],
    }
    checks = {
        "cu_num_draft_tokens": res_a["cu_num_draft_tokens"].tolist() == docstring_expect["cu_num_draft_tokens"],
        "logits_indices": res_a["logits_indices"].tolist() == docstring_expect["logits_indices"],
        "target_logits_indices": res_a["target_logits_indices"].tolist() == docstring_expect["target_logits_indices"],
        "bonus_logits_indices": res_a["bonus_logits_indices"].tolist() == docstring_expect["bonus_logits_indices"],
        "cu_num_sampled_tokens": res_a["cu_num_sampled_tokens"].tolist() == docstring_expect["cu_num_sampled_tokens"],
        "tli+1": res_a["target_plus_one"].tolist() == docstring_expect["draft_token_indices(tli+1)"],
    }
    example_a = {
        "what": "源码 docstring 的 5 请求数字（gpu_model_runner.py:L2853-L2859）",
        "provenance": "vllm/v1/worker/gpu_model_runner.py:L2851-L2924（docstring 数字 + 算式逐字复刻）",
        "cu_num_scheduled_tokens": cu_num_scheduled.tolist(),
        "num_draft_tokens": num_draft.tolist(),
        "num_sampled_tokens": res_a["num_sampled_tokens"].tolist(),
        "cu_num_sampled_tokens": res_a["cu_num_sampled_tokens"].tolist(),
        "cu_num_draft_tokens": res_a["cu_num_draft_tokens"].tolist(),
        "logits_indices": res_a["logits_indices"].tolist(),
        "bonus_logits_indices": res_a["bonus_logits_indices"].tolist(),
        "target_logits_indices": res_a["target_logits_indices"].tolist(),
        "target_logits_indices_plus_1": res_a["target_plus_one"].tolist(),
        "docstring_crosscheck": checks,
        "all_match_docstring": all(checks.values()),
        "counts": {
            "batch": 5,
            "num_tokens_sum_drafts": int(num_draft.sum()),       # 6
            "logits_rows": int(res_a["cu_num_sampled_tokens"][-1]),  # 11 = num_tokens + batch
            "padding_rows_would_be": 5 * (int(num_draft.max()) + 1),  # 20
            "padding_waste_rows": 5 * (int(num_draft.max()) + 1) - int(res_a["cu_num_sampled_tokens"][-1]),
        },
    }

    # ── 例 B：3 请求小例（具体 token、二次 gather 全程可见）────────────
    arange_scratch = ArangeScratch(256)
    cu_num_scheduled_b = np.array([3, 5, 8], dtype=np.int32)
    num_draft_b = np.array([2, 0, 1], dtype=np.int32)
    # 本拍 8 个输入位（3 请求分别占 [0,3)/[3,5)/[5,8)），token 具体可读
    input_ids_b = np.array([50, 61, 72, 10, 20, 30, 41, 55], dtype=np.int32)
    res_b = calc_spec_decode_metadata(num_draft_b, cu_num_scheduled_b, input_ids_b)

    example_b = {
        "what": "3 请求小例：req0 草稿 2 个（token 61,72 = 输入位 1,2）、req1 零草稿（ngram 猜不出）、req2 草稿 1 个（token 55 = 输入位 7）",
        "provenance": "vllm/v1/worker/gpu_model_runner.py:L2851-L2924（算式逐字复刻）",
        "cu_num_scheduled_tokens": cu_num_scheduled_b.tolist(),
        "num_draft_tokens": num_draft_b.tolist(),
        "num_sampled_tokens": res_b["num_sampled_tokens"].tolist(),
        "cu_num_sampled_tokens": res_b["cu_num_sampled_tokens"].tolist(),
        "cu_num_draft_tokens": res_b["cu_num_draft_tokens"].tolist(),
        "logits_indices": res_b["logits_indices"].tolist(),
        "bonus_logits_indices": res_b["bonus_logits_indices"].tolist(),
        "target_logits_indices": res_b["target_logits_indices"].tolist(),
        "target_logits_indices_plus_1": res_b["target_plus_one"].tolist(),
        "input_ids": input_ids_b.tolist(),
        "first_hop_input_ids_logits_indices": res_b["first_hop"].tolist(),
        "second_hop_draft_token_ids": res_b["draft_token_ids"].tolist(),
        "expected_drafts_by_hand": [61, 72, 55],
        "hand_check_matches": res_b["draft_token_ids"].tolist() == [61, 72, 55],
        "counts": {
            "batch": 3,
            "num_tokens_sum_drafts": int(num_draft_b.sum()),
            "logits_rows": int(res_b["cu_num_sampled_tokens"][-1]),
            "padding_rows_would_be": 3 * (int(num_draft_b.max()) + 1),
            "padding_waste_rows": 3 * (int(num_draft_b.max()) + 1) - int(res_b["cu_num_sampled_tokens"][-1]),
        },
        "layout_note": (
            "req1 零草稿仍占 1 个采样位（logits_indices 里的 4）——bonus 位是每请求 "
            "k_i+1 的 +1；draft_token_ids 只有 3 个（零草稿请求不占草稿位）"
        ),
    }

    trace = {
        "mechanism": "m4",
        "what": "三组 index 摊平算术：np.repeat+arange 累积和 + 二次 gather 错位一格",
        "replication_note": (
            "gpu_model_runner.py 不在精简版范围（dossier delete[5]）；本驱动逐字复刻 "
            "L2851-L2924 与 _get_cumsum_and_arange(L1743-L1767) 的 CPU numpy 主体，"
            "例 A 与源码 docstring 注释逐项核对全对上。"
        ),
        "example_a_docstring": example_a,
        "example_b_small": example_b,
    }

    out_path = pathlib.Path(__file__).resolve().parent / "ch34_m04_index_flattening.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print(json.dumps(trace, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
