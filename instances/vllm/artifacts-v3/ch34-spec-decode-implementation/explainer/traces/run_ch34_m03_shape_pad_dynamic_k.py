# ch34 m3 worked example driver — 形状三件事：统一 pad 到 1+K（[-1] 占位草稿）、
# 『宁可不排不破形状』、dynamic_sd_lookup 按批大小动态定 K。
#
# 取证方式（诚实声明）：scheduler.py 与 dynamic/utils.py 不在本章精简版范围
# （dossier delete[5]）。pad 判定（L881-L898）、占位草稿（L1076-L1078）、
# 动态 K 查表（L1192-L1197）与 build_dynamic_sd_schedule_lookup（dynamic/utils.py:
# L77-L140）的算式逐字复刻自 v0.27.1 真实源码；[-1] 占位草稿的下游效果由精简版
# rejection_sampler.rejection_sample 真跑验证（greedy Triton kernel 对 -1 直接拒）。
import json
import os
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np
import torch

from vllm.v1.sample.rejection_sampler import rejection_sample
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def r6(x):
    return round(float(x), 6)


# ── build_dynamic_sd_schedule_lookup（vllm/v1/spec_decode/dynamic/utils.py:
#    L77-L140 逐字复刻：区间展开成稠密 batch_size→K 查表、gap 顺延、尾段顺延、
#    min(vllm_num_spec, k) 封顶）────────────────────────────────────────
def build_dynamic_sd_schedule_lookup(
    num_speculative_tokens_per_batch_size,
    vllm_max_batch_size,
    vllm_num_speculative_tokens,
):
    # validate_and_normalize_dynamic_sd_schedule（L10-L73）在本驱动输入上恒合法，
    # 直接按 parse 分支复刻
    parsed_schedule = [
        (int(e[0]), int(e[1]), int(e[2]))
        for e in num_speculative_tokens_per_batch_size
    ]
    parsed_schedule.sort(key=lambda entry: entry[0])

    dense_schedule = [0] * (vllm_max_batch_size + 1)
    next_batch_size = 1
    last_num_speculative_tokens = None

    for range_start, range_end, num_speculative_tokens in parsed_schedule:
        if range_start > next_batch_size and last_num_speculative_tokens is not None:
            # Fill any gap before the next configured range by carrying forward
            # the previous K.
            for batch_size in range(
                next_batch_size, min(range_start, vllm_max_batch_size + 1)
            ):
                dense_schedule[batch_size] = min(
                    vllm_num_speculative_tokens, last_num_speculative_tokens
                )
        for batch_size in range(
            max(range_start, next_batch_size),
            min(range_end, vllm_max_batch_size) + 1,
        ):
            dense_schedule[batch_size] = min(
                vllm_num_speculative_tokens, num_speculative_tokens
            )
        next_batch_size = max(next_batch_size, range_end + 1)
        last_num_speculative_tokens = num_speculative_tokens
        if next_batch_size > vllm_max_batch_size:
            break

    # 尾段顺延（L140-L152 语义）
    for batch_size in range(next_batch_size, vllm_max_batch_size + 1):
        dense_schedule[batch_size] = min(
            vllm_num_speculative_tokens, last_num_speculative_tokens
        )
    return dense_schedule


def greedy_verify_padded(placeholder_drafts, argmax_id, vocab=100):
    """真跑：占位草稿 [-1]*K 进 greedy kernel → 第一位即拒、写 argmax、其余 -1。"""
    k = len(placeholder_drafts)
    assert all(t == -1 for t in placeholder_drafts)
    row = torch.zeros(vocab, dtype=torch.float32)
    row[argmax_id] = 10.0
    # k 个草稿位各有自己的 target logits 行（padded 请求三个采样位分布相同）
    target_logits = row.unsqueeze(0).repeat(k, 1).to(DEVICE)
    md = SpecDecodeMetadata(
        draft_token_ids=torch.tensor(placeholder_drafts, dtype=torch.int32, device=DEVICE),
        num_draft_tokens=[k],
        cu_num_draft_tokens=torch.tensor([k], dtype=torch.int32, device=DEVICE),
        cu_num_sampled_tokens=torch.tensor([k + 1], dtype=torch.int32, device=DEVICE),
        target_logits_indices=torch.zeros(k, dtype=torch.int32, device=DEVICE),
        bonus_logits_indices=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        logits_indices=torch.zeros(k + 1, dtype=torch.int32, device=DEVICE),
    )

    class SM:
        all_greedy = True
        all_random = False
        generators = {}
        temperature = None

    bonus = torch.full((1, 1), 99, dtype=torch.int32, device=DEVICE)
    out = rejection_sample(
        md.draft_token_ids, md.num_draft_tokens, md.max_spec_len,
        md.cu_num_draft_tokens, None, target_logits, bonus, SM(),
    )
    row_out = out[0].tolist()
    parsed = [t for t in row_out if t != -1]
    return row_out, parsed


def main():
    trace = {
        "mechanism": "m3",
        "what": "形状三件事：统一 pad 到 1+K / 宁可不排不破形状 / dynamic_sd_lookup 动态 K",
        "replication_note": (
            "scheduler.py:L881-L898/L1076-L1078/L1192-L1197 与 dynamic/utils.py:"
            "L77-L152 算式逐字复刻（文件不在精简版范围，dossier delete[5]）；"
            "占位草稿下游效果由精简版 rejection_sample 真跑验证。"
        ),
        "env": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
    }

    # ── 事 1：新 decode 请求 pad 到 1+K（L881-L898 逐字复刻判定）────────
    K = 3
    pad_cases = []
    # 场景 A：批里已有 3 个 RUNNING decode 请求（各 1+3=4 token），新请求 r4
    # 刚做完 prefill（num_tokens=6, num_computed=6 → num_new_tokens=1）
    num_spec_tokens = K
    dynamic_sd_lookup = None            # 静态 K 模式（dynamic 开启时不 pad，L885）
    num_sampled_tokens_per_step = 1
    num_new_tokens = 1                  # = request.num_tokens - num_computed_tokens (L879)
    scheduled_running_reqs = True
    prefill_scheduled = False
    token_budget = 2048
    num_computed_tokens = 6
    max_model_len = 64

    cond = (
        (num_spec_tokens > 0 and dynamic_sd_lookup is None)
        and num_sampled_tokens_per_step > 0
        and num_new_tokens == 1
        and (scheduled_running_reqs and not prefill_scheduled)
    )
    pad_spec_decode = False
    if cond:
        num_new_tokens = 1 + num_spec_tokens          # L891：pad 到 1+K
        if (
            num_new_tokens > token_budget
            or num_computed_tokens + num_new_tokens > max_model_len
        ):
            decision = "break：Prefer to not schedule than schedule un-padded（L894-L897）"
        else:
            pad_spec_decode = True
            decision = "pad_spec_decode=True：r4 以 1+3=4 token 进批，批形状统一"
    else:
        decision = "不 pad（条件不满足）"
    # L1076-L1078：占位草稿
    scheduled_spec = [-1] * num_spec_tokens if pad_spec_decode else None
    pad_cases.append({
        "case": "A 预算充足：新请求 pad 到 1+K",
        "provenance": "vllm/v1/core/sched/scheduler.py:L881-L898 + L1076-L1078",
        "running_reqs": 3, "running_token_shape": [1 + K] * 3,
        "new_req_num_new_tokens_before": 1,
        "condition_met": bool(cond),
        "num_new_tokens_after": num_new_tokens,
        "scheduled_spec_decode_tokens": scheduled_spec,
        "batch_shape_after": [1 + K] * 4,
        "decision": decision,
    })

    # 场景 B：token_budget 只剩 2 → pad 后 4 > 2 → 整拍不收该请求
    token_budget_b = 2
    num_new_b = 1 + num_spec_tokens
    over_budget = num_new_b > token_budget_b
    pad_cases.append({
        "case": "B 预算不足：pad 后 4 > 剩余预算 2",
        "provenance": "vllm/v1/core/sched/scheduler.py:L891-L897",
        "token_budget": token_budget_b,
        "num_new_tokens_padded": num_new_b,
        "over_budget": bool(over_budget),
        "decision": "break：宁可不排也不让批形状破——新请求等下一拍（保住其余 3 个请求的 cudagraph 形状）",
    })

    # 场景 C：dynamic K 开启 → 不走 pad（L885 条件 dynamic_sd_lookup is None 不满足）
    pad_cases.append({
        "case": "C dynamic_sd_lookup 开启",
        "provenance": "vllm/v1/core/sched/scheduler.py:L885",
        "condition_dynamic_none": False,
        "decision": "不 pad——动态 K 下每拍 K 由查表决定（批大小不同 K 不同，本就不存在统一形状）",
    })

    # ── 占位草稿的下游效果（真跑）──────────────────────────────────────
    row_out, parsed = greedy_verify_padded([-1, -1, -1], argmax_id=4)
    downstream = {
        "what": "[-1,-1,-1] 占位草稿进 greedy kernel（真跑）",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L715-L769（draft=-1 ≠ argmax → 首位即拒）",
        "drafts": [-1, -1, -1],
        "target_argmax": 4,
        "output_row": row_out,
        "parsed_output": parsed,
        "note": "输出 [4,-1,-1,-1] → parse_output 还原 [4]：新请求本拍照常拿 1 个 token，只是没有投机收益",
    }

    # ── 事 2：dynamic_sd_lookup 查表（L1192-L1197 + utils.py 构建）──────
    sched_cfg = [(1, 2, 3), (3, 4, 2)]
    lookup = build_dynamic_sd_schedule_lookup(sched_cfg, 4, 3)
    # L1192-L1197：num_spec_tokens_to_schedule = lookup[len(num_scheduled_tokens)]
    dyn_cases = []
    for batch_size in (1, 2, 3, 4):
        num_spec_tokens_to_schedule = lookup[batch_size]
        dyn_cases.append({
            "batch_size": batch_size,
            "num_spec_tokens_to_schedule": num_spec_tokens_to_schedule,
        })
    # gap 顺延 + min 封顶两个非平凡分支：
    lookup_gap = build_dynamic_sd_schedule_lookup([(1, 2, 3), (4, 4, 1)], 4, 3)
    lookup_cap = build_dynamic_sd_schedule_lookup([(1, 4, 9)], 4, 3)
    dynamic = {
        "provenance": "vllm/v1/core/sched/scheduler.py:L1192-L1197 + vllm/v1/spec_decode/dynamic/utils.py:L77-L152",
        "config": {"schedule": sched_cfg, "max_batch": 4, "num_spec_tokens": 3},
        "dense_lookup": lookup,
        "cases": dyn_cases,
        "gap_carry": {
            "config": [(1, 2, 3), (4, 4, 1)],
            "dense_lookup": lookup_gap,
            "note": "batch=3 落在配置区间外 → 顺延前段 K=3（utils.py gap 顺延分支）",
        },
        "cap": {
            "config": [(1, 4, 9)],
            "dense_lookup": lookup_cap,
            "note": "区间配 K=9 > vllm_num_speculative_tokens=3 → 封顶 3（min(vllm_num_spec, k)）",
        },
        "note": "查出的 num_spec_tokens_to_schedule 随 SchedulerOutput 下发，drafter 下一轮按它出草稿（重载缩 K、轻载放大 K）",
    }

    trace.update({
        "pad_cases": pad_cases,
        "placeholder_downstream": downstream,
        "dynamic_sd": dynamic,
    })

    out_path = pathlib.Path(__file__).resolve().parent / "ch34_m03_shape_pad_dynamic_k.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print(json.dumps(trace, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
