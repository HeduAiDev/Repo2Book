# ch34 m7 worked example driver — rejection_sample 调度：
# [B, max_spec_len+1] 预填 PLACEHOLDER=-1、all_greedy 早退（不消耗任何 RNG）、
# is_greedy mask 双 kernel 同批共存各自早退。
# 全程驱动精简版 rejection_sample 真跑（Triton 双 kernel）。
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
V = 8


def r6(x):
    return round(float(x), 6)


def make_md(num_draft, drafts_flat):
    cu_draft = np.cumsum(num_draft, dtype=np.int32)
    num_tokens = int(cu_draft[-1])
    batch = len(num_draft)
    return SpecDecodeMetadata(
        draft_token_ids=torch.tensor(drafts_flat, dtype=torch.int32, device=DEVICE),
        num_draft_tokens=num_draft,
        cu_num_draft_tokens=torch.from_numpy(cu_draft).to(DEVICE),
        cu_num_sampled_tokens=torch.tensor(
            np.cumsum([n + 1 for n in num_draft], dtype=np.int32)).to(DEVICE),
        target_logits_indices=torch.arange(num_tokens, dtype=torch.int32, device=DEVICE),
        bonus_logits_indices=torch.arange(batch, dtype=torch.int32, device=DEVICE),
        logits_indices=torch.arange(num_tokens + batch, dtype=torch.int32, device=DEVICE),
    )


def one_hot_rows(argmax_seq):
    rows = []
    for a in argmax_seq:
        row = torch.zeros(V, dtype=torch.float32)
        row[int(a)] = 10.0
        rows.append(row)
    return torch.stack(rows).to(DEVICE)


def main():
    trace = {
        "mechanism": "m7",
        "what": "rejection_sample 调度：-1 预填 / all_greedy 早退 / 双 kernel 同批共存",
        "env": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
    }

    # ══ Case A：all_greedy 早退——不消耗任何 RNG ════════════════════════
    # 真实路径里 greedy kernel 不吃 uniform/q；用 generator 状态不变来实证
    num_draft_a = [2, 1]
    drafts_a = [3, 5, 1]
    argmax_a = [3, 5, 1]  # 全对 → 全收 + bonus
    md_a = make_md(num_draft_a, drafts_a)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(2024)
    state_before = g.get_state().clone()

    class SMetaAllGreedy:
        all_greedy = True
        all_random = False
        generators = {0: g, 1: g}
        temperature = None

    bonus_a = torch.tensor([[99], [98]], dtype=torch.int32, device=DEVICE)
    out_a = rejection_sample(
        md_a.draft_token_ids, md_a.num_draft_tokens, md_a.max_spec_len,
        md_a.cu_num_draft_tokens, None,
        one_hot_rows(argmax_a), bonus_a, SMetaAllGreedy,
    )
    state_after = g.get_state()
    case_a = {
        "what": "all_greedy=True：greedy kernel 后直接 return（L448-L449）",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L421-L449",
        "params": {"num_draft_tokens": num_draft_a, "drafts": drafts_a,
                   "target_argmax": argmax_a, "max_spec_len": 2},
        "output_shape": list(out_a.shape),
        "output": out_a.tolist(),
        "buffer_prefilled": "PLACEHOLDER_TOKEN_ID=-1（L413-L419 预填；未写位保持 -1）",
        "generator_state_unchanged": bool(torch.equal(state_before, state_after)),
        "note": "softmax/recovered/uniform 全没跑——generator 一个随机数都没消耗（实证）",
    }

    # ══ Case B：混批 is_greedy mask——greedy/random 双 kernel 各自早退 ════
    # req0 greedy(temp=0) 两草稿：pos0 对、pos1 错 → greedy kernel 写 [3,7,-1]
    # req1 random(temp=1.0) 一草稿：由 random kernel 判（u 重构见 m9，此处记录结果）
    num_draft_b = [2, 1]
    drafts_b = [3, 5, 2]
    argmax_b = [3, 7, 0]  # req0: pos1 argmax=7 ≠ draft 5 → 拒
    # req1 的 target 分布：token2 概率 0.9（log 近似 one-hot 但留随机性）
    rows = [torch.zeros(V) for _ in argmax_b]
    for row, a in zip(rows, argmax_b):
        row[int(a)] = 4.0
    # req1 行换成软分布 p(2)=0.9, p(6)=0.1
    import math
    rows[2] = torch.zeros(V)
    rows[2][2] = math.log(0.9)
    rows[2][6] = math.log(0.1)
    target_b = torch.stack(rows).float().to(DEVICE)

    g0 = torch.Generator(device=DEVICE); g0.manual_seed(7)   # req0（greedy，不会消耗）
    g1 = torch.Generator(device=DEVICE); g1.manual_seed(11)  # req1（random 消耗）

    class SMetaMixed:
        all_greedy = False
        all_random = False
        # is_greedy = temperature == 0（L424）：[True, False]
        temperature = torch.tensor([0.0, 1.0], dtype=torch.float32, device=DEVICE)
        generators = {0: g0, 1: g1}

    # 重构 req1 的 uniform（同一 seed 同一消耗序列 = kernel 里实际用的 u）：
    # generate_uniform_probs 先对 req1 的 1 个草稿位 uniform_(generator=g1)
    g1_replay = torch.Generator(device=DEVICE); g1_replay.manual_seed(11)
    u1_replay = torch.empty(1, dtype=torch.float64, device=DEVICE)
    u1_replay.uniform_(generator=g1_replay)

    md_b = make_md(num_draft_b, drafts_b)
    bonus_b = torch.tensor([[99], [98]], dtype=torch.int32, device=DEVICE)
    out_b = rejection_sample(
        md_b.draft_token_ids, md_b.num_draft_tokens, md_b.max_spec_len,
        md_b.cu_num_draft_tokens, None, target_b, bonus_b, SMetaMixed,
    )
    # 注：greedy kernel 不读 generator、random kernel 对 greedy 请求早退，但
    # sample_recovered_tokens 的 q 按 [batch, V] 全批分配——req0 带 g0 且
    # num_draft=2>0 → q[0] 行会用 g0 覆写（采了但无人消费的真实副作用）。
    u1 = r6(u1_replay[0].item())
    p_t_x = r6(torch.softmax(target_b[2], dim=-1)[2].item())
    case_b = {
        "what": "混批：is_greedy=[True,False] 双 kernel 同批共存、各自早退",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L421-L485（is_greedy mask L424）+ L730-L733/L789-L792（kernel 早退）",
        "params": {
            "num_draft_tokens": num_draft_b,
            "drafts": drafts_b,
            "temperature": [0.0, 1.0],
            "is_greedy_mask": [True, False],
            "req0_target_argmax": [3, 7],
            "req1_target_probs": {"token2": p_t_x},
        },
        "output": out_b.tolist(),
        "output_shape": list(out_b.shape),
        "req1_uniform_reconstructed": u1,
        "req1_uniform_dtype": "float64（generate_uniform_probs L634-L638）",
        "req1_criterion": f"NO_DRAFT_PROBS：draft_prob=1 → 接受当且仅当 p_t(x)={p_t_x} >= u={u1}",
        "which_kernel": {
            "req0": "greedy kernel（random kernel 见 is_greedy=True 早退 L789-L792）",
            "req1": "random kernel（greedy kernel 见 is_greedy=False 早退 L730-L733）",
        },
        "note": (
            "两 kernel 同批 launch（L439+L471），is_greedy mask 行级分流——免拆批；"
            "greedy 请求也会被 sample_recovered_tokens 分到 q 行（全批分配），只是结果无人消费"
        ),
    }

    # ══ buffer 形状账 ════════════════════════════════════════════════
    buffer_account = {
        "what": "输出 buffer 第二维 = max_spec_len+1（+1 即 bonus 槽）",
        "case_a": {"batch": 2, "max_spec_len": 2, "shape": list(out_a.shape),
                   "cells": int(np.prod(out_a.shape)), "placeholder_cells":
                   int((out_a == -1).sum().item())},
        "case_b": {"batch": 2, "max_spec_len": 2, "shape": list(out_b.shape),
                   "cells": int(np.prod(out_b.shape)), "placeholder_cells":
                   int((out_b == -1).sum().item())},
        "provenance": "vllm/v1/sample/rejection_sampler.py:L413-L419",
    }

    trace.update({
        "case_a_all_greedy": case_a,
        "case_b_mixed": case_b,
        "buffer_account": buffer_account,
    })

    out_path = pathlib.Path(__file__).resolve().parent / "ch34_m07_dispatch.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print(json.dumps(trace, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
