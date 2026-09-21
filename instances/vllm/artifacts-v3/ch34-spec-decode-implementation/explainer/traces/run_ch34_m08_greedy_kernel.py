# ch34 m8 worked example driver — greedy 拒绝 kernel：
# draft==target_argmax 接受、拒绝位写 argmax（=one-hot 残差唯一 token）、
# 早停截断、全收补 bonus、padded draft(-1) 直接拒、num_draft=0 请求 bonus 写第 0 位。
# 全程驱动精简版 rejection_sample 真跑（all_greedy 路径，Triton greedy kernel）。
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


class SMeta:
    all_greedy = True
    all_random = False
    generators = {}
    temperature = None


def main():
    trace = {
        "mechanism": "m8",
        "what": "greedy 拒绝 kernel：draft==argmax 接受 / 拒绝位写 argmax / 早停 / 全收补 bonus / -1 直接拒",
        "env": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
    }

    # 4 请求一批（num_draft=[3,2,1,0]，max_spec_len=3 → buffer [4,4]）：
    #   req0：草稿 [3,5,6] 全对 argmax [3,5,6] → 全收 + bonus 90
    #   req1：草稿 [1,0]，argmax [1,7] → pos1 拒 → 写 7、早停
    #   req2：草稿 [-1]（pad 占位），argmax [4] → 直接拒 → 写 4
    #   req3：num_draft=0（ngram 猜不出）→ bonus 写第 0 位
    num_draft = [3, 2, 1, 0]
    drafts_flat = [3, 5, 6, 1, 0, -1]  # 3+2+1+0 = 6 个摊平草稿
    argmax_flat = [3, 5, 6, 1, 7, 4]   # 6 个 target 位 argmax（req3 无 target 行）
    bonus = torch.tensor([[90], [91], [92], [93]], dtype=torch.int32, device=DEVICE)

    md = make_md(num_draft, drafts_flat)
    # kernel 内部先算 target_argmax = target_logits.argmax(-1)（L438）
    target_logits = one_hot_rows(argmax_flat)
    target_argmax = target_logits.argmax(dim=-1).tolist()

    out = rejection_sample(
        md.draft_token_ids, md.num_draft_tokens, md.max_spec_len,
        md.cu_num_draft_tokens, None, target_logits, bonus, SMeta,
    )

    # 逐请求逐位解读（与 kernel L735-L761 的循环一一对应）
    walks = []
    flat_pos = 0
    for req, k in enumerate(num_draft):
        positions = []
        rejected_at = None
        for pos in range(k):
            draft = drafts_flat[flat_pos + pos]
            argmax = argmax_flat[flat_pos + pos]
            if rejected_at is None:
                if draft < 0:
                    decision = "padded draft(-1)：直接拒（L807-L809 的 greedy 对偶——draft=-1 ≠ 任何 argmax）"
                    rejected_at = pos
                    written = argmax
                elif draft == argmax:
                    decision = "draft == argmax：接受"
                    written = draft
                else:
                    decision = "draft != argmax：首个不等 → 写 argmax（=one-hot 残差的唯一 token）并停"
                    rejected_at = pos
                    written = argmax
            else:
                decision = "早停后不写：保持 -1"
                written = -1
            positions.append({
                "pos": pos, "draft": draft, "target_argmax": argmax,
                "decision": decision, "written": written,
            })
        if rejected_at is None:
            positions.append({
                "pos": k, "draft": None, "target_argmax": None,
                "decision": "全收 → bonus 写在第 num_draft 位（L755-L761）",
                "written": int(bonus[req, 0].item()),
            })
        walks.append({
            "req": req,
            "num_draft_tokens": k,
            "flat_interval": [flat_pos, flat_pos + k],
            "positions": positions,
            "output_row": out[req].tolist(),
            "valid_tokens": [t for t in out[req].tolist() if t != -1],
        })
        flat_pos += k

    account = {
        "what": "产出账：output = accepted + recovered + bonus",
        "buffer_shape": list(out.shape),
        "buffer_cells": int(np.prod(out.shape)),
        "valid_cells": int((out != -1).sum().item()),
        "placeholder_cells": int((out == -1).sum().item()),
        "req_tokens": [len([t for t in out[r].tolist() if t != -1]) for r in range(4)],
        "provenance": "vllm/v1/sample/rejection_sampler.py:L715-L769（docstring L38-L59 四类 token 术语）",
    }

    trace.update({
        "params": {
            "num_draft_tokens": num_draft,
            "drafts_flat": drafts_flat,
            "target_argmax_flat": target_argmax,
            "max_spec_len": 3,
            "bonus_ids": [90, 91, 92, 93],
        },
        "target_argmax_kernel_computed": target_argmax,
        "output_matrix": out.tolist(),
        "per_request_walks": walks,
        "account": account,
        "provenance": "vllm/v1/sample/rejection_sampler.py:L715-L769（rejection_greedy_sample_kernel 逐字）",
    })

    out_path = pathlib.Path(__file__).resolve().parent / "ch34_m08_greedy_kernel.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print(json.dumps(trace, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
