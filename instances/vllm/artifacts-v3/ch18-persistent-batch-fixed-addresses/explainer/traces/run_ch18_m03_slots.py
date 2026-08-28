"""ch18-m03 slot 复用与压实算法 —— 驱动脚本（InputBatch 直驱，host 纯 CPU）。

不走 runner：直接驱动 InputBatch 的 add_request / remove_request / condense，
逐步记录 BatchUpdateBuilder 内部（removed 降序表）与批次布局，把
『打洞 → pop_removed 复用最小空 slot → condense 尾部滑入（只拷活跃前缀）』
三段算法的每一步摊开。

剧本（max_num_reqs=8 / max_model_len=16 / block_size=16）：
  第一轮（同拍删→压实）：
    1-4   add a(10 tokens)/b(3)/c(1)/d(1) → slot 0/1/2/3
    5-6   remove b、remove c → 洞@1、洞@2；removed=[2,1]（降序维持）
    7     condense：双指针 last=3、peek=1 → d 从 3 滑入 1（只拷 1 个活跃
          token；b 的陈旧尾巴 [21,22] 留在 row1 不动）→ 洞@2 ≥ last → 截断
    8     refresh_metadata（真实拍末收尾；解封 builder，进第二轮）
  第二轮（同拍删→增复用）：
    9     remove a → 洞@0；removed=[0]
    10    add e → pop_removed=0 → e@0（最小洞优先）
    11    add f → pop_removed=None → f 落在 num_reqs=2（追加）
    12    condense → removed 已被 add 填平 → 零成本早退（docstring 分支）

跑法：python explainer/traces/run_ch18_m03_slots.py
产物：explainer/traces/ch18_m03_slots.json
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation._host_seams import SamplingParams  # noqa: E402
from implementation.gpu_input_batch import CachedRequestState, InputBatch  # noqa: E402

VOCAB = 32


def _greedy():
    return SamplingParams(temperature=0.0)


def _req_state(req_id, prompt, block_ids, output=()):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=_greedy(),
        generator=None,
        block_ids=(list(block_ids),),
        num_computed_tokens=0,
        output_token_ids=list(output),
    )


def _mk_batch():
    return InputBatch(
        max_num_reqs=8,
        max_model_len=16,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        vocab_size=VOCAB,
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks_per_req=[1],
    )


STEP = [0]


def _snap(ib, note, removed_public=None):
    """批次布局快照：_req_ids（含 None 洞）/ 映射 / num_reqs / 各行 token 前 4 列 /
    removed 表 / moved 表。

    removed 探针：默认读私有 _removed 的副本（纯观察——公开属性 removed 的首次
    读取会封账（removed_append 此后 RuntimeError，state.py:L84-L90），真实流的
    读只发生在本拍全部 remove 之后）；removed_public 给定时记录一次**合法时点**
    （本拍 remove 已全部完成）的公开属性读取，演示降序保证。
    """
    STEP[0] += 1
    step_no = STEP[0]
    rows = {}
    for i in range(len(ib._req_ids)):
        rid = ib._req_ids[i]
        n = int(ib.num_tokens_no_spec[i]) if rid is not None else -1
        rows[f"row{i}"] = {
            "req": rid,
            "active_prefix_len": n,
            "tokens_first4": ib.token_ids_cpu[i, :4].tolist(),
            "block": int(ib.block_table[0].block_table.np[i, 0]),
        }
    builder = {
        # 探针：镜像排序展示（不触发 _ensure_removed_sorted/封账）
        "removed_desc_probe": sorted(
            list(ib.batch_update_builder._removed), reverse=True
        ),
        "added_count": len(ib.batch_update_builder.added),
        "moved": [
            [int(m[0]), int(m[1]), getattr(m[2], "name", str(m[2]))]
            for m in ib.batch_update_builder.moved
        ],
    }
    if removed_public is not None:
        builder["removed_public_read_at_legal_point"] = removed_public
    return {
        "step": step_no,
        "note": note,
        "_req_ids": list(ib._req_ids),
        "req_id_to_index": dict(ib.req_id_to_index),
        "num_reqs": int(ib.num_reqs),
        "builder": builder,
        "rows": rows,
    }


def main():
    ib = _mk_batch()
    steps = []

    steps.append(_snap(ib, "初始：空批次，removed=[]"))

    # 第一轮：同拍 删→删→condense
    ib.add_request(_req_state("a", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [1]))
    steps.append(_snap(ib, "add a（10 tokens）→ pop_removed=None → slot=num_reqs=0"))
    ib.add_request(_req_state("b", [20, 21, 22], [2]))
    steps.append(_snap(ib, "add b（3 tokens）→ slot=1"))
    ib.add_request(_req_state("c", [30], [3]))
    steps.append(_snap(ib, "add c（1 token）→ slot=2"))
    ib.add_request(_req_state("d", [40], [4]))
    steps.append(_snap(ib, "add d（1 token）→ slot=3；批满 4 行"))

    ib.remove_request("b")
    steps.append(_snap(ib, "remove b → 打洞@1：_req_ids[1]=None、映射解绑、块表行清零；数据不搬；removed=[1]"))
    ib.remove_request("c")
    # 本拍 remove 已全部完成 → 此刻读公开属性是合法时点（condense 内部也在此后读）
    legal_removed = list(ib.batch_update_builder.removed)
    steps.append(_snap(
        ib,
        "remove c → 打洞@2；公开属性读取（合法时点）：removed=[2,1] 降序",
        removed_public=legal_removed,
    ))

    ib.condense()
    steps.append(_snap(
        ib,
        "condense：last=num_reqs(2)+len(removed)(2)-1=3；peek=1<3 → d 从 3 滑入 1"
        "（只拷活跃前缀 1 个 token=[40]；b 的陈旧尾巴 [21,22] 仍在 row1[1:3]）；"
        "last 降为 2 ∈ removed → 再降为 1；peek=2 ≥ 1 → break；截断 _req_ids 到 num_reqs=2",
    ))

    ib.refresh_metadata()
    steps.append(_snap(ib, "refresh_metadata：get_and_reset 产出 BatchUpdate（removed=[2]/moved=[(3,1)]）喂 logitsprocs 并解封 builder"))

    # 第二轮：同拍 删→增→增→condense
    ib.remove_request("a")
    legal_removed_r2 = list(ib.batch_update_builder.removed)
    steps.append(_snap(
        ib,
        "remove a → 打洞@0；公开属性读取（合法时点）：removed=[0]；num_reqs=1（只剩 d@1）",
        removed_public=legal_removed_r2,
    ))

    ib.add_request(_req_state("e", [50, 51], [5]))
    steps.append(_snap(ib, "add e → pop_removed=0 → e@0（最小空 slot 优先复用）"))

    ib.add_request(_req_state("f", [60], [6]))
    steps.append(_snap(ib, "add f → pop_removed=None（洞已填平）→ slot=num_reqs=2（追加）"))

    ib.condense()
    steps.append(_snap(ib, "condense → removed 为空（全被 add 填平）→ 零成本早退"))

    trace = {
        "driver": "run_ch18_m03_slots.py",
        "mechanism": "ch18-m03 slot 复用与压实（gpu_input_batch.py:L324-L348/L530-L548/L708-L838 + logits_processor/state.py:L18-L145）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch18 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "trace_environment": "Windows host 纯 CPU（InputBatch 直驱，不经 runner 前向）",
        "config": {
            "max_num_reqs": 8,
            "max_model_len": 16,
            "block_size": 16,
            "vocab": VOCAB,
            "requests": {
                "a": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                "b": [20, 21, 22],
                "c": [30],
                "d": [40],
                "e": [50, 51],
                "f": [60],
            },
        },
        "steps": steps,
    }

    out_path = os.path.join(os.path.dirname(__file__), "ch18_m03_slots.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print(f"trace -> {out_path}")
    for s in steps:
        print(f"[{s['note'][:46]}...] _req_ids={s['_req_ids']} "
              f"removed={s['builder']['removed_desc_probe']} num_reqs={s['num_reqs']}")


if __name__ == "__main__":
    main()
