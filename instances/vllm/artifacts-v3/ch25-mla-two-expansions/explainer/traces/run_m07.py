# ch25 m07 —— 批重排四区：reorder_batch_to_split_decodes_and_prefills + 后端自报阈值
# （vllm/v1/attention/backends/utils.py:L665-L742 互斥四区划分+换位；
#   gpu_model_runner.py:L1115-L1137 _may_reorder_batch / L7220-L7238 取最小阈值；
#   flashmla.py:L121 阈值 128 'process small prefills with decode pathway'）
# 场景一（手算档，threshold=2）：五请求乱序批 → 互斥四区判定 → 换位 →
#   在排好的批上跑真实 split_decodes_and_prefills 数边界。
# 场景二（生产档）：FlashMLA/FA-MLA 自报阈值与 runner 取最小。
from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from trace_common import T, dump

from vllm.v1.attention.backends.utils import (
    reorder_batch_to_split_decodes_and_prefills,
    split_decodes_and_prefills,
)

# ── 场景一：五请求乱序（tests TestReorderAndSplit.test_reorder_batch_four_regions 同款账）──


class FakeBatch:
    """真实 InputBatch 消费面：req_ids + num_computed_tokens_cpu /
    num_prompt_tokens + swap_states 换位钩子。"""

    def __init__(self, ids, computed, prompt):
        self.req_ids = list(ids)
        self.num_computed_tokens_cpu = np.array(computed)
        self.num_prompt_tokens = np.array(prompt)
        self.log = []

    def swap_states(self, a, b):
        self.log.append((int(a), int(b)))
        self.req_ids[a], self.req_ids[b] = self.req_ids[b], self.req_ids[a]
        for attr in ("num_computed_tokens_cpu", "num_prompt_tokens"):
            arr = getattr(self, attr)
            arr[a], arr[b] = arr[b], arr[a]


ids = ["p0", "long", "short", "d0", "d1"]
computed = [0, 8, 15, 10, 31]
prompt = [8, 20, 16, 10, 31]
scheduled = {"p0": 8, "long": 12, "short": 1, "d0": 1, "d1": 1}
threshold_toy = 2

# 互斥四区判定（与源码同式重放，供逐请求账）
has_context = [c > 0 for c in computed]
below = [scheduled[i] <= threshold_toy for i in ids]
done = [c >= p for c, p in zip(computed, prompt)]
region = []
for i in range(5):
    if not has_context[i]:
        region.append(3)                       # prefill（首块）
    elif not below[i]:
        region.append(2)                       # long_extend
    elif not done[i]:
        region.append(1)                       # short_extend
    else:
        region.append(0)                       # decode

batch = FakeBatch(ids, computed, prompt)
sched = SimpleNamespace(num_scheduled_tokens=scheduled)
changed = reorder_batch_to_split_decodes_and_prefills(
    batch, sched, decode_threshold=threshold_toy)
final_ids = list(batch.req_ids)
swaps = [(a, b) for a, b in batch.log]

# 已排好 → False（不动）
batch2 = FakeBatch(["d0", "d1", "p0"], [10, 31, 0], [10, 31, 8])
sched2 = SimpleNamespace(num_scheduled_tokens={"d0": 1, "d1": 1, "p0": 8})
unchanged = reorder_batch_to_split_decodes_and_prefills(
    batch2, sched2, decode_threshold=threshold_toy)

# 在排好的批上数边界（真实 split_decodes_and_prefills，threshold 同 2）
# 排好后 query_lens = [1,1,1,12,8]（d0,d1,short,long,p0）→ qsl 前缀和
qsl_sorted = torch.tensor([0, 1, 2, 3, 15, 23], dtype=torch.int32)
common_sorted = SimpleNamespace(
    max_query_len=12, num_reqs=5, num_actual_tokens=23,
    query_start_loc_cpu=qsl_sorted)
nd, npf, ndt, npt = split_decodes_and_prefills(
    common_sorted, decode_threshold=threshold_toy)

# ── 场景二：生产档阈值 ──
from vllm.v1.worker.gpu_model_runner import GPUModelRunnerSlice


def runner_with(thresholds):
    r = object.__new__(GPUModelRunnerSlice)
    r._attn_group_iterator = lambda: [
        SimpleNamespace(get_metadata_builder=lambda t=t: SimpleNamespace(
            reorder_batch_threshold=t))
        for t in thresholds
    ]
    return r


r1 = runner_with([128, 512])
r1.calculate_reorder_batch_threshold()
r2 = runner_with([128, None])
r2.calculate_reorder_batch_threshold()
r3 = runner_with([])
r3.calculate_reorder_batch_threshold()

trace = {
    "mechanism": "ch25-m07",
    "source": "run_m07.py @ implementation/（vLLM v0.27.1 只做减法精简版, host CPU）",
    "code_anchor": "vllm/v1/attention/backends/utils.py:L665-L742（reorder 四区+换位）· L564-L635（split 找边界）· gpu_model_runner.py:L1115-L1137/L7220-L7238 · flashmla.py:L121",
    "scenario_toy": {
        "decode_threshold": threshold_toy,
        "requests": [
            {"id": ids[i], "num_computed": computed[i],
             "num_prompt": prompt[i], "num_scheduled": scheduled[ids[i]],
             "has_context": has_context[i], "below_threshold": below[i],
             "done_prefilling": done[i], "region": region[i]}
            for i in range(5)
        ],
        "region_names": {0: "decode", 1: "short_extend", 2: "long_extend",
                         3: "prefill"},
        "order_before": ids,
        "order_after": final_ids,
        "swap_calls": swaps,
        "changed": bool(changed),
        "target_order_rule": "decode → short_extend → long_extend → prefill",
    },
    "idempotent_check": {
        "already_sorted_returns_false": bool(unchanged),
        "swap_log_empty": len(batch2.log) == 0,
    },
    "split_on_sorted_batch": {
        "query_lens_after_reorder": [1, 1, 1, 12, 8],
        "num_decodes": int(nd),          # 3
        "num_prefills": int(npf),        # 2
        "num_decode_tokens": int(ndt),   # 3
        "num_prefill_tokens": int(npt),  # 20
        "note": "short_extend（1 token 尾段）数进 decode 段——'process small prefills with decode pathway' 的计数面",
    },
    "production_thresholds": {
        "flashmla_reorder_threshold": 128,
        "flashmla_source": "vllm/v1/attention/backends/mla/flashmla.py:L121（注释原文 'process small prefills with decode pathway'）",
        "fa_mla_reorder_threshold": 512,
        "fa_mla_source": "vllm/v1/attention/backends/mla/flashattn_mla.py:L118（真实源码树现核，同款注释）",
        "runner_min_of_groups": {"min_128_512": r1.reorder_batch_threshold,
                                 "min_128_none": r2.reorder_batch_threshold,
                                 "no_groups": r3.reorder_batch_threshold},
        "runner_rule": "threshold = 全部 KV 组 builder 自报的最小值（gpu_model_runner.py:L7220-L7238）；空组 → None → 不重排",
    },
}

assert changed is True
assert final_ids == ["d0", "d1", "short", "long", "p0"]
assert (nd, npf, ndt, npt) == (3, 2, 3, 20)
assert r1.reorder_batch_threshold == 128 and r2.reorder_batch_threshold == 128
assert r3.reorder_batch_threshold is None
dump("m07.json", trace)
