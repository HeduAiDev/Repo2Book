# ch16 m8 逐层重叠：start_load_kv 异步发起全部层；每个注意力层执行前
# wait_for_layer_load 只等本层、执行后 save_kv_layer 异步存出
# （base.py:L304-L367 + kv_transfer_utils.py:L15-L61）。
# 部件1（真实运行）：4 层假模型跑一个真实前向拍，记录调用序。
# 部件2（教学时间账）：离散事件模拟——每层传输 2 拍（链路按层序串行）、
# 每层计算 3 拍，对比「等全部到齐再算」vs「逐层重叠」。时间为虚拟拍，
# 非实测 GPU 数（说明性模型；真实收益取决于传输/计算比）。
import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from _harness import MODULE_PATH, dump, kv_config, vllm_config  # noqa: E402
from implementation import kv_transfer_state  # noqa: E402
from implementation.base import KVConnectorMetadata  # noqa: E402
from implementation.forward_context import (  # noqa: E402
    ForwardContext,
    override_forward_context,
    set_forward_context,
)
from implementation.kv_connector_model_runner_mixin import (  # noqa: E402
    KVConnectorModelRunnerMixin,
)
from implementation.kv_transfer_utils import maybe_transfer_kv_layer  # noqa: E402

# ---------------- 部件1：真实调用序（4 层） ----------------
class TickMeta(KVConnectorMetadata):
    pass


@dataclass
class _Layer:
    kv_cache: Any = None


kv_transfer_state._KV_CONNECTOR_AGENT = None
cfg = vllm_config(
    kv_connector="ScriptConnector", kv_role="kv_consumer", module_path=MODULE_PATH
)
kv_transfer_state.ensure_kv_transfer_initialized(cfg, kv_config())
conn = kv_transfer_state.get_kv_transfer_group()
conn.events = []


@maybe_transfer_kv_layer
def attn_forward(layer_name: str, x):
    return x + 1


L = 4
layers = {f"l{i}": _Layer(torch.zeros(1, 2, 16, 8)) for i in range(L)}
fc = ForwardContext(
    no_compile_layers=layers,
    attn_metadata={f"l{i}": "md" for i in range(L)},
    slot_mapping={f"l{i}": torch.zeros(1) for i in range(L)},
)

from implementation.output import SchedulerOutput  # noqa: E402

out = SchedulerOutput(finished_req_ids=set())
out.kv_connector_metadata = TickMeta()
with set_forward_context(None, cfg):
    with KVConnectorModelRunnerMixin._get_kv_connector_output(out):
        with override_forward_context(fc):
            for i in range(L):
                attn_forward(f"l{i}", i)
real_seq = [e[0] for e in conn.events]
kv_transfer_state.ensure_kv_transfer_shutdown()

# ---------------- 部件2：时间账（离散事件模拟，虚拟拍） ----------------
T = 2   # 每层传输时长（拍）——链路按层序串行：layer i 就绪时刻 = (i+1)*T
C = 3   # 每层计算时长（拍）
ready = [(i + 1) * T for i in range(L)]          # [2, 4, 6, 8]

# 朴素：等全部层到齐（ready[-1]=8）才开始算——串行 = sum
naive = []
t = ready[-1]                                    # 全部传输完成才有第一层可算
for i in range(L):
    naive.append({"layer": i, "start": t, "end": t + C})
    t += C
naive_total = t                                  # 8 + 12 = 20

# 契约形态：start_load_kv 一次发起全部层，每层只等本层就绪与上一层算完
overlap = []
t = 0
for i in range(L):
    start = max(t, ready[i])
    overlap.append({"layer": i, "wait": start - t, "ready": ready[i],
                    "start": start, "end": start + C})
    t = start + C
overlap_total = t                                # 14

sum_T = L * T                                    # 8
sum_C = L * C                                    # 12

trace = {
    "mechanism": "m8 逐层重叠（wait_for_layer_load 只等本层）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "params": {
        "num_layers": L,
        "transfer_ticks_per_layer": T,
        "compute_ticks_per_layer": C,
        "时间单位": "虚拟拍（教学模型，非 GPU 实测；真实收益取决于传输/计算比）",
    },
    "real_call_sequence": real_seq,               # 真实运行：调用序
    "real_sequence_len": len(real_seq),
    "transfer_ready_ticks": ready,                # [2,4,6,8]
    "naive_schedule": naive,
    "naive_total_ticks": naive_total,             # 20 = sum(8) + sum(12)
    "naive_wait_before_first_compute": ready[-1], # 8：全部到齐才开工
    "overlap_schedule": overlap,
    "overlap_total_ticks": overlap_total,         # 14
    "sum_transfer_ticks": sum_T,                  # 8
    "sum_compute_ticks": sum_C,                   # 12
    "max_of_sums": max(sum_T, sum_C),             # 12
    "saved_ticks": naive_total - overlap_total,   # 6
    "only_layer0_waits_in_overlap": all(
        row["wait"] == 0 for row in overlap[1:]
    ),
    "formula": "端到端 ≈ max(Σ传输, Σ计算) + 首层传输 = max(8,12) + 2 = 14 ≠ Σ(8)+Σ(12)=20",
    "wait_for_save注": "wait_for_save 是强制同步点：不出这个栅栏，paged buffer 可能被"
                       "下一步覆写（base.py:L359-L367）——正确性优先于重叠极限。",
}
print(dump("m8", trace))
