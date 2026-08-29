# ch16 m7 worker 一拍生命周期：_get_kv_connector_output 包住 _model_forward——
# bind → start_load_kv →（前向内逐层 wait/save）→ finally：wait_for_save →
# get_finished → get_block_ids_with_load_errors → clear
# （mixin:L76-L112 + kv_transfer_utils.py:L37-L59 装饰器 + gpu_model_runner.py:L4420-L4456 挂点）。
# 场景：2 层假模型跑一个真实前向拍；finished_req_ids={"dead"}。
import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from _harness import MODULE_PATH, dump, kv_config, vllm_config  # noqa: E402
from implementation import kv_transfer_state  # noqa: E402
from implementation.forward_context import (  # noqa: E402
    ForwardContext,
    override_forward_context,
    set_forward_context,
)
from implementation.kv_connector_model_runner_mixin import (  # noqa: E402
    KVConnectorModelRunnerMixin,
)
from implementation.kv_transfer_utils import maybe_transfer_kv_layer  # noqa: E402
from implementation.output import SchedulerOutput  # noqa: E402
from implementation.base import KVConnectorMetadata  # noqa: E402


class TickMeta(KVConnectorMetadata):
    pass


@dataclass
class _Layer:
    kv_cache: Any = None


def make_out(finished: set) -> SchedulerOutput:
    out = SchedulerOutput(finished_req_ids=finished)
    out.kv_connector_metadata = TickMeta()
    return out


# ---- 装配 worker 侧全局 agent（ensure_kv_transfer_initialized，role=WORKER）----
kv_transfer_state._KV_CONNECTOR_AGENT = None
cfg = vllm_config(
    kv_connector="ScriptConnector",
    kv_role="kv_consumer",
    module_path=MODULE_PATH,
)
kv_transfer_state.ensure_kv_transfer_initialized(cfg, kv_config())
conn = kv_transfer_state.get_kv_transfer_group()
conn.events = []

# ---- 两层注意力：装饰器长在层上（wait 层前 / save 层后）----
calls = []


@maybe_transfer_kv_layer
def attn_forward(layer_name: str, x):
    calls.append(("attn", layer_name))
    return x + 1


layers = {"l0": _Layer(torch.zeros(1, 2, 16, 8)), "l1": _Layer(torch.zeros(1, 2, 16, 8))}
fc = ForwardContext(
    no_compile_layers=layers,
    attn_metadata={"l0": "md", "l1": "md"},
    slot_mapping={"l0": torch.zeros(1), "l1": torch.zeros(1)},
)

out = make_out({"dead"})
with set_forward_context(None, cfg):
    with KVConnectorModelRunnerMixin._get_kv_connector_output(out) as ko:
        with override_forward_context(fc):   # 模型前向（真实挂点=包住 _model_forward）
            attn_forward("l0", 1)
            attn_forward("l1", 1)

event_seq = [e[0] if e[0] != "query" else "query" for e in conn.events]
detail = [
    (e[0], e[1] if len(e) > 1 else None) for e in conn.events
]

trace = {
    "mechanism": "m7 worker 一拍生命周期",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "params": {
        "num_layers": 2,
        "finished_req_ids": ["dead"],
        "wait_for_save": True,
    },
    "event_sequence": [e[0] for e in conn.events],
    "event_detail": detail,
    "sequence_len": len(conn.events),        # 8
    "attn_calls": calls,
    "metadata_cleared_after_tick": not conn.has_connector_metadata(),
    "get_finished_ids": detail[-1][1],       # ["dead"]
    "lifecycle": [
        {"phase": 1, "call": "bind_connector_metadata", "meaning": "收调度器侧的不透明计划"},
        {"phase": 2, "call": "start_load_kv", "meaning": "异步发起全部层的加载（前向开始前）"},
        {"phase": 3, "call": "wait_for_layer_load(l0)→attn(l0)→save_kv_layer(l0)", "meaning": "层前等本层、层后存出"},
        {"phase": 4, "call": "wait_for_layer_load(l1)→attn(l1)→save_kv_layer(l1)", "meaning": "逐层重叠"},
        {"phase": 5, "call": "wait_for_save", "meaning": "强制同步点——防 paged buffer 被下一步覆写"},
        {"phase": 6, "call": "get_finished({dead})", "meaning": "上报异步收/发完成"},
        {"phase": 7, "call": "get_block_ids_with_load_errors", "meaning": "失败块上报（空集）"},
        {"phase": 8, "call": "clear_connector_metadata", "meaning": "一拍收尾"},
    ],
}

# ---- 对照：无 token 步（kv_connector_no_forward）也走收发，但 wait_for_save=False
conn.events = []
mro = KVConnectorModelRunnerMixin.kv_connector_no_forward(make_out(set()), cfg)
no_fwd_events = [e[0] for e in conn.events]
trace["no_forward_variant"] = {
    "event_sequence": no_fwd_events,
    "wait_for_save_present": "wait_for_save" in no_fwd_events,   # False
    "start_load_kv_present": "start_load_kv" in no_fwd_events,   # True
    "kv_connector_output_is_none": mro.kv_connector_output is None,
    "note": "无 token 也要收发（异步传输与 running 可不相交）；无前向=无覆写风险，跳过 wait_for_save（mixin:L36-L48）",
}
kv_transfer_state.ensure_kv_transfer_shutdown()
print(dump("m7", trace))
