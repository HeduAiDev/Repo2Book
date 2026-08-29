# ch16 m2 外部命中查询：None=『稍后再问』进 skipped 不堵队头；下一拍有答案
# 即正常调度（scheduler.py:L744-L789；base.py:L465-L498 的 None 语义）。
# 场景：64-token prompt；步1 connector 答 None；步2 答 (32, False) 同步命中。
# 幕3：ExampleConnector 磁盘版（example_connector.py:L251-L298）——worker 存盘
# → 调度器侧查同一 prompt 命中（文件系统=外部缓存的最小样例）。
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from _harness import (  # noqa: E402
    HASHER16,
    dump,
    kv_config,
    make_request,
    script_scheduler,
    vllm_config,
)
from implementation.base import KVConnectorRole  # noqa: E402
from implementation.example_connector import (  # noqa: E402
    ExampleConnector,
    align_to_block_size,
)
from implementation.kv_transfer import KVTransferConfig  # noqa: E402

s = script_scheduler([(None, False)])
req = make_request("r1", range(64))
s.add_request(req)
out1 = s.schedule()

step1 = {
    "step": 1,
    "connector_answer": "None（稍后再问）",
    "query_seen_by_connector": 0,          # 入参 num_computed_tokens（block_aligned_local）
    "r1_status": req.status.name,
    "r1_in_skipped_waiting": req in list(s.skipped_waiting),
    "r1_in_waiting": req in list(s.waiting),
    "r1_scheduled_tokens": out1.num_scheduled_tokens.get("r1", 0),
    "r1_blocks_held": 0,                    # 没占任何块
}

# 步2：connector 已有答案（32, False）→ 本地 0 + 外部 32 → 本拍即算 32
s.connector.ext_answer = [(32, False)]
out2 = s.schedule()
alloc_ev = [
    e for e in s.connector.events if e[0] == "update_state_after_alloc"
]
step2 = {
    "step": 2,
    "connector_answer": "(32, False) 同步命中",
    "query_seen_by_connector": 0,          # 本地命中 0 → block_aligned_local=0
    "local_hit_tokens": 0,
    "external_hit_tokens": 32,
    "num_computed_tokens_total": 0 + 32,
    "r1_scheduled_tokens": out2.num_scheduled_tokens["r1"],   # 64−32
    "prompt_tokens": 64,
    "r1_status": req.status.name,
    "update_state_after_alloc_events": alloc_ev,               # ("r1", 32)
    "connector_metadata_built": isinstance(
        out2.kv_connector_metadata, type(out2.kv_connector_metadata)
    ),
}

# ---- 幕3：ExampleConnector 磁盘版——外部缓存=文件系统的最小样例 ----
STORAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".kv_store_tmp")
if os.path.exists(STORAGE):
    shutil.rmtree(STORAGE)
cfg = vllm_config(kv_connector="ExampleConnector", kv_role="kv_consumer",
                  extra={"shared_storage_path": STORAGE})
sched_side = ExampleConnector(cfg, KVConnectorRole.SCHEDULER, kv_config())
worker_side = ExampleConnector(cfg, KVConnectorRole.WORKER, kv_config())
tokens = list(range(40))
from implementation.example_connector import ExampleConnectorMetadata  # noqa: E402

meta = ExampleConnectorMetadata()
meta.add_request(tokens, [1, 2], 16, True, [])
worker_side.bind_connector_metadata(meta)
worker_side.save_kv_layer("layer.0", torch.randn(3, 2, 16, 8), None)
r9 = make_request("r9", tokens, hasher=HASHER16)
got = sched_side.get_num_new_matched_tokens(r9, 0)
step3 = {
    "step": 3,
    "connector_answer": f"({got[0]}, {got[1]}) 磁盘命中",
    "prompt_tokens": 40,
    "num_tokens_to_check_block_aligned": align_to_block_size(40 - 1, 16),  # 32
    "worker_saved_blocks": 2,
    "query_seen_by_connector": 0,
    "external_hit_tokens": got[0],          # 32 − 0
    "note": "worker 先存盘（save_kv_layer → 磁盘 safetensors）；调度器侧查同一 "
            "prompt → 命中 32：文件系统就是外部缓存（example_connector.py:L251-L286）",
}
shutil.rmtree(STORAGE, ignore_errors=True)

trace = {
    "mechanism": "m2 外部命中查询 get_num_new_matched_tokens",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "params": {
        "block_size": 16,
        "prompt_tokens": 64,
        "pool_blocks": 64,
        "kv_role": "kv_consumer",
        "场景": "步1 connector 答 None；步2 答 (32, False) 同步命中；"
              "幕3 ExampleConnector 磁盘版（40-token prompt）",
    },
    "steps": [step1, step2, step3],
    "query_call_log": [
        {"call": 1, "answer": "None"},
        {"call": 2, "answer": [32, False]},
    ],
}
print(dump("m2", trace))
