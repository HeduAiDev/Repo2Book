# ch16 m14 worker 直写池：register_kv_caches 注册池张量后，connector 按
# block_ids + slot_mapping（block_id×block_size+offset）直接读写 GPU 内存——
# ExampleConnector 的 extract（save 方向）/inject（load 方向）是最小参考实现，
# 外部缓存=磁盘文件（example_connector.py:L122-L149/L221-L249）。
# 场景：4 块池张量 [4, 2, 16, 8]；40-token prompt 按块对齐取前 32 token，
# 块表 [1, 2]（最后 1 token 留给重算要 logits——align_to_block_size(len-1)）。
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from _harness import BLOCK, dump, kv_config, vllm_config  # noqa: E402
from implementation.base import KVConnectorRole  # noqa: E402
from implementation.example_connector import (  # noqa: E402
    ExampleConnector,
    ExampleConnectorMetadata,
    align_to_block_size,
)

HERE = os.path.dirname(os.path.abspath(__file__))
STORAGE = os.path.join(HERE, ".kv_store_tmp")
if os.path.exists(STORAGE):
    shutil.rmtree(STORAGE)


def store_meta(tokens, block_ids, is_store):
    meta = ExampleConnectorMetadata()
    meta.add_request(tokens, block_ids, BLOCK, is_store, [])
    return meta


cfg = vllm_config(
    kv_connector="ExampleConnector",
    kv_role="kv_consumer",
    extra={"shared_storage_path": STORAGE},
)
conn = ExampleConnector(cfg, KVConnectorRole.WORKER, kv_config())

tokens = list(range(40))
BLOCK_IDS = [1, 2]
meta = store_meta(tokens, BLOCK_IDS, is_store=True)
conn.bind_connector_metadata(meta)
slot = meta.requests[0].slot_mapping

# paged buffer：每槽塞一个可验证的指纹值 = slot 编号本身
paged = torch.zeros(4, 2, BLOCK, 8)
for s_ in slot.tolist():
    paged[s_ // BLOCK, :, s_ % BLOCK] = float(s_)

# ---- save 方向：extract 按 slot 抽出该层该请求的 KV，落盘 ----
conn.save_kv_layer("layer.0", paged, None)
from safetensors.torch import load_file  # noqa: E402

fname = conn._generate_filename_debug(
    "layer.0", meta.requests[0].token_ids, []
)
saved = load_file(fname)["kv_cache"]

# ---- load 方向：inject 按 slot 写回（先清空池模拟重分配后的脏块） ----
load_meta = store_meta(tokens, BLOCK_IDS, is_store=False)
conn.bind_connector_metadata(load_meta)


class _Layer:  # start_load_kv 经 forward_context 取该层池张量
    kv_cache = paged


class _FC:
    no_compile_layers = {"layer.0": _Layer}
    attn_metadata = {"layer.0": "md"}


paged.zero_()
conn.start_load_kv(_FC())
restored_ok = True
for s_ in slot.tolist():
    got = paged[s_ // BLOCK, :, s_ % BLOCK]
    if not torch.equal(got, torch.full_like(got, float(s_))):
        restored_ok = False
        break

untouched_probe_slot = 5  # 块 0 的槽 5：不在请求块表 → 注入后仍为 0
untouched_stays_zero = bool(
    torch.equal(paged[0, :, 5], torch.zeros_like(paged[0, :, 5]))
)

block_idxs = (slot // BLOCK).tolist()
offsets = (slot % BLOCK).tolist()

trace = {
    "mechanism": "m14 worker 直写池（inject/extract 的 slot 寻址）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "params": {
        "block_size": BLOCK,
        "paged_buffer_shape": [4, 2, 16, 8],   # [num_pages, 2, page_size, head_dim]
        "prompt_tokens": 40,
        "addressed_tokens": align_to_block_size(40 - 1, BLOCK),  # 32
        "block_ids": BLOCK_IDS,                 # 落在块 1、2
        "对齐注": "ExampleConnector 只寻址 align_to_block_size(len-1)=32 个 token"
                 "（最后 1 token 留给重算要 logits；example_connector.py:L265-L286）",
    },
    "slot_addressing": {
        "formula": "slot = block_id×16 + offset",
        "num_slots": len(slot),                     # 32
        "slots_head4": slot[:4].tolist(),           # [16,17,18,19]（块 1 开头）
        "slots_tail4": slot[-4:].tolist(),          # [44,45,46,47]（块 2 结尾）
        "first_slot_of_block1": slot[0].item(),     # 16 = 1×16+0
        "last_slot_of_block1": slot[15].item(),     # 31 = 1×16+15
        "first_slot_of_block2": slot[16].item(),    # 32 = 2×16+0
        "last_slot_of_block2": slot[31].item(),     # 47 = 2×16+15
        "block_idxs_unique": sorted(set(block_idxs)),    # [1,2]
        "offsets_range": [min(offsets), max(offsets)],  # [0,15]
    },
    "save_direction": {
        "extract": "layer[block_idxs, :, offsets]——按 slot 从 paged buffer 抽出",
        "saved_shape": list(saved.shape),       # [32, 2, 8]：32 个 slot
        "saved_values_match_slots": bool(
            torch.equal(saved[:, 0, 0].cpu(),
                       slot.to(torch.float32).cpu())
        ),
        "storage": "磁盘 safetensors 文件（外部缓存=文件系统）",
    },
    "load_direction": {
        "inject": "dst[block_idxs, :, offsets] = src——按 slot 直写池内存",
        "buffer_zeroed_before_inject": True,
        "round_trip_equal": restored_ok,
        "untouched_slot_stays_zero": untouched_stays_zero,
        "untouched_probe_slot": untouched_probe_slot,
    },
    "意义": "GPU 侧只认 block_id（ch13 槽位恒等式的延续）；connector 拿到计划里的 "
           "block_ids + 注册的池张量，就能在 worker 进程直写 KV 内存——不需要 "
           "经过任何 vLLM 中间层。",
}
print(dump("m14", trace))
shutil.rmtree(STORAGE, ignore_errors=True)
