# ch38 explainer 驱动脚手架：复用 tests/_kv_harness 的装配面（调度器半边+worker 半边
# 同进程、HostOffloadingWorker 同步 DMA 替身），驱动 implementation/ 精简版取 trace。
# 运行环境：host（Windows conda CPython 3.11.11、torch 2.11.0 CPU、无 CUDA/无 vllm 包）
# ——SEAM 见 implementation/impl-notes.md §3（HostStream/swap_blocks_batch/HostOffloadingWorker/
# fs io 的 O_BINARY）；DMA 三戒的 stream/event 时序 host 不可观察，以源码注释为准。
from __future__ import annotations

import json
import sys
from pathlib import Path

TRACES = Path(__file__).resolve().parent
CHAPTER = TRACES.parent.parent  # ch38-kv-pooling/
TESTS = CHAPTER / "tests"
IMPL = CHAPTER / "implementation"
for _p in (str(TESTS), str(IMPL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HOST_ENV = {
    "python": "3.11.11 (conda, Windows)",
    "torch": "2.11.0+cu128 (CPU 执行)",
    "numpy": "1.26.4",
    "zmq": "25.1.0",
    "cuda": False,
    "vllm_pkg": False,
    "impl": "instances/vllm/artifacts-v3/ch38-kv-pooling/implementation（精简版，69 tests passed）",
    "seams": [
        "HostStream（wait_stream/wait_event no-op）——DMA 三戒时序 host 不可观察，正文锚 gpu_worker.py 注释",
        "swap_blocks_batch = ctypes.memmove 逐描述符搬运（真源 C++ cache_kernels.cu 批量拷贝 kernel）",
        "HostOffloadingWorker：同一套 compute_sub_block_ptrs 描述符数学同步搬字节（真 CPUOffloadingWorker 需 CUDA）",
        "Windows 无 /dev/shm → SharedOffloadRegion 走 spec 的 no-mmap 回退分支；tiering 测试以 numpy 池替身",
        "MooncakeStore 查询走真 ZMQ RPC（LookupKeyClient 真身），对端 REP 服务按脚本回结果",
        "P2P session 机器（ZMQ/NIXL 实现体）不进精简版；协议消息类/三角色键/PYTHONHASHSEED 门 host 全覆盖",
    ],
}


def dump(name: str, obj: dict) -> None:
    obj = {"host_env": HOST_ENV, **obj}
    p = TRACES / name
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print(f"wrote {p}")


def ctx(req_id: str = "r1"):
    from vllm.v1.kv_offload.base import ReqContext

    return ReqContext(req_id=req_id)


def key(seed: int, group: int = 0):
    from vllm.v1.kv_offload.base import make_offload_key

    return make_offload_key(seed.to_bytes(32, "big"), group)
