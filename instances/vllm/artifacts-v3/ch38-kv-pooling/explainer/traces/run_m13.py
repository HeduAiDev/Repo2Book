# m13 P2P 层：三角色键解析 + PYTHONHASHSEED 硬门 + 对称 peer 查询语义 + 协议消息校验。
import os

from _driver_common import TESTS, ctx, dump, key

import sys

sys.path.insert(0, str(TESTS))
import numpy as np  # noqa: E402
from _kv_harness import make_kv_config, make_vllm_config  # noqa: E402

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (  # noqa: E402
    build_offloading_config,
)
from vllm.v1.kv_offload.base import Locality, LookupResult, Medium, ReqContext  # noqa: E402
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec  # noqa: E402
from vllm.v1.kv_offload.tiering.p2p.manager import (  # noqa: E402
    P2PDestInfo,
    P2PSourceInfo,
    _parse_dest,
    _parse_source,
)
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (  # noqa: E402
    FetchMsg,
    LookupMsg,
    LookupRespMsg,
    TransferDoneMsg,
)

# ── ① 三角色键解析（kv_transfer_params 由编排层 router/EPP 驱动）──
consumer = {
    "remote_kv_source": {
        "kv_request_id": "kq-1",
        "remote_host": "10.0.0.2",
        "remote_port": 5710,
    }
}
src = _parse_source(consumer)
pd_consumer = {
    "remote_prefiller": {
        "kv_request_id": "kq-2",
        "remote_host": "10.0.0.3",
        "remote_port": 5710,
    }
}
src_pd = _parse_source(pd_consumer)
producer = {"remote_decoder": {"kv_request_id": "kq-3"}}
dst = _parse_dest(producer)
role_keys = {
    "remote_kv_source": {
        "kv_request_id": src.kv_request_id,
        "peer_id": src.peer_id,
        "do_probe": src.do_probe,
    },
    "remote_prefiller": {
        "kv_request_id": src_pd.kv_request_id,
        "do_probe": src_pd.do_probe,
        "note": "P/D 模式跳过 Lookup 直接 Fetch（对端 prefiller 有全部块）",
    },
    "remote_decoder": {"kv_request_id": dst.kv_request_id},
    "no_keys": {"source": str(_parse_source({})), "dest": str(_parse_dest({}))},
    "symmetry": "无固定 P/D 角色：同一实例对不同请求可当 consumer 也可当 producer",
}

# ── ② PYTHONHASHSEED 硬门：未设拒启 ──
cfg = make_kv_config()
vcfg = make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20})
spec = CPUOffloadingSpec(build_offloading_config(vcfg, cfg))
pool = memoryview(np.zeros((2, 64), dtype=np.uint8))
from vllm.v1.kv_offload.tiering.p2p.manager import P2PSecondaryTierManager  # noqa: E402

seed_gate = {}
saved = os.environ.pop("PYTHONHASHSEED", None)
try:
    try:
        P2PSecondaryTierManager(spec, pool, tier_type="p2p")
        seed_gate["unset"] = "constructed"
    except ValueError as e:
        seed_gate["unset"] = f"ValueError: {e}"
finally:
    if saved is not None:
        os.environ["PYTHONHASHSEED"] = saved

# ── ③ 设了种子 → 可建；PD consumer 查询立即 HIT / 普通请求 MISS（无对端退化分支）──
os.environ["PYTHONHASHSEED"] = "0"
try:
    tier = P2PSecondaryTierManager(spec, pool, tier_type="p2p")
    pd_ctx = ReqContext(
        req_id="d1",
        kv_transfer_params={
            "remote_prefiller": {
                "kv_request_id": "kq",
                "remote_host": "h",
                "remote_port": 1,
            }
        },
    )
    tier.on_new_request(pd_ctx)
    lookup_pd = tier.lookup(key(0), pd_ctx)
    lookup_plain = tier.lookup(key(0), ctx())
finally:
    os.environ.pop("PYTHONHASHSEED", None)
    if saved is not None:
        os.environ["PYTHONHASHSEED"] = saved

lookup_case = {
    "seed_set": "PYTHONHASHSEED=0",
    "pd_consumer_lookup": lookup_pd.name,
    "plain_request_lookup": lookup_plain.name,
    "note": "PD consumer 视对端 prefiller 拥有全部块→HIT；普通请求无对端 session→MISS（无对端退化分支）",
}

# ── ④ 线协议消息校验（Lookup→Resp→Fetch→NIXL WRITE→TransferDone）──
protocol = {}
LookupMsg.validate({"kv_request_id": "k", "keys": [b"a"], "round_seq": 0})
protocol["lookup_ok"] = True
try:
    LookupMsg.validate({"kv_request_id": "k", "keys": "notalist", "round_seq": 0})
    protocol["lookup_bad"] = "validated"
except ValueError:
    protocol["lookup_bad"] = "ValueError"
FetchMsg.validate({"kv_request_id": "k", "keys": [b"a"], "block_indexes": [3], "round_seq": 1})
try:
    FetchMsg.validate({"kv_request_id": "k", "keys": [b"a", b"b"], "block_indexes": [1], "round_seq": 0})
    protocol["fetch_len_mismatch"] = "validated"
except ValueError as e:
    protocol["fetch_len_mismatch"] = f"ValueError: {e}"
try:
    FetchMsg.validate({"kv_request_id": "k", "keys": [b"a"], "block_indexes": [-1], "round_seq": 0})
    protocol["fetch_negative_index"] = "validated"
except ValueError:
    protocol["fetch_negative_index"] = "ValueError"
TransferDoneMsg.validate({"kv_request_id": "k", "success": True, "round_seq": 0})
try:
    TransferDoneMsg.validate({"kv_request_id": "k", "success": "yes", "round_seq": 0})
    protocol["done_bad_success"] = "validated"
except ValueError:
    protocol["done_bad_success"] = "ValueError"
LookupRespMsg.validate({"kv_request_id": "k", "keys": [b"a"], "hits": [True]})
try:
    LookupRespMsg.validate({"kv_request_id": "k", "keys": [b"a"], "hits": []})
    protocol["resp_len_mismatch"] = "validated"
except ValueError as e:
    protocol["resp_len_mismatch"] = f"ValueError: {e}"
protocol["anchor"] = "p2p/session/protocol.py:L178-L218"
protocol["watchdog"] = "_UNBOUND_STORE_TIMEOUT_S=60s 收死对端缓冲（session 机器不进精简版——删除项 9）"

dump(
    "m13.json",
    {
        "role_keys": role_keys,
        "pythonhashseed_gate": seed_gate,
        "lookup_symmetric_semantics": lookup_case,
        "protocol_messages": protocol,
    },
)
