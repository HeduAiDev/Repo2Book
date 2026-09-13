"""Driver for s1 (需块预测慢路径完整公式：三参六行账 + fast-path 开关的武装时刻)
— host run against the ch13 companion.

三个数字例（对齐 dossier supplement.A_block_accounting A1/A2）：
  E2  全注意力 + 3 块前缀命中（48 token）：冷命中（ref_cnt==0 躺自由队列）
      → num_new = max(7−3, 0) = 4、可驱逐 3 → 预测 7 = 自由队列净减量
      （touch 3 + get_new_blocks(4)；热命中 ref_cnt≥1 → 可驱逐 0 → 预测 4）。
  E3  滑窗 W=16 + connector 外部 32 token：total_computed=64 →
      skipped_tokens = max(0, 64−16+1) = 49 → skipped_blocks = 49//16 = 3
      （floor：第 4 块跨 48/49 边界、一角在窗内须留实体）；命中 2 块全落跳段
      → 剔除后 evictable=0 → num_new = max(7−max(3, 2), 0) = 4 → 预测 4。
  A1  fast-path 开关 = num_cached_block 记账位：短请求（总长 < 16）写回幂等闸
      0>=0 早退、永不武装；100-token 收尾写回 6 个满块武装；武装后 decode
      走 fast-path（101→0、112→0、113→1，max(cdiv−已持, 0)）。

口径说明（与 m5 的 f 行同款）：精简版删了滑窗子类与 partial 命中，E3 的
get_num_skipped_tokens 由 driver 按 pin 源码公式（single_type_kv_cache_
manager.py:L1057-L1083，max(0, T − W + 1)）在子类里逐字覆写；预测器主体
（六行账）仍是 impl 实跑。E2/E3 的命中块由 get_new_blocks + free_blocks
制造（冷）或只 get 不 free 制造（热），与 m5 f 行同款。
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from implementation.kv_cache_manager import KVCacheManager  # noqa: E402
from implementation.request import Request, RequestStatus  # noqa: E402

BLOCK_SIZE = 16
LAYER = "model.layers.0.self_attn.attn"
SWA_W = 16  # E3 的滑窗宽度（token）


def make_manager(num_blocks: int = 20) -> KVCacheManager:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=8, head_size=128, dtype=torch.float16
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[LAYER])
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[LAYER], kv_cache_spec=spec)],
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=512,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )


def make_request(req_id: str, n: int) -> Request:
    req = Request(request_id=req_id, prompt_token_ids=list(range(n)))
    req.status = RequestStatus.WAITING
    return req


def predict(single, req_id, num_tokens, new_computed_blocks,
            total_computed_tokens, num_local_computed_tokens):
    return single.get_num_blocks_to_allocate(
        request_id=req_id,
        num_tokens=num_tokens,
        new_computed_blocks=new_computed_blocks,
        total_computed_tokens=total_computed_tokens,
        num_local_computed_tokens=num_local_computed_tokens,
        num_tokens_main_model=num_tokens,
    )


def main():
    mgr = make_manager()
    single = mgr.coordinator.single_type_managers[0]

    # ---- E2：全注意力 + 3 块命中（48 token），无外部、无投机 ----
    # 冷命中：取 3 块再归还 -> ref_cnt==0、躺自由队列（驱逐候选）
    hit3_cold = mgr.block_pool.get_new_blocks(3)
    mgr.block_pool.free_blocks(hit3_cold)
    assert all(b.ref_cnt == 0 for b in hit3_cold)
    free_before = mgr.block_pool.get_num_free_blocks()
    pred_cold = predict(single, "e2", 100, hit3_cold, 48, 48)
    # 分配侧对账：touch 3 块（离开自由队列）+ get_new_blocks(4) -> 净减 7 = 预测
    mgr.block_pool.touch(hit3_cold)
    new4 = mgr.block_pool.get_new_blocks(4)
    free_after = mgr.block_pool.get_num_free_blocks()
    net_delta = free_before - free_after
    assert pred_cold == 7 and net_delta == 7 and len(new4) == 4

    # 热命中：只取不还 -> ref_cnt>=1、不在自由队列 -> 可驱逐 0
    hit3_hot = mgr.block_pool.get_new_blocks(3)
    assert all(b.ref_cnt == 1 for b in hit3_hot)
    pred_hot = predict(single, "e2-hot", 100, hit3_hot, 48, 48)
    assert pred_hot == 4

    # ---- E3：滑窗 W=16 + 外部 32 token（total_computed=64，本拍再算 36）----
    # 精简版删滑窗子类：按 pin 公式 max(0, T - W + 1)（L1057-L1083）覆写
    swa_cls = type("SWAOverride", (type(single),), {
        "get_num_skipped_tokens": lambda self, n: max(0, n - SWA_W + 1),
    })
    saved_cls = single.__class__
    single.__class__ = swa_cls
    try:
        hit2 = mgr.block_pool.get_new_blocks(2)
        mgr.block_pool.free_blocks(hit2)
        skipped_tokens = single.get_num_skipped_tokens(64)   # 49
        skipped_blocks = skipped_tokens // BLOCK_SIZE          # 3（floor）
        pred_e3 = predict(single, "e3", 100, hit2, 64, 32)
    finally:
        single.__class__ = saved_cls
    assert skipped_tokens == 49 and skipped_blocks == 3 and pred_e3 == 4

    # ---- A1：fast-path 开关（num_cached_block 记账位）的武装时刻 ----
    m3 = make_manager()
    s3 = m3.coordinator.single_type_managers[0]
    # 短请求：总长 5..15 时 num_full_blocks 恒 0，幂等闸 0>=0 早退、永不武装
    req_grow = make_request("arm-grow", 5)
    first_armed_len = None
    for length in range(5, 17):
        s3.cache_blocks(req_grow, length)
        if "arm-grow" in s3.num_cached_block:
            first_armed_len = length
            break
    assert first_armed_len == 16
    short_value = s3.num_cached_block.get("arm-grow")

    # 100-token：首排后收尾写回 6 个满块（100//16）武装；decode 走 fast-path
    req100 = make_request("arm-100", 100)
    m3.allocate_slots(req100, 100)
    s3.cache_blocks(req100, 100)
    armed_value = s3.num_cached_block.get("arm-100")
    assert armed_value == 6
    req100.status = RequestStatus.RUNNING
    fp = [predict(s3, "arm-100", n, [], n, n) for n in (101, 112, 113)]
    assert fp == [0, 0, 1]

    out = {
        "driver": "run_s1_slow_path.py",
        "mechanism": "s1 需块预测慢路径完整公式（single_type_kv_cache_manager.py:L178-L230；A1 fast-path 开关 L194-L200/L282/L289/L445-L448/L477）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版",
        "provenance": "dossier supplement.A_block_accounting A1/A2（读者反馈回修 2026-09-13）",
        "config": {"block_size": BLOCK_SIZE, "pool_blocks": 20, "swa_window_W": SWA_W},
        "e2_full_attention_hit": {
            "setup": "prompt=100，前缀命中 3 块（48 token），无外部、无投机；慢路径六行账",
            "num_tokens_need_slot": 100,
            "num_required_blocks_cdiv": 7,
            "num_req_blocks": 0,
            "len_new_computed_blocks": 3,
            "num_local_computed_blocks": "len(命中) + 已持 = 3 + 0 = 3",
            "num_skipped_blocks": 0,
            "num_new_blocks": "max(7 - max(0, 3), 0) = 4",
            "cold_hit_ref_cnt0": {"num_evictable_blocks": 3, "predicted": 7,
                                  "formula": "4 新块 + 3 可驱逐 = 7"},
            "hot_hit_ref_cnt_ge1": {"num_evictable_blocks": 0, "predicted": 4,
                                    "formula": "共享得越热，容量越省"},
            "free_queue_net_delta": {
                "free_before": free_before, "free_after": free_after,
                "touched": 3, "get_new_blocks": 4, "net": net_delta,
                "equals_predicted": net_delta == pred_cold,
                "note": "touch(3) 从自由队列中间摘走 + get_new_blocks(4) 队头取 -> 净减 7",
            },
        },
        "e3_swa_skip_plus_external": {
            "setup": "滑窗 W=16；本地命中 2 块（32 token）+ connector 外部 32 token -> total_computed=64；本拍再算 36 -> num_tokens=100",
            "num_tokens_need_slot": 100,
            "num_required_blocks_cdiv": 7,
            "num_req_blocks": 0,
            "len_new_computed_blocks": 2,
            "num_local_computed_blocks": "2 + 0 = 2",
            "total_computed_tokens": 64,
            "num_skipped_tokens": skipped_tokens,
            "num_skipped_blocks": skipped_blocks,
            "skip_floor_note": "49//16=3 取 floor 不取 cdiv：第 4 块跨 48/49 窗边界、token 49..63 仍在窗内必须留实体",
            "num_new_blocks": "max(7 - max(3, 2), 0) = 4（max 落在跳段支）",
            "num_skipped_new_computed_blocks": "max(0, 3 - 0) = 3 -> 命中 2 块全落跳段前缀、剔除后剩 0",
            "num_evictable_blocks": 0,
            "predicted": pred_e3,
            "override_note": "精简版删滑窗子类：get_num_skipped_tokens 由 driver 按 pin 公式（single_type_kv_cache_manager.py:L1057-L1083，max(0, T - W + 1)）覆写；预测器主体（六行账）为 impl 实跑。分配侧对账（null 占位 3 + 外部段 get_new_blocks(1) + 新块 3 = 4）是代码走读推演：allocate_external_computed_blocks 在精简版已删（→ ch16）",
        },
        "a1_fastpath_gate_arming": {
            "short_request": {
                "setup": "block_size=16，开缓存；prompt=5",
                "grow_lengths": "总长 5..15 的每个收尾拍：num_full_blocks = L//16 = 0",
                "idempotent_gate": "num_cached_blocks(0) >= num_full_blocks(0) -> 早退，账位不写",
                "first_armed_len": first_armed_len,
                "armed_value_at_16": short_value,
                "note": "总长到 16 才写回第 1 个满块（num_full_blocks=1）武装",
            },
            "prompt100": {
                "first_prefill_predicted": 7,
                "num_tokens_to_cache": 100,
                "num_full_blocks_written_back": armed_value,
                "decode_fastpath": [
                    {"num_tokens": 101, "cdiv": 7, "num_req_blocks": 7, "predicted": 0},
                    {"num_tokens": 112, "cdiv": 7, "num_req_blocks": 7, "predicted": 0},
                    {"num_tokens": 113, "cdiv": 8, "num_req_blocks": 7, "predicted": 1},
                ],
                "note": "cache_blocks 的幂等闸与 L477 账位推进为 impl 实跑（哈希登记在精简版已删 → ch15）；每 16 个 token 多要一块",
            },
        },
    }

    dst = Path(__file__).resolve().parent / "s1_slow_path.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps({
        "pred_cold": pred_cold, "pred_hot": pred_hot, "pred_e3": pred_e3,
        "net_delta": net_delta, "skipped_tokens": skipped_tokens,
        "skipped_blocks": skipped_blocks, "first_armed_len": first_armed_len,
        "armed_value": armed_value, "fastpath_decode": fp,
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
