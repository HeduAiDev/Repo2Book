# ch15 m22 partial→full 时序驱动：读者反馈③「为什么 cache_full_block 里面直接就有
# hash、什么时候算的都不知道；为什么还是 partial→full 的更新过程，不是
# num_cached_blocks >= num_full_blocks 就短路了吗？partial 什么时候写进去的」。
# 场景（block_size=64 > hash_block_size=16，块大于哈希粒度的 partial-hit 粒度配置）：
#   A 的 prompt 恰 48 token（落在 64-token 块内部）→
#   prefill 拍：allocate_slots → cache_blocks(48)：num_full_blocks=0 → 幂等闸短路，
#     满块零登记；但 FullAttentionManager.cache_blocks 覆写在 super() 之后继续
#     _cache_partial_tail_block(48) → cache_partial_block 注册块内条目 @48 ——这是
#     「partial 什么时候写进去」的答案：与满块写回同一次 cache_blocks 调用、短路
#     挡不住它（覆写先调 super 再补尾巴）。
#   decode 拍逐 token 49..63：每拍 append_output_token_ids 续算哈希（64 边界未到、
#     账本不动）；cache_blocks 的幂等闸仍短路；partial 登记幂等（already_cached）。
#   第 16 个生成 token 使 num_tokens=64：block_hashes 出现第 4 枚（hash[3]）→
#     下一拍 cache_blocks(64)：num_full_blocks=1 > num_cached_blocks=0 → 不短路 →
#     cache_full_blocks 走 L284 晋升：摘 @48 短条目、插 @64 满条目。
# 全程记录：block_hashes 长度、幂等闸判定、partial 登记动作、map 条目与覆盖边界。
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec)
from implementation.request import Request  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
import torch  # noqa: E402

kcu.init_none_hash(sha256)
HASHER16 = kcu.get_request_block_hasher(16, sha256)


def kv_config(specs, num_blocks=64):
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[],
                         kv_cache_groups=[KVCacheGroupSpec([f"l.{i}"], s)
                                          for i, s in enumerate(specs)])


sched = Scheduler(kv_config([
    FullAttentionSpec(block_size=64, num_kv_heads=2, head_size=8,
                      dtype=torch.float16),
    MambaSpec(block_size=64, shapes=((8, 8),), dtypes=(torch.float32,),
              mamba_cache_mode="align"),
], 64), max_model_len=512, scheduler_block_size=64, hash_block_size=16)
mgr = sched.kv_cache_manager
pool = mgr.block_pool
man = mgr.coordinator.single_type_managers[0]   # full 组管家（L779 覆写所在）

LOG = []


def map_entries():
    m = pool.cached_block_hash_to_block._cache
    out_e = []
    for blk in man.req_to_blocks.get("a", []):
        if blk.block_hash is not None:
            out_e.append({"block_id": blk.block_id,
                          "num_tokens": blk.block_hash_num_tokens,
                          "alias_keys": len(
                              pool.cached_block_hashes_by_block.get(
                                  blk.block_id, ()))})
    return {"map_size": len(m), "block_ledger": out_e}


def one_tick(req, num_new):
    """一拍 = 准入（首拍）+ 分配槽位（内部含写回 cache_blocks）。"""
    if req.num_computed_tokens == 0:
        blocks, hit, _ = sched.admission_lookup(req)
    else:
        blocks, hit = mgr.empty_kv_cache_blocks, 0
    out = mgr.allocate_slots(req, num_new, num_new_computed_tokens=hit,
                             new_computed_blocks=blocks)
    assert out is not None
    req.num_computed_tokens += num_new


req = Request("a", list(range(48)), block_hasher=HASHER16)

# --- prefill 拍：48 token 一次算完 ---
one_tick(req, 48)
LOG.append({"拍": "prefill（num_tokens=48）",
            "block_hashes_len": len(req.block_hashes),
            "num_full_blocks": 48 // 64, "num_cached_blocks": 0,
            "幂等闸": "0 >= 0 → 短路（满块零登记）",
            "partial 登记": "boundary=48（48//16*16）→ cache_partial_block 注册块内条目 @48",
            **map_entries()})

# --- decode 拍：逐 token 49..64，跨过 64 边界那一拍触发晋升 ---
for gen_tok in range(49, 65):
    req.append_output_token_ids(gen_tok)   # 采样后入账：顺手续算哈希
    pre_cached = man.num_cached_block.get("a", 0)   # 拍前满块进度账
    one_tick(req, 1)
    if gen_tok in (50, 63, 64):
        full = gen_tok // 64
        if gen_tok == 64:
            LOG.append({
                "拍": "decode（num_tokens=64，第 16 个生成 token）",
                "block_hashes_len": len(req.block_hashes),
                "num_full_blocks": full,
                "num_cached_blocks_拍前": pre_cached,
                "幂等闸": f"{pre_cached} >= {full}？否——跨过块边界，闸翻转放行",
                "partial→full": "cache_full_blocks L284：块 1 已有 @48 条目 → 摘短插长，"
                                "主哈希覆盖边界 48→64；随后 _cache_partial_tail_block "
                                "把 @48 作为别名补登回来（prompt 边界恒 48）——"
                                "一块挂主哈希+别名两条目（反向索引记账）",
                **map_entries()})
        else:
            LOG.append({"拍": f"decode（num_tokens={gen_tok}）",
                        "block_hashes_len": len(req.block_hashes),
                        "num_full_blocks": full,
                        "num_cached_blocks_拍前": pre_cached,
                        "幂等闸": f"{pre_cached} >= {full} → 短路（幂等）",
                        "partial 登记": "already_cached → 幂等，无新条目",
                        **map_entries()})

out = {
    "params": {"block_size": 64, "hash_block_size": 16,
               "prompt": "0..47 共 48 token（prefill 一拍算完）",
               "decode": "生成 token 49..64 逐拍",
               "config_note": "block 64 > hash 16：FullAttentionManager.cache_blocks "
                              "覆写在 super()（满块+幂等闸）之后继续 "
                              "_cache_partial_tail_block——短路的只是满块区间，"
                              "partial 尾登记不受它管"},
    "timeline": LOG,
    "final": {"block_hashes_len": len(req.block_hashes),
              "num_cached_block_ledger": man.num_cached_block.get("a"),
              **map_entries()},
    "answer": {
        "hash_when": "请求侧账本：构造时 update_block_hashes 算 @16/@32/@48 三枚"
                     "（request.py:L208-L209），append 第 64 个 token 时补第 4 枚"
                     "（L249-L265）——cache_full_blocks 从不算哈希，只消费 "
                     "request.block_hashes（block_pool.py:L240-L241 docstring 原话）",
        "partial_when": "与满块写回同一次 cache_blocks 调用：FullAttentionManager."
                        "cache_blocks = super().cache_blocks（幂等闸+满块登记）"
                        "+ _cache_partial_tail_block（块内尾边界），single_type:"
                        "L779-L819",
        "shortcircuit_truth": "幂等闸 num_cached_blocks >= num_full_blocks 只挡满块"
                              "区间登记；num_cached_block 只被满块登记推进——"
                              "token 跨过块边界那一刻 num_full_blocks 涨而账本没涨，"
                              "下一拍闸必然放行，晋升不是特例是常态",
    },
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m22.json"),
          "w", newline="\n", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
