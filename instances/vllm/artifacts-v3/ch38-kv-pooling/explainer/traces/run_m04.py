# m4 CPU 池准入与驱逐：store_threshold 频次过滤→去重→需逐数 vs 可逐数→LRU evict(protected)→
# ref_cnt -1→0 翻转；事件 BlockStored/BlockRemoved。
from _driver_common import ctx, dump, key

from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


def mgr(num_blocks=4, store_threshold=1, policy="lru"):
    return CPUOffloadingManager(
        num_blocks=num_blocks,
        cache_policy=policy,
        enable_events=True,
        store_threshold=store_threshold,
    )


def state(m):
    return {
        "free": m._get_num_free_blocks(),
        "evictable": m._num_evictable_cache_blocks,
        "write_pending": m._num_write_pending_blocks,
        "allocated": m._num_allocated_blocks,
    }


# ── ① 单块生命周期四态：MISS→写入中(HIT_PENDING)→HIT→去重 ──
m = mgr(num_blocks=4)
k0 = key(0)
lifecycle = {
    "lookup_before_store": m.lookup(k0, ctx()).name,
    "state_after_store_pending": None,
    "lookup_while_write_in_flight": None,
    "ref_cnt_while_pending": None,
    "lookup_after_complete_store": None,
    "ref_cnt_after_complete": None,
    "re_store_dedup_keys_to_store": None,
}
out = m.prepare_store([k0], ctx())
lifecycle["prepare_store_keys"] = [int.from_bytes(k0[:32], "big")]
lifecycle["state_after_store_pending"] = state(m)
lifecycle["lookup_while_write_in_flight"] = m.lookup(k0, ctx()).name
lifecycle["ref_cnt_while_pending"] = m._policy.get(k0).ref_cnt
lifecycle["state_note"] = "ref_cnt=-1 语义=写入中（complete_store 才翻 0 可读可逐）"
m.complete_store([k0], ctx())
lifecycle["lookup_after_complete_store"] = m.lookup(k0, ctx()).name
lifecycle["ref_cnt_after_complete"] = m._policy.get(k0).ref_cnt
lifecycle["state_after_complete"] = state(m)
out2 = m.prepare_store([k0], ctx())
lifecycle["re_store_dedup_keys_to_store"] = len(out2.keys_to_store)

# ── ② LRU 驱逐次序 + touch 刷新鲜 ──
m = mgr(num_blocks=3)
keys = [key(i) for i in range(3)]
for k in keys:
    m.prepare_store([k], ctx())
m.complete_store(keys, ctx())
m.touch([key(0)], ctx())  # k0 刷新鲜 → k1 变最旧
out = m.prepare_store([key(9)], ctx())
lru_case = {
    "pool_num_blocks": 3,
    "stored_keys": [0, 1, 2],
    "touch_refreshed_key": 0,
    "new_key": 9,
    "num_blocks_to_evict": 1,
    "evicted_keys": [int.from_bytes(k[:32], "big") for k in out.evicted_keys],
    "lookup_evicted": m.lookup(key(1), ctx()).name,
    "lookup_survivor_touched": m.lookup(key(0), ctx()).name,
    "state_after": state(m),
    "events": [
        {"keys": [int.from_bytes(k[:32], "big") for k in e.keys], "removed": e.removed}
        for e in m.take_events()
    ],
}

# ── ③ 驱逐失败→None 拒收（可逐数不足：全部被 prepare_load 钉住）──
m = mgr(num_blocks=2)
keys = [key(i) for i in range(2)]
for k in keys:
    m.prepare_store([k], ctx())
m.complete_store(keys, ctx())
m.prepare_load(keys, ctx())  # ref_cnt=1 → 全部不可逐
reject = m.prepare_store([key(9)], ctx())
evict_fail_case = {
    "pool_num_blocks": 2,
    "pinned_keys": [0, 1],
    "pinned_by": "prepare_load（ref_cnt=1）",
    "prepare_store_returns": "None" if reject is None else "output",
    "num_blocks_to_evict": 1,
    "num_evictable_cache_blocks": m._num_evictable_cache_blocks,
    "state_after": state(m),
    "semantics": "需逐数 1 > 可逐数 0 → 驱逐会失败 → None（best-effort 拒收）",
}

# ── ④ 输入保护集：已存块必须留下 ──
m = mgr(num_blocks=2)
keys = [key(i) for i in range(2)]
for k in keys:
    m.prepare_store([k], ctx())
m.complete_store(keys, ctx())
out = m.prepare_store([key(0), key(9)], ctx())
protected_case = {
    "input_keys": [0, 9],
    "already_stored_in_input": [0],
    "keys_to_store": [int.from_bytes(k[:32], "big") for k in out.keys_to_store],
    "evicted_keys": [int.from_bytes(k[:32], "big") for k in out.evicted_keys],
    "lookup_k0_after": m.lookup(key(0), ctx()).name,
    "note": "输入里已存的 k0 不进驱逐候选（源码注释：a block that was already stored must remain）",
}

# ── ⑤ ref_cnt 钉住/解钉对驱逐的影响 ──
m = mgr(num_blocks=2)
keys = [key(i) for i in range(2)]
for k in keys:
    m.prepare_store([k], ctx())
m.complete_store(keys, ctx())
spec = m.prepare_load([key(0)], ctx())
pinned_while_loading = m.prepare_store([key(9)], ctx())
m.complete_load([key(0)], ctx())
unpinned_after = m.prepare_store([key(10)], ctx())
refcnt_case = {
    "pinned_key": 0,
    "new_key_while_pinned": 9,
    "new_key_after_unpin": 10,
    "prepare_load_block_ids": [int(b) for b in spec.block_ids],
    "while_pinned_evicted": [int.from_bytes(k[:32], "big") for k in pinned_while_loading.evicted_keys],
    "after_complete_load_evicted": [int.from_bytes(k[:32], "big") for k in unpinned_after.evicted_keys],
    "note": "load 期间 k0 被 ref_cnt 钉住只能逐 k1；complete_load 解钉后 k0 可逐",
}

# ── ⑥ store_threshold=2 频次过滤（counts LRU tracker）──
m = mgr(num_blocks=4, store_threshold=2)
kt = key(0)
m.lookup(kt, ctx())  # 计数 1
out1 = m.prepare_store([kt], ctx())
m.lookup(kt, ctx())  # 计数 2
out2 = m.prepare_store([kt], ctx())
threshold_case = {
    "store_threshold": 2,
    "seen_once_keys_to_store": len(out1.keys_to_store),
    "seen_twice_keys_to_store": len(out2.keys_to_store),
    "counts_tracker_len": len(m.counts),
    "counts_value": m.counts[kt],
    "note": "出现 ≥2 次的块才值得搬；TieringOffloadingSpec 显式 raise 禁用 ≥2（cascade 要求全块）",
}

# ── ⑦ 事件面：stored / removed ──
m = mgr(num_blocks=1)
ka = key(0)
m.prepare_store([ka], ctx())
m.complete_store([ka], ctx())
stored_events = list(m.take_events())
m.prepare_store([key(1)], ctx())  # 池 1 块 → 逐 ka
removed_events = list(m.take_events())
events_case = {
    "stored_event_count": len(stored_events),
    "stored_removed_flag": stored_events[0].removed,
    "stored_medium": stored_events[0].medium.value,
    "removed_event_count": len(removed_events),
    "removed_keys": [int.from_bytes(k[:32], "big") for k in removed_events[0].keys],
    "removed_removed_flag": removed_events[0].removed,
}

# ── ⑧ complete_store 失败回收 ──
m = mgr(num_blocks=4)
kf = key(0)
m.prepare_store([kf], ctx())
m.complete_store([kf], ctx(), success=False)
fail_case = {
    "lookup_after_failed_store": m.lookup(kf, ctx()).name,
    "state_after": state(m),
}

dump(
    "m04.json",
    {
        "lifecycle_four_states": lifecycle,
        "lru_eviction_with_touch": lru_case,
        "eviction_failure_none": evict_fail_case,
        "input_keys_protected": protected_case,
        "refcnt_pin_unpin": refcnt_case,
        "store_threshold_filter": threshold_case,
        "events_stored_removed": events_case,
        "complete_store_failure": fail_case,
    },
)
