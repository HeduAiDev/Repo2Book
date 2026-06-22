# ch15《分页 KV 缓存机制：块池与前缀缓存》交付 APPROVED

- **Type**: delivery
- **Chapter**: ch15
- **Date**: 2026-06-22
- **Timestamp**: 2026-06-22T14:56:41Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: kv-cache, paging, prefix-caching, block-pool, lru, ref-count, lazy-eviction, chained-hash

## What happened

自底向上解读 vllm/v1/core 分页 KV 缓存主路径(单组+全注意力+开前缀缓存)：(1) KVCacheBlock 元数据(block_id 不变身份/ref_cnt/_block_hash setter 断言+reset_hash/is_null)。(2) FreeKVCacheBlockQueue 哨兵双向链表 LRU——popleft/popleft_n/remove(O(1) 中间删除,命中救回前提)/append_n,自实现而非 deque。(3) 引用计数+惰性驱逐：get_new_blocks(popleft_n+_maybe_evict_cached_block)/touch(命中 ref+1 必要时 remove 出队)/free_blocks(归零入队尾保留 hash)；ref_cnt==0 ⟺ 在 free queue 不变量；null_block popleft+is_null。(4) 链式块哈希 hash_block_tokens(parent,tokens,extra_keys 三元组,NONE_HASH 兜底)——前缀全同才同哈希、一断即停天然正确；extra_keys(mm_hash/lora_name/cache_salt 仅首块)防跨语义错误命中。(5) BlockHashToBlockMap union 退化 dict 不去重(维持 block_id append-only)。(6) find_longest_cache_hit islice(max_num_blocks) 逐块 get_cached_block break-on-miss;get_computed_blocks 守卫 enable_caching/skip + max_cache_hit_length=num_tokens-1。(7) allocate_slots 三段式:容量检查(num_evictable 把可驱逐命中块算进)返 None 触发抢占→allocate_new_computed_blocks touch 挂命中块(复用非重分配)→get_new_blocks 补块 + cache_full_blocks 登记哈希;free reversed 尾块先驱逐。删除点全 # SUBTRACTED(metrics/滑窗/Mamba/多组/投机草稿头/上下文并行/connector)。四 linter 全 PASS、host 25/25 测试、行为逐条核对 pin f3fef123 忠实子集、6 配图。

## Why it matters

首次掀开 ch14 当黑盒的 kv_cache_manager(要块/还块/命中)，给出分页+前缀缓存完整机制;结清 f7(CacheConfig.block_size/enable_prefix_caching 真正发挥作用)与 f11(抢占重算靠未驱逐前缀块直接复用缓解);为下一章 attention backend(KV 写入物理块)铺好地址簿。

## What to remember

ch15 已 APPROVED 归档。bible 新增 7 条 ch15 精简版接口(KVCacheBlock/FreeKVCacheBlockQueue/BlockPool/BlockHashToBlockMap/链式哈希族/KVCacheManager/FullAttentionManager)。伏笔 f7(plant ch03)、f11(plant ch14) 均已回收 status=resolved resolved_in=ch15,本章无应埋新伏笔、无悬挂。关键不变量:ref_cnt==0 ⟺ 块在 free queue;释放保留 hash、拖到被 get_new_blocks 取走才 _maybe_evict 清——这是前缀缓存跨请求/跨抢占复用全部秘密。review 3 条 issue 均 non-blocking nit。启下:attention backend 把 KV 写进这些块。
