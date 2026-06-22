# ch16《KV 块分配与多注意力协调》交付 APPROVED

- **Type**: delivery
- **Chapter**: 16
- **Date**: 2026-06-22
- **Timestamp**: 2026-06-22T15:56:12Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: kv-cache, allocate_slots, coordinator, hybrid-attention, fixed-point, admission-cap, sliding-window, APPROVED

## What happened

ch16 在 ch15 块池/前缀缓存数据结构之上补全分配决策与多注意力协调三大深水区，承接但不重复 ch15 块池/LRU/链式哈希内部。(1) allocate_slots 三阶段总装：阶段一 remove_skipped_blocks 先释放窗外块(逆序换 null 不删除保 append-only 下标、free_blocks 归还、遇 null 早停幂等)+ get_num_blocks_to_allocate 精确预算检查(num_new_blocks skipped 折抵 + num_evictable 可驱逐命中块计入容量)不够返 None 触发 ch14 抢占；阶段二 allocate_new_computed_blocks(切 skipped 头 + touch 命中块共享 + null 填 skipped 段 + external 命中走 get_new_blocks 真实块非借用)；阶段三 allocate_new_blocks 补 new+lookahead 槽位 + cache_full_blocks 封顶 request.num_tokens(lookahead 草稿不入缓存)。(2) admission cap：SWA/chunked-local 的 max_admission_blocks_per_request 经 get_manager_for_kv_cache_spec 注入，与 ch15 启动期池估算同口径单一真相源，防 issue #39734 死锁/中途 OOM。(3) KVCacheCoordinator 三态工厂 get_kv_cache_coordinator：!enable_caching→NoPrefixCache(任意组数含0)/单组→Unitary(直委托唯一 manager)/多组→Hybrid(verify_and_split 分桶+full 排首+lcm_block_size)。(4) Hybrid 不动点迭代 find_longest_cache_hit：每注意力类型接受或缩短命中长度、缩短即重启，单调递减+lcm 倍数离散下界 0 保证 <=L/lcm 轮收敛，full 排首给紧上界、simple-hybrid 一轮 early-break。四 linter 全 PASS、host 26/26 测试(断言逐条对照 pin f3fef123 真实源码可观察行为含 docstring 自证载荷)、为忠实子集只做减法。review overall_verdict=APPROVED，3 条均 non-blocking nit。

## Why it matters

ch16 是 Part 分页 KV 缓存的分配决策与多注意力协调收口章：把 ch14 当黑盒的 allocate_slots/free 与 ch15 自底向上铺的块池/哈希数据结构，在本章合拢成完整分配编排 + Hybrid 混合注意力协调。不动点收敛、admission cap 同口径、skipped 块回收三条是高级读者理解 vLLM v1 KV 管理决策层的关键。

## What to remember

ch16 在 ch15 块池/前缀缓存数据结构之上补全分配决策与多注意力协调三大深水区，承接但不重复 ch15 块池/LRU/链式哈希内部。(1) allocate_slots 三阶段总装：阶段一 remove_skipped_blocks 先释放窗外块(逆序换 null 不删除保 append-only 下标、free_blocks 归还、遇 null 早停幂等)+ get_num_blocks...
