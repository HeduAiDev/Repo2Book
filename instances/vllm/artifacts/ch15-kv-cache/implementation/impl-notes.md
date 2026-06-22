# ch15 精简版实现说明（只做减法）

单组全注意力 + 前缀缓存开启的主路径忠实子集。与真实 vLLM 同名同结构同控制流，
只删 `subtraction_plan.delete` 批准项，`must_keep` 符号全部保留。所有删除处带
`# SUBTRACTED:` 注释，每个 def/class 带 `# SOURCE: vllm/...:Lxxx`。

精简版纯单元（不 import vllm），host `python3 -m pytest tests/` 即可跑（25 passed）。

## 1:1 Source Map（精简版 ↔ 真实 vLLM ↔ 改动 ↔ 原因）

| 精简版 | 真实 vLLM | 改动 | 原因 |
|---|---|---|---|
| `kv_cache_utils.KVCacheBlock` | `vllm/v1/core/kv_cache_utils.py:L113` | 删 `__repr__` | 仅调试打印块指针，不参与控制流 |
| `kv_cache_utils.NONE_HASH` | `kv_cache_utils.py:L91` | 直接随机种子，删 `init_none_hash` 的 PYTHONHASHSEED/CBOR 分支 | 可复现哈希是运维选项 |
| `kv_cache_utils.FreeKVCacheBlockQueue` | `kv_cache_utils.py:L162` | 原样保留 popleft/popleft_n/remove/append/append_n/哨兵 | 双向链表 LRU 是本章核心，`must_keep` 全在 |
| `kv_cache_utils.generate_block_hash_extra_keys` | `kv_cache_utils.py:L501` | 删 prompt_embeds 分支 | prompt-embeds 输入专用，常规 token 恒 [] |
| `kv_cache_utils._gen_prompt_embeds_extra_hash_keys` | `kv_cache_utils.py:L475` | 整体删除 | 同上；保留 mm/lora/cache_salt 三类已足以演示语义隔离 |
| `kv_cache_utils.hash_block_tokens` / `get_request_block_hasher` | `kv_cache_utils.py:L539 / L635` | 原样（hasher 的 curr_mm_idx=-1 优化保留） | 链式块哈希与请求侧逐满块哈希是核心 |
| `block_pool.BlockHashToBlockMap` | `vllm/v1/core/block_pool.py:L34` | 原样（含 union 单块/dict 多块、pop） | 去重映射语义要讲，`must_keep` |
| `block_pool.BlockPool.__init__` | `block_pool.py:L149` | 删 events/metrics 字段与参数 | 事件订阅 + 统计旁路，默认关闭 |
| `block_pool.BlockPool.cache_full_blocks` | `block_pool.py:L211` | 删 block_size!=hash_block_size 重算、is_null 跳过、BlockStored 事件 | 多组重算单组不触发；null 满块仅滑窗/Mamba；事件旁路 |
| `block_pool.BlockPool.get_new_blocks` | `block_pool.py:L322` | 删 metrics 调用（保留 enable_caching 双分支） | 纯统计旁路；控制流不变 |
| `block_pool.BlockPool._maybe_evict_cached_block` | `block_pool.py:L354` | 删 metrics + BlockRemoved 事件 | 旁路；reset_hash 主干保留 |
| `block_pool.BlockPool.touch` / `free_blocks` | `block_pool.py:L391 / L408` | 删 metrics 调用 | 旁路；ref_cnt ±1 与队列摘除/归还主干保留 |
| `single_type_kv_cache_manager.SingleTypeKVCacheManager` | `vllm/v1/core/single_type_kv_cache_manager.py:L30` | 删 ABC、dcp/pcp、admission cap、external/connector、skipped blocks 子句 | 单组全注意力下均 no-op / 0 |
| `single_type.FullAttentionManager.find_longest_cache_hit` | `single_type_kv_cache_manager.py:L447` | 删 dcp/pcp 乘子、use_eagle 丢尾块、alignment 裁剪 | 单组乘子=1、对齐恒成立、eagle 正交 |
| `kv_cache_coordinator.KVCacheCoordinator` | `vllm/v1/core/kv_cache_coordinator.py:L28` | 删 ABC、eagle、events/metrics/dcp/pcp、多组构造、cross-attn/encoder、common-prefix/skipped/new_step | 单组主路径直接构造唯一 FullAttentionManager |
| `kv_cache_coordinator.UnitaryKVCacheCoordinator` | `kv_cache_coordinator.py:L324` | 删 dcp/pcp block_size 乘子 | 单卡为 1 |
| `kv_cache_manager.KVCacheManager` | `vllm/v1/core/kv_cache_manager.py:L106` | 删 eagle/log_stats/metrics/events/dcp/pcp、工厂分派、运维接口 | 直接构造 Unitary 协调器；统计/运维旁路 |
| `kv_cache_manager.KVCacheManager.allocate_slots` | `kv_cache_manager.py:L225` | 删 lookahead/external/delay/encoder/full_seq_must_fit、remove_skipped_blocks(no-op) | 投机/connector/编码器/SWA 准入对全注意力本地路径均 0/no-op；三段式主干保留 |
| `kv_cache_manager.KVCacheBlocks` | `kv_cache_manager.py:L21` | 删 allow_none 重载、unhashed/new_empty | 便捷形态，不改 block_id 提取 |
| `request.Request` / `FullAttentionSpec` | `vllm/v1/request.py:L40` / `kv_cache_interface.py` | 最小视图：只留 block_hashes/all_token_ids/mm_features/lora_request/cache_salt/num_computed_tokens/num_preemptions/skip_reading_prefix_cache + update_block_hashes | 让精简版 host 可跑、可打断点；字段语义与真实一致；删 prompt_embeds/张量形状字段 |

## 伏笔回收锚点（供 writer）
- **f7**：`CacheConfig.block_size` 即 `FullAttentionSpec.block_size` / `hash_block_size`，是
  切块/哈希/分配/命中的最小粒度（`request_block_hasher` 逐 block_size 切、`find_longest_cache_hit`
  `max_length // block_size`）；`enable_prefix_caching` = `BlockPool.enable_caching` 决定
  `get_computed_blocks` 是否查命中、`get_new_blocks` 是否惰性驱逐、选 Unitary 协调器。
- **f11**：`free_blocks` 归还块时**保留 block_hash**（仅 ref_cnt→0 入 free queue 作驱逐候选），
  抢占请求重 prefill 时 `get_computed_blocks` 命中这些未驱逐前缀块、`touch` 救回，省去重算。
  见测试 `test_preempted_request_reuses_undropped_prefix_blocks`。
