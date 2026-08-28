# ch15 前缀缓存 — impl-notes（只做减法精简版）

对应真实源码 pin **vLLM v0.27.1 (6e448d0ea)**，行号全部本日现核（2026-08-28，
`instances/vllm/source`），**不是** v2 资产的 v0.21.0 旧行号。
运行：`cd instances/vllm/artifacts-v3/ch15-prefix-caching && python -m pytest tests/ -q`
（45 passed；纯 host 单元/契约测试，不 import vllm——cbor2/torch CUDA 面以
HOST SEAM 承载，见下）。

本章精简版跑 **enable_prefix_caching=True 支**（默认开），对照 ch13 的 False
支（ch13 全绿的是 NoPrefixCache 原生路径 kv_cache_coordinator.py:L864-L876）。
控制流闭合点：`get_kv_cache_coordinator` 的单组 → Unitary / 多组 → Hybrid
分派；`get_new_blocks` 走摘哈希支（block_pool.py:L664-L668）；
`free_blocks` 劈分生效（L735 条件真）；`allocate_slots` 进 `cache_blocks`
写回（kv_cache_manager.py:L554-L563）——ch13 里被砍成黑盒的
allocate_slots 两个参数（new_computed_blocks/num_new_computed_tokens）本章
全部通电。

## 包结构（与真实树同名同构）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `kv_cache_utils.py` | `vllm/v1/core/kv_cache_utils.py` | **哈希链族**：BlockHash/BlockHashWithGroupId 键打包、NONE_HASH 种子（PYTHONHASHSEED）、hash_block_tokens（Merkle 链）、extra keys 谓词+四源并三源（mm/lora/cache_salt 仅首块）、request_block_hasher（只算新满块）、BlockHashListWithBlockSize 惰性重串、resolve_block_hashes + KVCacheBlock 七字段 + FreeKVCacheBlockQueue 全原语（ch13 已建、本章哈希面复用） |
| `block_pool.py` | `vllm/v1/core/block_pool.py` | BlockHashToBlockMap（NOTE #1 不去重/NOTE #2 union）+ BlockPool 前缀面：get_cached_block、cache_full_blocks（block_mask/晋升）、cache_partial_block、_get_partial_block_hash、_insert/_remove_cached_block_hashes（主+反向索引）、move_block_hashes、get_new_blocks+_maybe_evict_cached_block（惰性驱逐）、touch、free_blocks（劈分）、reset_prefix_cache |
| `single_type_kv_cache_manager.py` | `vllm/v1/core/single_type_kv_cache_manager.py` | ABC 账本（partial-hit +1 预算/CoW 换尾/_apply_cow/reachable_boundaries 组装/free 逆序）+ FullAttentionManager（phase 1 miss 即停 + phase 2 块内自高向低 + _cache_partial_tail_block）+ SlidingWindowManager（右到左窗口连续段 + 稀疏驻留两段 mask）+ MambaManager（边界状态 finder + Marconi 特赦 mask；align 分配内部删）+ 四特化壳 + 注册表装配 |
| `kv_cache_coordinator.py` | `vllm/v1/core/kv_cache_coordinator.py` | retention 校验 + 基类扇出 + KVCacheCoordinatorNoPrefixCache + Unitary + Hybrid（verify_and_split full 排首/_cache_hit_alignment_tokens/enable_partial_hash_hits/find_longest_cache_hit 不动点）+ get_kv_cache_coordinator 三态分派 |
| `kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py` | KVCacheBlocks 包装 + KVCacheManager 前缀面：prefix_cache_lookup_enabled、get_computed_blocks（skip 谓词/max_cache_hit_length=num_tokens−1/junction 写回）、allocate_slots 三段式（ch13 骨架通电）、free、take_kv_cache_block_copies、reset_prefix_cache |
| `request.py` | `vllm/v1/request.py` | Request 哈希面：block_hashes 账本、update_block_hashes（构造尾+append 增量）、shared_prefix_boundary 字段、skip_reading_prefix_cache 谓词 |
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py` | 第 3/8/9/10-12 站调度侧半边：admission_lookup（准入查命中+junction 写回）、_mamba_block_aligned_split（四停点）、pack_kv_cache_block_copies、_free_cow_retained_blocks+_drain_deferred_frees（步序栅栏）、_preempt_request（F2 起点）、_free_request_blocks + FCFSRequestQueue 最小镜像 |
| `engine_core.py` | `vllm/v1/engine/core.py` | L220-L229 装配开关（ENGINE SEAM 抽出为 assemble_block_hasher） |
| `cache.py` | `vllm/config/cache.py` | prefix_match_unit（=hash_block_size）/enable_prefix_caching（默认 True）/prefix_caching_hash_algo（默认 sha256）/mamba_cache_mode 语义账位 |
| `output.py` | `vllm/v1/core/sched/output.py` | SchedulerOutput 的 kv_cache_block_copies + new_block_ids_to_zero 两字段切面 |
| `gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | L1219-L1228 两件副作用（清零账位 + CoW 真拷贝）ENGINE SEAM 抽出 |
| `worker_utils.py` | `vllm/v1/worker/utils.py` | copy_kv_cache_blocks_inplace（块主序存储整块搬运；HOST SEAM CPU 镜像） |
| `torch_utils.py` | `vllm/utils/torch_utils.py` | async_tensor_h2d + PIN_MEMORY=False（HOST SEAM） |
| `hashing.py` | `vllm/utils/hashing.py` | sha256（pickle→SHA-256）+ get_hash_fn_by_name（sha256 支） |
| `envs.py` | `vllm/envs.py` | VLLM_PREFIX_CACHE_RETENTION_INTERVAL 一个环境位 |
| `stats.py` | `vllm/v1/metrics/stats.py` | PrefixCacheStats（record 的 preempted 分账——命中率官方口径） |
| `kv_cache_interface.py` | `vllm/v1/kv_cache_interface.py` | spec 家族最小面（Full/SWA/Mamba/ChunkedLocal/RSWA/Cross/SinkFull）+ KVCacheConfig 两属性 |
| `kv_cache_spec_registry.py` | `vllm/v1/kv_cache_spec_registry.py` | spec→manager 注册表（ch14 同款镜像） |
| `math_utils.py` | `vllm/utils/math_utils.py` | cdiv |

## 1:1 Source Map（关键段；改动=减法或 seam，原因=批准条/章节边界）

| 精简版 | 真实源 | 改动 | 原因 |
|---|---|---|---|
| `kv_cache_utils.py hash_block_tokens` | `kv_cache_utils.py:L596-L623` | 逐字（链式本体：H((parent, tokens, extra_keys))，首块 parent=NONE_HASH） | m1 must_keep——第一主角 |
| `kv_cache_utils.py request_block_hasher` | `kv_cache_utils.py:L705-L748` | 逐字（早停/只哈希满块/prev 链式/extra_keys 逐块问询） | m1 must_keep |
| `kv_cache_utils.py generate_block_hash_extra_keys` | `kv_cache_utils.py:L558-L593` | prompt_embeds_keys 源删（`+ prompt_embeds_keys` 一项） | dossier.delete 第 8 条 |
| `kv_cache_utils.py init_none_hash` | `kv_cache_utils.py:L99-L114` | CBOR 告警段删（条件依赖已删的 cbor 变体集） | hashing.py 变体随第 8 条邻界删 |
| `block_pool.py BlockHashToBlockMap` | `block_pool.py:L33-L140` | 逐字（get_one_block/contain/insert/pop 全原语；NOTE #1/#2 docstring 原文） | m2 must_keep |
| `block_pool.py cache_full_blocks` | `block_pool.py:L225-L342` | 核心环逐字减事件行（L268-L270/L292/L298-L299）与事件发布段（L301-L342） | dossier.delete 第 1 条 |
| `block_pool.py cache_partial_block` | `block_pool.py:L445-L544` | 核心逐字减事件段（L507/L513-L543）；_get_partial_block_parent_hash_and_start（L559-L569）随事件删 | 第 1 条；m13 must_keep |
| `block_pool.py _remove/_insert/move` | `block_pool.py:L571-L590/L607-L627/L629-L645` | 逐字（主哈希+反向索引双向维护/CoW 重指） | m9/m10 must_keep |
| `block_pool.py get_new_blocks/_maybe_evict` | `block_pool.py:L647-L700` | metrics 调用行删（L669-L670/L675-L676/L691-L692）、事件行删（L699） | 第 2/1 条；m9 must_keep |
| `block_pool.py touch` | `block_pool.py:L702-L717` | metrics 行删（L716-L717） | 第 2 条；m5/F2 第四步 |
| `block_pool.py free_blocks` | `block_pool.py:L719-L742` | 逐字（劈分两半 blocks_with_hash/blocks_without_hash + prepend_n/append_n 原样） | m8 must_keep（局部变量名保留） |
| `single_type add_local_computed_blocks` | `single_type:L232-L289` | 逐字（touch→null 填充→extend→num_cached_block→partial 记账） | m5 must_keep |
| `single_type allocate_new_blocks` | `single_type:L330-L369` | 逐字（CoW 前缀 L347-L357 保留——本章新通电；ch13 曾删） | m13 must_keep |
| `single_type _apply_cow` | `single_type:L405-L425` | 逐字（原地换尾/登记拷贝对/cow +1 引用） | m13 must_keep |
| `single_type FullAttention.find` | `single_type:L681-L777` | dcp 参数+×dcp 分支删（L691-L692/L701-L704）、eagle 丢尾删（L764-L769） | 第 3/4 条；m4 must_keep（phase1/phase2 逐字） |
| `single_type SWA.find` | `single_type:L896-L993` | dcp/pcp assert 删、eagle 对齐微调段删（L951-L954/L974-L991） | 第 3/4 条；m15 的 SWA finder 主干逐字 |
| `single_type SWA/Mamba.reachable_block_mask` | `single_type:L995-L1055/L1358-L1414` | eagle 的 shift=1 与 need+1 删（恒 shift=0/use_eagle=False） | 第 3 条；m16/m17 主干逐字（分段尾+特赦两段） |
| `single_type MambaManager` | `single_type:L1253-L1744` | find（L1279-L1356）+ reachable_mask（L1358-L1414）+ get_num_skipped（L1667-L1673）保留；align 分配内部四重写+_cache_partial_tail_block mamba 版+cached_blocks_this_step 族（L1452-L1651/L1675-L1744）删 | 第 6 条；Marconi 面（mask/finder）全保留 |
| `single_type RSWA/ChunkedLocal/Cross/SinkFull` | `single_type:L832-L875/L1095-L1250/L1747-L1807/L1810-L1833` | 类内部删、最小壳 + 注册表条目保留 | 第 7 条 |
| `coordinator _validate_retention` | `coordinator:L30-L57` | 逐字（三态校验：None/非负/整除/只对 SWA-Mamba） | m17 must_keep |
| `coordinator Hybrid.__init__` | `coordinator:L527-L589` | pcp/dcp 断言段删（L570-L580）；enable_partial_hash_hits 逐字（dcp==1 条件随第 4 条删后恒真） | 第 4 条；m13 装配前提 must_keep |
| `coordinator Hybrid.find` | `coordinator:L685-L817` | eagle 段删（L722-L725/L747/L750-L765/L780-L784）、find_longest_cache_hit_per_group 删（L819-L848）；不动点主循环/收尾/num_uncached 逐字 | 第 3/5 条；m15/m16 must_keep |
| `coordinator 三态分派` | `coordinator:L851-L903` | 观测参数删；False→NoPrefixCache/单组→Unitary/多组→Hybrid 逐字 | m18 must_keep |
| `manager get_computed_blocks` | `kv_cache_manager.py:L229-L295` | kv_cache_report_mode='full' 事件段删（L266-L284）；skip 谓词/max_cache_hit_length/junction 折算逐字 | 第 1 条；m4/m16 must_keep |
| `manager allocate_slots` | `kv_cache_manager.py:L344-L565` | watermark/full-ISL/reserved/lookahead/external-挂块段删（ch14/16/ch33 边界）；三段式主干+命中挂块+写回逐字；delay_cache_blocks 参数占位保留 | 第 5 条「保留 delay_cache_blocks 参数占位」 |
| `manager take_kv_cache_block_copies` | `kv_cache_manager.py:L831-L846` | 逐字（drain→KVCacheBlockCopy+retained 两端） | m14 must_keep |
| `request.py update_block_hashes/append` | `request.py:L249-L265` | 逐字（增量时机——哈希随 token 到达） | m1 must_keep |
| `request.py shared_prefix_boundary` | `request.py:L190-L193` | 逐字（Marconi junction 落点字段） | m16 must_keep |
| `scheduler admission_lookup` | `scheduler.py:L744-L766` | connector 分支删（L749-L759）；本地命中+junction 写回逐字（ENGINE SEAM 抽出） | 第 5 条；第 3 站 |
| `scheduler _mamba_block_aligned_split` | `scheduler.py:L362-L437` | chunk 预算对齐段删（L402-L409→ch10）、eagle 回退删（L393-L394）；四停点+取最早逐字 | 第 3 条 + ch10 边界；m16 停点 |
| `scheduler pack/_free_cow/_drain` | `scheduler.py:L1181-L1190/L2356-L2380` | producer 手递手段删（L1165-L1179）；CoW 打包+步序栅栏逐字 | 第 5 条；m14 must_keep |
| `scheduler _preempt_request` | `scheduler.py:L1274-L1315` | encoder/spec/stale/事件段删；free 全部块+归零+回队头逐字（ENGINE SEAM 无 timestamp 默认 0） | 第 4/10 条；F2 起点 must_keep |
| `engine_core assemble_block_hasher` | `engine/core.py:L220-L229` | 控制流逐字（ENGINE SEAM 抽出为函数） | 第 2 站 must_keep 面 |
| `worker copy_kv_cache_blocks_inplace` | `worker/utils.py:L528-L564` | 逐字（HOST SEAM：CPU 张量等价复现，GPU 面容器验） | m14 must_keep（管线终点） |
| `stats PrefixCacheStats` | `v1/metrics/stats.py:L18-L32/L115-L142` | 逐字（record 的 preempted 分账）；manager 侧 record/make 两口删 | 第 10 条「留空实现」；m19 must_keep |
| `block_pool reset_prefix_cache` | `block_pool.py:L763-L797` | metrics/events 段删；全空闲才清+清表+全块 reset_hash 逐字 | m20 must_keep |
| `interface KVCacheConfig 两属性` | `kv_cache_interface.py:L991-L993/L1013-L1022` | has_mamba_layers 逐字；needs_kv_cache_zeroing 的混合精度半边删 | ch13 边界 |

## HOST SEAM / SEAM 清单（不 import vllm 的等价复现点）

1. **LOGGER SEAM**：`vllm.logger.init_logger` → stdlib `logging.getLogger`
   （reset_prefix_cache 的 warning/info 账目同构）。
2. **HOST SEAM（torch）**：`PIN_MEMORY=False`、`copy_kv_cache_blocks_inplace`
   在 CPU 张量上等价跑（块主序 set_/view/索引搬运语义与 GPU 相同；带宽语义
   容器验）。
3. **hashing 变体**：sha256_cbor/xxhash 族随可选依赖删（默认 sha256；
   `get_hash_fn_by_name` 只留 sha256 支）——dossier 第 8 条邻界。
4. **ENGINE SEAM**：scheduler 三段（admission_lookup/pack_kv_cache_block_
   copies）与 engine/core 装配段从大函数抽出为方法/函数以便单测，控制流逐字
   （ch13 同款惯例）；`_zero_block_ids` 内景 → ch13（清零账位以注释保留）。
5. **FCFSRequestQueue 最小镜像**：真实 `vllm/v1/core/sched/request_queue.py`
   的 FCFS deque + prepend_request（ch10/11 已建全量优先级面）。

## 与 ch13/ch14 精简版的关系（读者跑三本对照）

- **ch13（False 支）**：allocate_slots 的 `new_computed_blocks`/
  `num_new_computed_tokens` 两参数在本章通电（那里早退/恒空）；get_new_blocks
  的摘哈希支、free_blocks 的劈分在 ch13 是死分支、本章活。
- **ch14**：hash_block_size 的**选择**（GCD/prefix_match_unit）与两粒度
  resolve 在 ch14 定账；本章只**消费** hash_block_size 参数（每文件头注明）。
- mamba align 分配内部（last_state_block_idx 状态块滚动等）按 dossier.delete
  第 6 条删——find 的 fine 分支与 Marconi mask 保留，故「mamba 边界状态写回」
  在测试里以同一 `cache_partial_block` 原语补登记（真实由被删的重写完成，
  block_pool.py:L1729-L1735 同款调用）。
- **impl≠pin 边界（仅 block_size==hash_block_size 的 mamba-align 配置）**：
  真实 align 模式 MambaManager 的块表是 `[NULL×(块数−1), 唯一状态块]`
  （get_num_skipped_tokens=num_computed−1 → null 占位，
  single_type:L1667-L1673），只把**最后一个**状态块登记进哈希表（如
  A(80)/块 16 → 仅 hash[4]@80）——而 get_computed_blocks 的
  `max_cache_hit_length=num_tokens−1` 永远探不到链尾 → mamba 组恒 miss、
  混合不动点把整笔命中拖到 0（钉版 v0.27.1 容器差分实测 hit==0/
  boundary==64；对 NULL 块摘哈希是 no-op）。impl 按 delete 第 6 条删了
  align 分配内部、走基类回退，mamba 组像 full 一样逐块登记——该逐块命中
  **只是减法的副作用、不是 pin 行为**。因此 m15/m16 的 mamba reconcile /
  Marconi 场景一律在 partial-hit 粒度配置（block_size(64) > hash_block_
  size(16)，即 `enable_partial_hash_hits` 面）驱动：该配置下 impl 的边界
  替换已与钉版差分逐字节一致（双向验证配方：A(48)→free→new_step_starts→
  摘 mamba 组 @48 条目→B(80 共享前 48) 查询 → hit==0/boundary==48；不摘时
  两侧同为 48/0）。写作时勿把 block_size==hash_block_size 配置下的 mamba
  组命中讲成真实行为。

## 验收

- `python -m pytest tests/ -q`：**45 passed**（0 failed/skipped/warnings）
- `python scripts/lint_fidelity.py <本章>`：**0 BLOCKING**（保真度全绿）
- must_keep 60+ 符号全在（lint over_subtraction 零命中）
