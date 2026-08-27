# ch13 分页 KV — impl-notes（只做减法精简版）

对应真实源码 pin **vLLM v0.27.1 (6e448d0ea)**，行号全部本日现核（2026-08-27，
`instances/vllm/source`），**不是** v2 资产的 v0.21.0 旧行号。
运行：`cd instances/vllm/artifacts-v3/ch13-paged-kv && python -m pytest tests/ -q`
（54 passed；纯 host 单元/契约测试，不 import vllm——CUDA 面（Triton kernel
launch）以 HOST SEAM 承载，见下）。

本章精简版跑 **enable_prefix_caching=False** 支（cache_config 的正交开关）：
`get_kv_cache_coordinator` 的 `if not enable_caching: return KVCacheCoordinator
NoPrefixCache(...)` 是源码原生路径（kv_cache_coordinator.py:L864-L876）。控制流
闭合点：get_new_blocks 走免摘哈希支（block_pool.py:L671-L676）、free_blocks 劈
分跳过（L735 条件假）、allocate_slots 在 L551 早退不进 cache_blocks、前缀命中
恒 0——哈希链语义 ch15 另做精简版。

## 包结构（与真实树同名同构；两处 utils.py 重名以可辨前缀区分）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `kv_cache_utils.py` | `vllm/v1/core/kv_cache_utils.py` | KVCacheBlock 七字段 + FreeKVCacheBlockQueue 全原语（popleft/popleft_n/remove/append/prepend_n/append_n/get_all_free_blocks） |
| `block_pool.py` | `vllm/v1/core/block_pool.py` | 池构造（null_block 占 0 号）+ get_new_blocks/touch/free_blocks + get_num_free_blocks/get_usage |
| `kv_cache_interface.py` | `vllm/v1/kv_cache_interface.py` | KVQuantMode + KVCacheSpec/AttentionSpec（real_page_size_bytes 公式）+ FullAttentionSpec + KVCacheConfig/GroupSpec/Tensor（needs_kv_cache_zeroing） |
| `single_type_kv_cache_manager.py` | `vllm/v1/core/single_type_kv_cache_manager.py` | 基类账本（req_to_blocks/num_cached_block/get_num_blocks_to_allocate/allocate_new_blocks/take_new_block_ids/cache_blocks 账位/pop/free）+ FullAttentionManager + get_manager_for_kv_cache_spec |
| `kv_cache_coordinator.py` | `vllm/v1/core/kv_cache_coordinator.py` | 基类扇出 + KVCacheCoordinatorNoPrefixCache + get_kv_cache_coordinator 的 False 支 |
| `kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py` | KVCacheBlocks 包装 + KVCacheManager（allocate_slots 三段式/free/take_new_block_ids/cache_blocks 调用点/create_kv_cache_blocks/empty_kv_cache_blocks） |
| `request.py` | `vllm/v1/request.py` | KV 账本消费面（num_computed_tokens/num_tokens/status/block_hashes 空账位/is_finished） |
| `cache.py` | `vllm/config/cache.py` | DEFAULT_BLOCK_SIZE=16 / block_size None→默认 |
| `math_utils.py` | `vllm/utils/math_utils.py` | cdiv / largest_power_of_2_divisor |
| `torch_utils.py` | `vllm/utils/torch_utils.py` | PIN_MEMORY（HOST SEAM False）/ get_dtype_size / async_tensor_h2d |
| `v1_utils.py` | `vllm/v1/utils.py` | CpuGpuBuffer（CPU/GPU 双镜像容器——m15 的载体） |
| `block_table.py` | `vllm/v1/worker/block_table.py`（+backends/utils.py:L45-L46 常量折入） | BlockTable（append_row/commit_block_table/compute_slot_mapping）+ _compute_slot_mapping_kernel 恒等式本体 |
| `worker_utils.py` | `vllm/v1/worker/utils.py` | _zero_kv_blocks_kernel + KVBlockZeroer + AttentionGroup |
| `output.py` | `vllm/v1/core/sched/output.py` | NewRequestData/CachedRequestData/SchedulerOutput（new_block_ids_to_zero 契约面） |
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py` | 第 2/5/6/11/12 站调度侧半边（站点抽块） |
| `gpu_input_batch.py` | `vllm/v1/worker/gpu_input_batch.py` | CachedRequestState.block_ids + InputBatch 块表行面（容器内景 → ch18） |
| `gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | _update_states KV 镜像 / _prepare_inputs 块表先行拷贝+positions / _get_block_table / _allocate+_reshape 物理池 |

## 1:1 Source Map（关键段；改动=减法或 seam，原因=批准条/章节边界）

| 精简版 | 真实源 | 改动 | 原因 |
|---|---|---|---|
| `kv_cache_utils.py KVCacheBlock` | `kv_cache_utils.py:L117-L176` | 七字段逐字（含哈希两字段与 set/reset_hash——ch15 账位）；__repr__ 保留 | m2 must_keep 七字段 |
| `kv_cache_utils.py FreeKVCacheBlockQueue` | `kv_cache_utils.py:L184-L413` | 全原语逐字 | m3 must_keep（哨兵/O(1) remove/零分配） |
| `block_pool.py __init__` | `block_pool.py:L162-L191` | 块数组预构 + 自由队列 + null_block popleft 占 0 号逐字；events/metrics/哈希表字段删 | 第 1/2/3 条 |
| `block_pool.py get_new_blocks` | `block_pool.py:L647-L677` | 两分支保留；caching 支内 _maybe_evict_cached_block 调用删（哈希表恒空、无哈希可摘）；metrics 调用删 | 第 2/3 条 |
| `block_pool.py touch/free_blocks` | `block_pool.py:L702-L742` | 逐字（metrics 调用删）；free_blocks 的哈希劈分两行 prepend_n/append_n **原样保留**（False 支全走 append_n） | 第 2 条；m12/m4 |
| `single_type get_num_blocks_to_allocate` | `single_type:L144-L230` | cdiv 主算术 + fast-path + skipped 推导 + evictable 计数逐字；admission cap 分支删、partial-hit +1 删 | 第 4/9 条；m5 |
| `single_type allocate_new_blocks` | `single_type:L330-L369` | 差值分配主干逐字（req_blocks.extend 挂账 + new_block_ids 记录）；CoW 前缀删 | 第 9 条；m6 第 4 站 |
| `single_type add_local_computed_blocks` | `single_type:L232-L289` | touch/extend/num_cached_block 登记；skipped null 填充与 partial 尾块删 | 第 4/9 条 |
| `single_type cache_blocks` | `single_type:L427-L477` | 幂等闸 + num_cached_block 推进；reachable 掩码与 cache_full_blocks 调用删（调用点账位保留） | 第 3/4 条；must_keep |
| `single_type free` | `single_type:L519-L527` | 逐字（reversed 逆序——"tail blocks are freed first"） | m12 |
| `coordinator` 基类扇出 | `kv_cache_coordinator.py:L60-L382` | get_num_blocks_to_allocate/allocate_new_blocks(扇出)/free/pop_blocks_for_free/remove_skipped_blocks 逐字；cross/external/eagle/retention/dcp 删 | 第 4/5/6/7 条 |
| `coordinator NoPrefixCache` | `kv_cache_coordinator.py:L385-L432` | 逐字减 eagle/dcp 实参；find_longest_cache_hit 恒空命中保留 | 本章 False 支原生路径 |
| `kv_cache_manager KVCacheBlocks` | `kv_cache_manager.py:L32-L114` | blocks 字段/__add__/get_block_ids(allow_none)/new_empty 逐字；@overload 桩与 unhashed 删 | 第 11 条 |
| `kv_cache_manager allocate_slots` | `kv_cache_manager.py:L344-L565` | 三段式逐字：L440-L446 零 token 护栏 / L453-L461 合账 / L490-L508 no-op 回收 / L510-L527 容量检查（不够 None）/ L529-L547 挂命中+分新块 / L551-L552 False 早退 / L554-L563 写回；watermark 分支+full-ISL 闸+external/delay/reserved 删 | 第 7/8 条；m6 绝对主角 |
| `kv_cache_manager free/take_new_block_ids` | `kv_cache_manager.py:L567-L578、L796-L801` | 逐字（pins 删） | 第 7 条；m8 通道 |
| `scheduler allocate_slots_for_waiting` | `scheduler.py:L973-L985` | 站点抽块逐字（预算/connector 实参删） | m6 第 2 站 |
| `scheduler allocate_slots_for_running` | `scheduler.py:L575-L629` | while True 抢占环逐字（PRIORITY 支与账目回滚删——ch10/11） | m11 第 11 站 |
| `scheduler _preempt_request` | `scheduler.py:L1274-L1308` | 块侧两件事（free 全部块+computed 归零）逐字；encoder/spec/stale 删 | 第 4/5 条 |
| `scheduler make_new_reqs_data` | `scheduler.py:L1144-L1149` | 逐字（V2 分支删） | 第 12 条；m7 |
| `scheduler _make_cached_request_data` | `scheduler.py:L1400-L1467` | L1451-L1453 增量 get_block_ids(allow_none=True) 逐字；PP/V2 删 | 第 7/12 条；m7 |
| `scheduler _get_new_block_ids_to_zero` | `scheduler.py:L1260-L1272` | 逐字（_skip_zero_block_ids 删） | 第 7 条；m8 第 6 站 |
| `scheduler _free_blocks/_free_request_blocks` | `scheduler.py:L2329-L2354` | 逐字（deferred 栅栏分支删——ch12，即时还块） | m12 第 12 站 |
| `gpu_model_runner _update_states` | `gpu_model_runner.py:L1190-L1494` | KV 切面：L1202-L1217 清档 / L1219-L1222 清零 / L1266-L1309 新请求建档 / L1355-L1361 循环头 / L1441-L1452 差量 extend+整表替换 / L1471-L1474 落行——各块逐字；spec/async/ngram/PP 删 | 第 4/5/7/9 条；m7 第 7 站 |
| `gpu_model_runner _prepare_inputs` | `gpu_model_runner.py:L1960-L2201` | L1977-L1979 commit 第一句逐字（"overlap the copy"）/ L2188-L2191 positions GPU 组装逐字 / L2197-L2201 compute_slot_mapping 派发逐字；token/采样组装删 | m15/m9/m14 |
| `gpu_model_runner _get_block_table` | `gpu_model_runner.py:L2325-L2341` | 逐字（EncoderOnly 分支删）；NULL_BLOCK_ID 填 pad 行 | 第 4 条；m14/F7 |
| `gpu_model_runner _allocate/_reshape` | `gpu_model_runner.py:L7312-L7353、L7400-L7413` | 每层一块 int8 缓冲 + num_blocks=numel//page_size_bytes 逐字；视图用标准布局 [num_blocks, **2**, block_size, kv_heads, head_dim]（2×=K/V 两半——real_page_size_bytes 公式的 2×；backend 形状仲裁 → ch21）；packed 删 | 第 4 条；m10 |
| `block_table append_row/commit` | `block_table.py:L138-L154、L213-L214` | 逐字（hybrid 细分删） | 第 4 条；m7/m15 |
| `block_table _compute_slot_mapping_kernel` | `block_table.py:L379-L442` | PAD 尾 program + 恒等式主干逐字；CP 三处（TOTAL_CP_WORLD_SIZE/is_local/local_block_offsets/tl.where PAD）按常数 1 烘干删；BLOCKS_PER_KV_BLOCK=1 乘子保留 | 第 6 条；m9 must_keep |
| `worker_utils KVBlockZeroer` | `worker/utils.py:L44-L213` | kernel + 段表预计算 + zero_block_ids 逐字（CPU 分支 HOST SEAM） | m8 |

## HOST SEAM / ENGINE SEAM 登记（跨章边界，非减法）

- **compute_slot_mapping 的 CPU 镜像**（block_table.py `_compute_slot_mapping_host`）：
  host 无 CUDA launch——kernel 本体的逐行镜像（同一恒等式、同一 PAD 尾、同一
  变量名/扁平寻址）；CUDA 分支逐字保留，容器内（scripts/vllm_docker.sh）真跑。
- **KVBlockZeroer 的 CPU 分支**（worker_utils.py）：经**同一张绝对地址表**用
  ctypes.memset 按 (block_id, seg) 置零——kernel L85-L88 的 offset 算术逐行对应。
- **CpuGpuBuffer / BlockTable on CPU**：.gpu 与 .cpu 同为 CPU 张量（同构造、真
  拷贝）——双镜像契约（CPU 写 .np / commit 拷 .gpu[:n] 活跃行）逐字成立。
- **_StandardLayoutBackend / _Ctx**（gpu_model_runner.py）：真实 backend 由
  注意力后端注册表装配（→ ch21）、static_forward_context 来自 compilation_config；
  切面用全注意力标准布局（block_dim=0/1 由 duck type 提供）承载同一
  `get_kv_cache_block_dim` 契约位。
- **Scheduler / GPUModelRunner 站点抽块**（ENGINE SEAM，ch12 同款纪律）：从
  schedule()/_update_states/_prepare_inputs 的内联块抽出为方法以便单测——抽出
  而非改写，控制流逐字；整章 schedule() 的 token 预算面归 ch10/11。
- **InputBatch**（ch18 边界）：req_id_to_index/num_computed_tokens_cpu/block_table
  行面；condense/持久批维护内景归 ch18。add_request 的 add_row 取 `block_ids[0]`
  组（MultiGroup 扇出删后单组直写——真实 L397-L398 扇出各组）。
- **estimate_cached_tokens / get_zeroing_block_ids_in_range / record_blocks_
  for_zeroing / new_empty / truncate_computed_blocks 面上保留**：不在 delete
  批准清单 → 按防过度删减纪律原样保留（前者哈希账位恒空 → 0；后两者清零通道
  的正交补口）。

## 减法批准对照（dossier.subtraction_plan.delete 12 条 → 落点）

1. KV cache events 全套 → block_pool/kv_cache_manager 的 events 字段与
   take_events/emit/_build/_emit_block_removed 删；output.py 事件字段删。
2. metrics_collector → block_pool 三处调用与 kv_cache_manager 观测面删。
3. 前缀缓存哈希侧全链 → BlockHashToBlockMap/cache_full_blocks/cache_partial_
   block/_insert/_remove/move_block_hashes/_maybe_evict/get_cached_block/
   find_longest_cache_hit 族/get_computed_blocks(+_for_connector)/request_block_
   hasher 族删（BlockHash/BlockHashWithGroupId 类型账位与 KVCacheBlock 哈希
   字段保留——ch15 的账位）。
4. 混合/多组家族 → SlidingWindow/ChunkedLocal/Mamba/RSWA/Cross/Sink 管理器、
   Hybrid 协调器、admission cap、MultiGroupBlockTable/map_to_kernel_blocks/
   use_hybrid_blocks/token_alignment/SlotMappingMode.NONE/registry 注册删；
   Unitary 协调器随其唯一实体方法（哈希侧 find_longest_cache_hit）删。
5. eagle → use_eagle/eagle_group_ids/is_eagle_group/drop_eagle 删。
6. DCP/PCP → 乘子与组探测删；kernel CP 三处按常数 1 烘干。
7. external/connector/P-D → num_external_computed_tokens/delay_cache_blocks/
   reserved_blocks/allocate_external/_partial_tail_pins/partial_tail_offloads/
   evict_blocks/_skip_zero_block_ids 删。
8. full-ISL 闸与 watermark 分支 → allocate_slots L463-L488 删（watermark_
   blocks 字段账位保留——消费分支 ch14）。
9. CoW/partial-hit → _partial_hit_reqs/_apply_cow/take_pending_cow_copies/
   _has_partial_local_hit/take_kv_cache_block_copies/kv_cache_block_copies 删。
10. prompt_embeds extra keys → 随哈希侧删。
11. 运维/统计旁路 → reset_prefix_cache/get_num_common_prefix_blocks/
    get_block_ids_for_computed_tokens/get_unhashed_block_ids(_all_groups)/
    make_prefix_cache_stats/kv_cache_event_metadata/new_step_starts 事件段删。
12. V2 runner 分支 → scheduler L1132-L1142 与 output 的 V2 字段删。

## 与 must_keep 的对账

dossier.subtraction_plan.must_keep 全部 55 个符号均在精简版出现
（lint_fidelity 的 over_subtraction 检查绿）。要点对账：`prepend_n`/`append_n`
的调用点在 free_blocks 劈分两行原样保留（False 支全走 append_n，prepend 语义
→ ch15）；`cache_blocks` 三级调用点齐备（facade 门 → coordinator 扇出 →
manager 账位）；`_compute_slot_mapping_kernel` 烘干 CP 后保留 L418-L420/
L430-L442 主干；`new_block_ids_to_zero` 字段在 SchedulerOutput L253-L256 逐字；
`DEFAULT_BLOCK_SIZE`/`real_page_size_bytes`/`page_size_bytes` 公式逐字。

## 账本口径备注（writer 可用）

- worked example：prompt 100 token / block_size 16 → **7 块（112 槽）**，
  尾部浪费 12 < 16（`test_single_request_waste_under_one_block`）；块 id 从 1
  起（0 被 null_block 占——`test_pool_construction_and_null_block_takes_id_zero`）。
- 槽位恒等式：块表行 [3,1,7] → positions 0..47 映射 48..63 / 16..31 / 112..127
  （`test_identity_slot_equals_block_times_size_plus_offset`）；PAD 尾
  [num_tokens, max) = -1（`test_tail_padded_with_pad_slot_id`）。
- 页公式：block_size=16/kv_heads=8/head=128/fp16 → real_page_size_bytes =
  2×16×8×128×2 = **65536 B**；Llama-2-7B 每 token 每层 2×32×128×2 = 16384 B
  （`test_real_page_size_bytes_formula`/`test_per_token_kv_bytes_llama2_7b`）。
- 需块预测：fresh 100 token → 7；running fast-path 已持 7 再要 113 → 1；
  spec 拒绝回退 need<held → 0；可驱逐命中块 +1 计入
  （`TestNumBlocksToAllocate` 四测）。
- None→抢占：RUNNING 长大时池干 → allocate_slots None → while True pop 队尾
  最新者 preempt（FCFS）→ 被抢者 PREEMPTED/computed=0/块全还
  （`test_running_loop_none_triggers_preempt_oldest`）。
- 终局逆序 free：分配 [1..5]、逆序归还 → 队尾 [5,4,3,2,1]；耗尽新鲜块后按
  5,4,3,2,1 再分配（`test_free_reverse_order_tail_first`——LRU 尾优先序）。
- 逆序 half 的可观测语义：free_blocks(caching 关) 全走 append_n 保传入序；
  "tail first" 的驱逐优先级由 reversed 的**入队序**体现（队头先出）。
- 清零账：混合精度两组（fp16+fp32）构造 needs_kv_cache_zeroing=True 的合法
  场景（kv_cache_interface.py:L1013-L1022 混合精度半边）；uniform 单组 →
  False → _get_new_block_ids_to_zero 返回 None（`test_get_new_block_ids_to_
  zero_drains_each_step` 两支都测）。
- worker 镜像：新请求全量 block_ids 建档 + add_row 整行写；在跑请求差量
  extend + append_row 追加；恢复者整表替换（`TestWorkerMirror` 三态）。
