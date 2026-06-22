# ch16 实现笔记 —— KV 块分配与多注意力协调（只做减法的精简版）

本章在 ch15（块池/LRU/哈希数据结构 + 单组全注意力主路径）之上，补全三件 ch15 显式
留作可省略分支的深水区：allocate_slots 完整三阶段、KVCacheCoordinator 三态拓扑、
Hybrid 的不动点迭代命中查找。共享基础设施（`block_pool.py` / `kv_cache_utils.py`）
直接复用 ch15 的精简版，未改动。

本章精简版 1:1 映射的真实源文件（pin f3fef123）：

- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/core/kv_cache_coordinator.py`

## 相对 ch15 新纳入（不是新发明，是把 ch15 SUBTRACTED 的真实分支补回）

- `allocate_slots`：补回 `num_lookahead_tokens` / `num_external_computed_tokens` /
  `delay_cache_blocks` / `num_encoder_tokens` / `full_sequence_must_fit` 参数与
  `remove_skipped_blocks` 调用、`full_sequence_must_fit` 准入闸、`total_computed_tokens`
  含 external 的合并。
- `SingleTypeKVCacheManager`：补回 `get_num_blocks_to_allocate` 的 skipped 折抵 +
  可驱逐块计数 + `apply_admission_cap` 夹取；`allocate_new_computed_blocks` 的
  null 填 skipped 段 + external 命中 `get_new_blocks`；`remove_skipped_blocks` 逆序换
  null + 归还；`get_num_skipped_tokens` 基类默认 0。
- 新增差异化子类 `SlidingWindowManager`（skipped = max(0, n-window+1)）、
  `ChunkedLocalAttentionManager`（skipped 向下取整到 chunk 边界）及其 `find_longest_cache_hit`。
- `KVCacheCoordinator` 基类逐组转发 + 三个具体协调器
  （NoPrefixCache / Unitary / Hybrid）+ `get_kv_cache_coordinator` 三态工厂。
- `spec_manager_map` + `get_manager_for_kv_cache_spec`（含 SWA/chunked-local 的
  `max_admission_blocks_per_request` 注入）。

## 1:1 Source Map（精简版 ↔ 真实 vllm:Lxxx ↔ 改动 ↔ 原因）

| 精简版符号 | 真实源 vllm:Lxxx | 改动 | 原因 |
|---|---|---|---|
| `KVCacheManager.allocate_slots` | `kv_cache_manager.py:L225-L416` | 保留全部三阶段 + 五个可选参数/分支；只删 `log_stats` 等观测 | 本章主骨架，writer 全章围绕它展开 |
| `SingleTypeKVCacheManager.get_num_blocks_to_allocate` | `single_type_kv_cache_manager.py:L88-L167` | 完整保留 skipped 折抵 / 可驱逐计数 / admission cap；删 dcp/pcp 乘子 | 预算检查精确预测，本章理论落点 |
| `SingleTypeKVCacheManager.allocate_new_computed_blocks` | `single_type_kv_cache_manager.py:L169-L240` | 保留 null 填 skipped + external `get_new_blocks`；`TQFullAttentionSpec` 并列删（本章不造该 spec） | 阶段二三步：touch / null 填充 / external 分配 |
| `SingleTypeKVCacheManager.remove_skipped_blocks` | `single_type_kv_cache_manager.py:L385-L426` | 原样保留逆序换 null + free_blocks + 遇 null 早停 | 阶段一释放 skipped 块核心 |
| `SlidingWindowManager.get_num_skipped_tokens` | `single_type_kv_cache_manager.py:L606-L632` | 保留公式，docstring ASCII 图压缩为文字 | 窗外块回收的窗口语义 |
| `ChunkedLocalAttentionManager.get_num_skipped_tokens` | `single_type_kv_cache_manager.py:L741-L785` | 保留向下取整到 chunk；docstring 三例压缩为一例 | chunk 边界 skipped 语义 |
| `SlidingWindowManager.find_longest_cache_hit` | `single_type_kv_cache_manager.py:L512-L604` | 删 use_eagle 多匹配/丢尾、dcp/pcp | 右到左 contiguous 命中查找 |
| `get_kv_cache_coordinator` | `kv_cache_coordinator.py:L594-L642` | 保留三态分支；删 eagle/events/dcp/pcp/metrics 透传 | 协调层入口三态工厂 |
| `HybridKVCacheCoordinator.find_longest_cache_hit` | `kv_cache_coordinator.py:L487-L591` | 保留不动点 while 循环 + full 早停 + simple-hybrid 一轮 + 末尾截断；删 eagle/多块尺寸换算 | 本章核心：不动点收敛 |
| `HybridKVCacheCoordinator.verify_and_split_kv_cache_groups` | `kv_cache_coordinator.py:L436-L485` | 保留分桶 + full 排首 + lcm；删 eagle_attn_group_indices | 构造期拓扑解析 |
| `get_manager_for_kv_cache_spec` | `single_type_kv_cache_manager.py:L1155-L1173` | 原样保留 admission cap 注入 | spec→manager 映射 + 准入上限单一真相源 |
| `spec_manager_map` | `single_type_kv_cache_manager.py:L1142-L1152` | 保留三类注意力条目；其余 spec→manager 条目删（spec 类已 SUBTRACTED） | 多注意力类型分发注册表 |

## 减法清单（仅 dossier `subtraction_plan.delete` 批准项）

- `full_sequence_must_fit` 分支：**保留**（dossier 要求保留 `apply_admission_cap` 语义链路），
  仅未在主路径测试中默认启用。
- metrics / kv_cache_events / log_stats / prefix_cache_stats：ch15 已删，纯观测。
- Mamba align 模式 / CrossAttention encoder-decoder 体 / SinkFullAttention sink 块摘取：
  仅保留类壳 + spec_manager_map 条目（条目本身随其 spec 类一并删，见 map 注释），
  内部专用体 SUBTRACTED。
- dcp/pcp > 1 并行扩展、eagle/投机解码在 find_longest_cache_hit 的分支、
  block_size != hash_block_size 多块尺寸换算：默认不触发的旁路分支。

## 测试

`tests/test_kv_cache_alloc.py`（26 例，host 纯单元，不 import vllm）覆盖：三态工厂、
spec→manager 映射 + admission cap 注入、三类 get_num_skipped_tokens 差异、
remove_skipped_blocks 逆序换 null + 幂等早停 + 全注意力 no-op、allocate_slots 端到端、
lookahead 不入缓存 + 预留槽位、external 命中分配真实块、可驱逐命中块计入预算、
admission cap 夹取、Unitary 委托、Hybrid 分桶 full 排首 + 不动点收敛到一致命中长度。

跑法：`python3 -m pytest instances/vllm/artifacts/ch16-kv-cache/tests/ -q`
