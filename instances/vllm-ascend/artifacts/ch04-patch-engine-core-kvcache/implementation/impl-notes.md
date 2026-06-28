# ch04 精简版实现笔记 — 顶替引擎核心：KV-cache / 调度 / spec 的昇腾化 patch（subtract-only）

只做减法：与 vllm-ascend 真实源码**同名、同结构、同控制流**，只删不增。每个 `# SUBTRACTED:`
都标了删了什么 / 为何仍正确 / 原行号；每个 def/class 标 `# SOURCE: vllm_ascend/...:Lxxx`。
昇腾 NPU/CANN 代码 host 不可跑；**patch 重绑定 / 配置改写 / page-size 计算本身是纯 Python**，
故 `tests/` 用 stub 把 vLLM 目标命名空间放进 `sys.modules`，import（被减法后的）patch 模块，
断言重绑定与可观察计算如真仓库一样生效（5 案例全覆盖，host `pytest` 11 passed）。

只删除了 dossier `subtraction_plan.delete` 明确批准的 5 项；`must_keep` 的 21 个符号全部保留
（lint_fidelity 校验通过）。

## 1:1 Source Map

| 精简版文件 | 源码（规范路径） | 案例 / 招式 | 主要减法（仅 delete 批准项） |
|---|---|---|---|
| `patch_mamba_config.py` | `vllm_ascend/patch/platform/patch_mamba_config.py:L1-L119` | 案例1 block_size 16→128，技法③方法替换 | 删末尾 mamba page size padding 百分比日志 + `using_kv_transfer_with_hybrid` / `mamba_cache_mode='align'` / `mamba_block_size` 设定（L24-26, L93-116）；保留 `kernel_block_size=128`、`attn_block_size` 对齐恒等式 + assert、写回 `cache_config.block_size`、末行重绑 |
| `patch_kv_cache_interface.py` | `vllm_ascend/patch/platform/patch_kv_cache_interface.py:L1-L266` | 案例2 MLAAttentionSpec 子类化，技法①整类替换 + ⑤别名重绑 | 删 `AscendMLAAttentionSpec.sparse_kv_cache_ratio` 属性及内嵌 `get_sparse_head_dim_virtual`（L91-160）；保留 Sparse-C8 4 元组 `page_size_bytes`、字段、`merge`、`max_memory_usage_bytes`、`AscendSlidingWindowMLASpec`、三处模块属性重绑 |
| `patch_kv_cache_coordinator.py` | `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L1-L368` | 案例3 主线 CP+hybrid 前缀缓存，技法①+②/③+⑤ | 删 `find_longest_cache_hit` 不动点迭代扫描细节（L225-281）；保留方法签名/docstring/`_get_block_hashes`、`_get_effective_block_size` 调用点、末尾按有效块截断；完整保留去 dcp/pcp 断言的 `__init__`、`verify_and_split_kv_cache_groups`（`lcm_block_size`）、工厂 `get_kv_cache_coordinator`（含回落 `_orig`）+ kv_cache_manager from-import 补绑 |
| `patch_kv_cache_utils.py` | `vllm_ascend/patch/platform/patch_kv_cache_utils.py:L1-L258` | 案例3 配套 resolve_kv_cache_block_sizes，技法③+⑤ | 删 DeepseekV4 专用布局函数体 `group_and_unify_kv_cache_specs` / `_get_kv_cache_groups_uniform_groups` / `_get_kv_cache_config_deepseek_v4`（L68-247，保函数名+动机占位）；完整保留 `_ascend_resolve_kv_cache_block_sizes`（lcm×dcp×pcp / gcd 替换 PR#40860 的 raise）+ kv_cache_utils/engine.core 双重绑 |
| `patch_qwen3_next_mtp.py` | `vllm_ascend/patch/worker/patch_qwen3_next_mtp.py:L1-L50` | 案例4 bind_kv_cache 跳 NPU raise，技法③方法替换 | 无 delete 批准项 → 全文保留（每 layer_index 取 `layer_names[0]` 绕过原 `NotImplementedError`；`extract_layer_index` 分组；`utils.bind_kv_cache` 末行重绑） |
| `block_table.py` | `vllm_ascend/worker/v2/block_table.py:L19-L105` | 案例5 int32 slot_mapping，子类覆盖（非 patch） | 删 `compute_slot_mappings` 的 triton kernel 调用体 + `@triton.jit _compute_slot_mappings_kernel`（L86-178，占位）；保留 `AscendBlockTables.__init__` 中 `del self.slot_mappings` → int32 重建（reshape_and_cache 要求 int32） |

> 命名校正：源码类名为 `AscendBlockTables`（复数，子类 `BlockTables`），dossier 文字写作 `AscendBlockTable`——精简版以**真实源码**为准用 `AscendBlockTables`，must_keep 符号 `slot_mappings` 保留。

## 测试映射（tests/test_kvcache_patches.py，host pytest，11 passed）

| 测试 | 验证的真实可观察行为 |
|---|---|
| `test_mamba_config_pins_kernel_block_size_128` | 案例1：重绑 `verify_and_update_config` 后，`kernel_block_size=128` → `attn_block_size=128*cdiv(ssm,128*attn_token)`，对齐恒等式成立、写回 `cache_config.block_size`（非 16） |
| `test_mla_spec_sparse_c8_page_size_bytes` | 案例2：Sparse-C8 A3 路径 `page_size_bytes = kv_bytes + qli_bytes(int8) + qli_scale(fp16)`（4 元组），与非 sparse 路径分流 |
| `test_mla_spec_rebinds_three_namespaces` | 案例2：技法①两处 kv_cache_interface 重绑 + 技法⑤ mla_attention from-import 别名重绑 |
| `test_sliding_window_mla_real_page_size` | 案例2：`AscendSlidingWindowMLASpec.real_page_size_bytes = storage_block_size*heads*head_size*size(dtype)` |
| `test_coordinator_effective_block_size_cp_and_compress` | 案例3：`_get_effective_block_size = block_size × (dcp*pcp) × compress_ratio`；MambaSpec+caching 短路返回原块 |
| `test_coordinator_factory_falls_back_to_upstream` | 案例3：非 deepseek_v4 + 单组/无 CP → 回落 `_orig_get_kv_cache_coordinator`（最小侵入） |
| `test_coordinator_rebinds_and_cache_trap` | 案例3：技法②/③工厂重绑 + 技法⑤ kv_cache_manager 早绑引用补绑 |
| `test_resolve_block_sizes_cp_multi_group_lcm_gcd` | 案例3：多组+CP>1 不 raise，`scheduler=lcm×dcp×pcp / hash=gcd` |
| `test_resolve_block_sizes_single_group_and_no_cp_fallback` | 案例3：单组 `block_size×dcp×pcp`；多组无 CP 回落上游 + engine.core from-import 双重绑 |
| `test_bind_kv_cache_takes_first_layer_per_index` | 案例4：同 index 多 layer_name 取 `[0]`、不 raise、forward_context 全绑定 |
| `test_block_table_slot_mappings_is_int32` | 案例5：`del` 父类 int64 `slot_mappings` 后以 int32 重建（dtype/shape 校验） |

运行：`python3 -m pytest tests/ -q`（host 即可）。保真度：`python3 scripts/lint_fidelity.py <chapter_dir>` 全通过。
