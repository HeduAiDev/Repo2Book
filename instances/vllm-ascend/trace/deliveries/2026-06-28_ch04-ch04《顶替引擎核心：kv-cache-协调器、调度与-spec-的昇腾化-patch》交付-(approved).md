# ch04《顶替引擎核心：KV-cache 协调器、调度与 spec 的昇腾化 patch》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 04
- **Date**: 2026-06-28
- **Timestamp**: 2026-06-28T17:02:19Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch04, vllm-ascend, kv-cache, block_size, MLAAttentionSpec, DSA, Sparse-C8, CP-hybrid-prefix-cache, bind_kv_cache, int32-slot_mapping, rebinding

## What happened

ch04 完成全流水线并经 reviewer APPROVED。以 KV-cache/内存形态层 patch 为主线，三段式（原算法→昇腾约束→重绑定手法，复用 ch03 五技法语汇只点名）讲透五案例群：① block_size 16→128（mamba 在 NPU 不支持 16，patch_mamba_config）；② MLAAttentionSpec/SlidingWindowMLASpec 子类化扩展 DSA/Sparse-C8（3→4 元组 KV：int8 key+fp16 scale，page_size_bytes 重写）；③ CP+hybrid 前缀缓存（AscendHybridKVCacheCoordinator 去 dcp/pcp==1 断言 + _get_effective_block_size 有效块大小 + _ascend_resolve_kv_cache_block_sizes 用 lcm×dcp×pcp 替 raise + 工厂只在 CP+多组+缓存时回落替换，技法⑤双绑 kv_cache_utils/engine.core）；④ bind_kv_cache 绕非 CUDA 平台 NotImplementedError 取 layer_names[0]；⑤ int32 slot_mapping（AscendBlockTable 子类覆盖，reshape_and_cache 要求 int32）。对照基座 vLLM v0.21.0（vllm/v1/kv_cache_interface.py·kv_cache_coordinator.py·kv_cache_utils.py·worker/utils.py）核对被 patch 原函数。精简版只验可读控制流（纯 Python 重绑定/配置改写逻辑 host 可跑）。前向链接 ch16/ch22 KV 管理（不展开）。

## Why it matters

ch04 是 KV-cache/内存形态层旗舰案例章，确立『为正确性与硬件约束而 patch』的三段式分析范式，把 ch03 五技法语汇落到真实 KV 管理改造上；后续 ch16/ch22 KV 管理章节承接本章前向引用。

## What to remember

review APPROVED，12 条 issue 全 non-blocking/negotiable：8 条保真/教学润色（_ascend_resolve helper 名未字面落点但逻辑完整、MLA 分支折叠引用、DSA-MLA 3 元组泛化消歧、L381 不重复迭代引导句与 L390 表格衔接、4.3 三层收束句、收敛证明嵌套特例旁注、公式缩写符号锚定）+ 4 条 reader-comprehension（DCP/PCP、A3/A5、rank、reshape_and_cache、有效块大小先动机后代码）。bible 已登记 7 个 ch04 精简版接口。本章无应埋/应回收伏笔（bible due 空）；f2(check_and_update_config→ch05) 仍 open。
