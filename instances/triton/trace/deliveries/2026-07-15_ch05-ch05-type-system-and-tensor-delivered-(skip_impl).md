# ch05 type-system-and-tensor delivered (skip_impl)

- **Type**: delivery
- **Chapter**: 05
- **Date**: 2026-07-15
- **Timestamp**: 2026-07-15T20:09:50Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch05, skip_impl, type-system, fp8, dtype, tensor, cast, bitcast, backend-capability-seam

## What happened

第五章《值在 Triton 里长什么样：三层类型、tensor 与 cast》交付。kind=skip_impl(无精简版)。八机制三层齐(直觉/机制/不变量，性能相关配量化)：①dtype/pointer_type/block_type 三层套娃，都实现 to_ir 从内向外下降(core.py)②fp8 家族 8bit 三元组(fp_mantissa_width/primitive_bitwidth/exponent_bias)——同位宽不同分法、尾数↔量程守恒③to_ir 大 dispatch 且开头查 supported_fp8_dtypes，后端能力接缝另一端 CUDABackend.parse_options 按 capability 动态拼(nvidia/backend/compiler.py)④validate_block_shape 两道门(每维 2 的幂+numel≤2^20=1048576，_utils.py)⑤tensor=(handle,type) 提货单、shape/numel/dtype 全派生、shape 裹 constexpr、约四十 dunder 全转发 semantic 零决策⑥semantic.cast 大 dispatch 每支发一个真 IR op、bf16↔非fp32 借道 fp32 两跳⑦bitcast 等宽位重解释(tt.bitcast，位宽不等即 ValueError)。pin v3.2.0 headless 编译取证(三元组字段、validate 四 case、cast/bitcast 追踪期 IR op 计数)。5 张图全 blind PASS(fig-fp8-tradeoff 曾因 claim 与图不符 FAIL、修 claim 后回填；fig-three-layer-types 整改 pointer_type 输出框误绑的 fp8 剧透注解)。全部门禁绿、GitHub 渲染 0 未渲染。pipeline 在 review-exhausted 后退出(唯一 blocking 是 chapter-map 缺失，Lead 补齐并过盲审)，未走到 Archive 站，本次手工归档。figure-timing 死锁经 Lead 修 writer 契约根治。

## Why it matters

本章把『一个值在 Triton 里长什么样』拆到底，立起后续所有类型操作的地基：三层类型+to_ir 下降链是 ch06 类型提升/ch07 造块/ch19 tt.* 词汇表的前提；fp8 三元组与 cast op 计数把 dtype 选型讲成可落手的性能账(带宽砍四分之一↔精度粗两数量级)；supported_fp8_dtypes 按 capability 拼是后端能力接缝、ch36 CUDABackend 兑现；tensor=(handle,type) 提货单心智模型贯穿全书追踪期。

## What to remember

ch05 done(skip_impl)。plant f6(fp8 后端能力接缝→ch36)/f7(cast 真开销+dunder 全转发→ch06 类型提升)入 arc-map。5 图入 figures.json(chapter-map/fig-three-layer-types/fig-fp8-tradeoff/fig-tensor-handle-type/fig-cast-dispatch)，16 术语入 glossary(总 40→56)，14 概念入 concepts。run-ledger 手工补(write_review_rounds=2，escalated=review-exhausted chapter-map 缺失)。ch05 无应回收伏笔(bible.py due ch05 空)。
