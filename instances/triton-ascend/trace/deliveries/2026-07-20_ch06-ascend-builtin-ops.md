# ch06 交付：昇腾内建算子——索引搬运、向量算子与定制 cast

- **Type**: delivery
- **Chapter**: ch06
- **Date**: 2026-07-20
- **Timestamp**: 2026-07-20
- **Agents involved**: analyst, implementer, tester, explainer, illustrator, writer, ch06rev, ch06fix, ch06map, Lead, archivist
- **User present**: false
- **Tags**: triton-ascend, part-2, deep, language-layer, mem-ops, index-boundary, zero-copy, address-space, vec-ops, ascend-cast, annotation-mark, dialect-prefix-correction

## What happened

Part 2 语言层第三章（物理章号 ch06），kind=**deep**（含只做减法的 implementation/ 精简版 + tests），deps=ch05。正文 1125 行，10 项章级门禁全绿，精简版 **37 tests passed**，7 图（6 机制图 + chapter-map）blind_review 全 **PASS**。

**主线：四个 mem_ops 是「裸指针世界 ↔ buffer 世界」的接缝。** 地基是**地址空间四档口径**（ch05 立、ch06 沿用）：`HIVMAttrs.td:L188-L194` 定义 **7** 级 address space，但 `third_party/ascend/ascend_ir.cc:L412-L418` 的 `py::enum_` **只导出 5 级**（L1/UB/L0A/L0B/L0C）——**Zero 与 GM 不进 Python**，kernel 里写不出 `space=GM` 的 buffer。于是跨 GM 的带索引访问只能由内建承担，它们的签名永远「一头裸指针、一头 UB tile」。

**十四机制要点**：
1. **接缝**（m1）：AddressSpace pybind 枚举 → GM 只有地址、片上才有名字。
2. **三级映射**（m2 `gather_out_to_ub`，`mem_ops.py:L180-L329`）：index 格子 → 源坐标 → 字节地址；`index_boundary` 把越界格摘出换 `other`。结果与 index 同形这条不变量由 C++ `create_gather_out_to_ub`（`triton_ascend.cc:L208-L235`）拼 `RankedTensorType` 钉死，非 Python 侧约定。
3. **反向搬运**（m3 `scatter_ub_to_out` 标量 value 自动广播；m4 `index_put` index 摊平成 1D + `dim<rank-1` → 越界作废一整条）。
4. **3 对 1 分野**（m5）：`index_put`/`gather_out_to_ub`/`scatter_ub_to_out` 带 `index_boundary`，**`index_select_simd` 没有**（docstring 明写不查越界）——零拷贝 tile 选取**买到粒度、卖掉越界检查**；占位协议 `read_shape[dim]==-1`、`src_offset[dim]` 可为 -1、超界自动截断。
5. **位宽契约（已知不一致，疑似 bug，已写进正文）**（m6）：gather/scatter **硬编码** stride=i64 / offsets=i32；`index_put` 却用 `require_i64 = index.dtype.is_int64()` **一个开关同时决定三者**（`_utils.py:L36-L54`）。所谓「统一契约」是假象。
6. **片上词汇表**（m7/m8）：`insert_slice`/`extract_slice`/`get_element` 落**上游 tensor 方言**（InsertSliceOp/ExtractSliceOp/ExtractOp），不在昇腾方言。
7. **同 API 两条路**（m9 `flip`）：SIMD 一条 `ascend.flip`；SIMT 无此算子，退化 log2(n) 轮 reshape+xor-swap。
8. **sort**（m10）：只排末维（对照 `index_select_simd` 偏偏不准选末维）、dtype 白名单、int8/int16 自动加饱和提示。
9. **cast**（m11 `ascend_cast_impl` 整个顶替基座 `semantic.cast`）：bf16/fp16→非 fp32 拆两跳经 float32；saturate 整型收窄按芯片分「挂提示」/「绕道 fp32」。
10. **不换算子只贴便条**（m12）：`overflow_mode` 校验列表只认 `trunc`/`saturate`，**docstring 拼成 `sautrate`**；实现经 `create_annotation_mark` 挂 `annotation::MarkOp`（另一个 builder 经清单挂载，`extension/builder.py:L63-L86`）。
11. **三种写法 + 落点表**（m13/m14）：同一个 `index_select` 手写基线发 5 个算子 / 内建发 1 条 / 什么都不做由编译器改写成 `ascend.indirect_load`；落点表把每个 builtin 归到昇腾方言 / 上游 tensor 方言 / annotation 方言。

**流水线经过**：以 `skip_archive=true` 并行发车。Write 站曾因**环境守卫逃生**中止，Review/Map 由 Lead 补派命名 agent（ch06rev/ch06fix/ch06map）完成，`reviews/review-report.json` 已落盘。首轮 verdict=**REVISE**（7 blocking / 8 non-blocking），7 条已全部修完：B1 内嵌源码被改一行（违反逐字契约且改出 bug）、B2 「算子条数 5」无法就地验证且与落点表自相矛盾、**B3 `tt.indirect_load` 应为 `ascend.indirect_load`**、B4 小结整节无源码引用（lint_source_grounding 机械 BLOCKING）、B5-B7 reader-comprehension 聚集升级组（按 exp-2026-07-18-01 触发一次 write↔review 回环）。评审同时记：Lead 列的 7 个高风险点 6 个完全 PASS、14 机制 13 个逐项过账、数值表全部手算复核无误，**取证纪律（无一处真机数值）是同书目前最严的一章**。

**取证口径（三级，正文明写）**：① 仓库自带 `python/triton/runtime/ascend_interpreter.py` 解释器参考实现（读作「按定义应该算出什么」）；② 跑精简版记录的 builder 调用序列——真实 builder 需 CANN 编 C++ 绑定，host 上由只记账、返回哨兵值的 **FakeBuilder** 站位，故读作「前端校验全过、走到建 op 这一步」，不是真机 emit；③ 官方用例 kernel 按循环结构逐句复刻的 numpy 版。凡「越界会读到什么」这类真机才能定论的，一律标参考实现行为、真机未验证。

## Why it matters

ch06 把 Part 2 语言层的另半边（带索引的跨 GM 搬运 + 片上向量算子 + 被改写的 cast）钉死成一张**落点表**：每个 Python 内建 → 哪个 `create_*` → 落哪个方言的哪个算子。后半程讲下降（P4/P5）时，这张表就是「从哪儿来」的索引。**地址空间四档口径**（.td 7 档 / pybind 5 档，GM 不进 Python）是全书凡讲 buffer 归属与搬运方向都要引的地基事实。

**方言前缀订正必须传下去**：昇腾方言算子前缀是 **`ascend.`** 不是 `tt.`（`TT_Ascend_Op` 绑 `TritonAscend_Dialect`，其 `let name = "ascend"`）。本轮 `tt.indirect_load` 这个错名曾传播到正文、两张图、explainer、traces 共 **6 处文件**才被评审逮住——已登记进 Bible 术语表，后续章节写任何昇腾方言算子名前先核这条。

## What to remember

- **诚实边界**：host 无 NPU/CANN，全章无一处真机数值；三级取证口径见上。
- **本章无 arc-map 正式伏笔埋/回收**（`bible.py due ch06` 两个清单皆空）。前向线索记此：落点表 → P4/P5 下降章；`ascend.indirect_load` 的自动改写 → 编译器 pass 部分；`annotation::MarkOp` 承载的 hint 通道 → 与 ch03 埋的 `compile_hint`（`aux_ops.py:L114-L133`）在 P2 ch08 汇合。
- **事实校准点（勿再回退）**：①`.td` 7 档 / pybind 5 档，Zero 与 GM 不进 Python；②`index_boundary` 3 对 1，`index_select_simd` 不查越界；③位宽两套写法（一个开关 vs 两处硬编码），是**疑似 bug 而非设计**；④`insert_slice`/`extract_slice`/`get_element` 落**上游 tensor 方言**；⑤`overflow_mode` 文档拼作 `sautrate`，以校验列表为准。
- **可复用经验**：读演进中的扩展算子代码，**能对照校验列表就别信 docstring**（本章三处不一致全靠这条逮住）。
- Bible 回写：glossary +18 条、concepts +16 条、figures +7 条、interfaces 登记 ch06 精简版签名（按真实包树 `implementation/python/triton/…` + `implementation/third_party/ascend/…`，规范路径前缀）。
