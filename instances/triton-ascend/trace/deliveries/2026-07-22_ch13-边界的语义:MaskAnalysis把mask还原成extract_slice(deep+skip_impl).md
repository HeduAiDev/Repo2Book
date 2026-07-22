# ch13 交付：边界的语义——MaskAnalysis 把 mask 还原成 extract_slice

- **Type**: delivery
- **Chapter**: ch13
- **Date**: 2026-07-22
- **Timestamp**: 2026-07-22T21:30:00Z
- **Agents involved**: analyst, writer, illustrator, reviewer, archivist
- **User present**: False
- **Tags**: triton-ascend, part-3, deep, skip_impl, maskanalysis, extract-slice, subview, boundary-semantics, rectangle-intersection

## What happened

Part 3 第 5 章·deep+skip_impl（纯 C++ MLIR pass 章，承 ch12 结构化访存物化，deps=ch12）：《边界的语义：MaskAnalysis 把 mask 还原成 extract_slice》。把 ch12 §12.7 明确点过但未展开的「有 mask 分支」彻底讲清——`MaskAnalysis` 怎么把 mask 表达式解析还原成矩形边界，再发射成 `tensor.extract_slice`/`memref.subview`。13 个机制全覆盖：**m1 `MaskState` 五字段三形态**（标量/裸 range/矩形掩码，`isEmpty`/`isMask` 判定门槛）是数据结构基石；**m2 `parse` 递归下降 + `TypeSwitch` 11 分支**（`ExtSI` 透传、`DivSI` 已停用）是解析主干；**m3 `parseCmp`（本章心脏）**——`arith.cmpi` 5 种谓词（slt/sle/sge/eq/ne）把 range 的 (start,end) 与标量 bound 熔进 (offset,dim)，`cmpDim` 唯一性约束 + `ne` 特判要求 lhs 出自 `arith.select`；**m4 `parseAnd`→`minStates`** 逐维求矩形交，解释「为什么没有 `parseOr`」（并集非矩形）；m5 `parseMakeRange` range 叶子；m6 形状传播（`parseSplat`/`parseBroadcast`/`parseExpandDims`）；**m7 `clampToNonNegativeIndex`** 负维度只在常量时夹 0（非常量为 atomic UT 妥协放行）；**m8 `getExtractSlice`/`getSubview`** 把矩形配全 1 strides 发射 `tensor.extract_slice`（tensor 域）/`memref.subview`（memref 域）——核心洞察「结构化世界用切片表达边界，非 GPU 逐元素 predication」的落点；m9 scalar 分支（`parseAdd`→`addStates`）；m10 `parseSel` + `cmpi ne` 特判（select 障眼法）；m11 `runMaskAnalysis` 入口 + 插入点管理；m12 消费端衔接 `LoadStoreConverter`；m13 `eraseInsertedOps` 死代码回收。

5 张机制图（fig13-1 三形态卡片、fig13-2 递归下降流程、fig13-3 五谓词对照、fig13-4 矩形交示意、fig13-5 双域发射流程）+ 本章地图 chapter-map，独立盲审首轮 PASS（0 failure）、map 站 1 轮 PASS。write↔review 1 轮收敛，`lint_trace_consistency` 通过（正文数值推演表与 explainer 素材一致，零漂移），`lint_dossier`/`lint_explainer` 仅预期性 manual-trace warn（skip_impl 章无可跑精简版，宿主无 CANN，已声明走 pin 精确源码 + lit 夹具交叉验证）。verdict=**APPROVED**：逐机制勾选表（13/13 核对，6 个 core 机制三层/trace/invariant/量化全齐，7 个 supporting 机制按 dossier 的 `needs_figure`/`needs_worked_example` 标记如实覆盖）+ 8 条 non-blocking 建议，**0 条 blocking**。

## Why it matters

ch13 把「结构化路径怎么处理边界」这件事第一次讲透：GPU 式的逐元素 predication 在张量层是不透明的运行期决策，而 `MaskState` 把掩码表达式的语义收缩成一个可被后续 tiling/bufferization 识别的矩形子区间（offsets/sizes/strides），这是全书「结构化描述 → 可变换 IR」主张在边界处理上的具体例证。`parseAnd` 只支持交（矩形∩矩形=矩形）而没有 `parseOr`（矩形∪矩形一般非矩形）这一条设计决策，也直接铺垫了 ch14「结构化装不下时怎么办」的必然性——不是实现疏漏，而是矩形代数的表达力边界。

## What to remember

- **本章心脏**：`parseCmp`（`MaskAnalysis.cpp:L440-L558`）——唯一把标量 bound 熔进 `(offset,dim)` 的地方，5 种谓词各有剪尾/抬头/定点/全保的边界推导，且 `ne` 特判要求 lhs 出自 `arith.select`（衔接 m10「select 障眼法」）。
- **核心洞察落点**：`getExtractSlice`/`getSubview`（`MaskAnalysis.cpp:L133-L195`）把矩形发射成 `tensor.extract_slice`/`memref.subview`——结构化世界用「切片」表达边界，不是逐元素 predication；`getSubview` 是消费端 `LoadStoreConverter` 有 mask 分支实际调用的版本。
- **无 `parseOr` 的设计原因**：`parseAnd`→`minStates` 只支持矩形交（逐维 `newOffset=max`、`newEnd=min`、`clamp≥0`），因为两个矩形的并集一般不是矩形；不连续掩码（如两段 `cmpi` 用 `ori` 拼出）parse 直接失败，交给 ch14 的 Unstructure 兜底路径。
- **负维度处理的有意妥协**：`clampToNonNegativeIndex`（`L68-L79`）只对常量值夹 `max(0,·)`，非常量本可发 `max(value,0)` 但会让 atomic max/min 的单测挂掉，故原样放行——一处显式记录在源码注释里的正确性权衡。
- **遗留的 non-blocking 打磨项**（未来若有小修窗口可顺手做，不影响本章交付）：m6 形状传播缺一步最小数值轨迹；m11 内容分散在 §13.1/§13.9 两处；fig13-1/fig13-2 图内 provenance 行号偏差 1-3 行；§13.1 无编号小标题，与 ch11/ch12 逐节编号体例不一致；§13.4「同一 bound=10」表述与 `ne` 特判矛盾，且 `selOp` 检查先出现后解释（顺序颠倒）；「select 下沉」与「UCC 保护」两处术语点名但从未展开哪怕一句直觉。
- Bible 回写：glossary +8 条（`MaskState`、`MaskState::parse`、`parseCmp`、`parseAnd`→`minStates`、`clampToNonNegativeIndex`、`getExtractSlice`/`getSubview`、`runMaskAnalysis`/`runMaskAnalysisImpl`、`parseSplat` splat-as-mask 特判）；concepts +8 条（对应 m1/m2/m3/m4/m7/m8 的机制摘要 + 核心洞察一条 + splat-as-mask 一条）；figures +6 条（5 机制图 + chapter-map）；interfaces **不新增**（skip_impl 无精简版，无接口可注册）；arc-map **无变动**（`bible.py due ch13` 两清单皆空，本章无正式伏笔埋/回收——§13.9 小结指向 ch14 属顺序性前瞻，非需登记回收的具体承诺）。
- 诚实边界：host 无 NPU/CANN，交叉验证走 pin 精确源码（`MaskAnalysis.cpp`/`.h` @2badfc89e ~735 行）+ `unittest/Conversion/**/*.mlir` lit 夹具（如 `DiscreteMaskAccess`/`loadstore.mlir`），不伪造编译器 dump。下一站：ch14《结构化装不下时：Unstructure 兜底路径》。
