# ch14 交付：结构化装不下时——Unstructure 兜底路径与 gather/scatter 标量化

- **Type**: delivery
- **Chapter**: ch14
- **Date**: 2026-07-22
- **Timestamp**: 2026-07-22T00:00:00Z
- **Agents involved**: analyst, writer, illustrator, reviewer, archivist
- **User present**: False
- **Tags**: triton-ascend, part-3, deep, skip_impl, offsetanalysis, unstructureconversionpass, axisinfo, scalarize, gather-scatter

## What happened

Part 3 第 6 章·deep+skip_impl（纯 C++ MLIR pass 章，承 ch11-13 结构化成功路径，deps=ch12+ch13）：《结构化装不下时：Unstructure 兜底路径与 gather/scatter 标量化》。讲结构化下降链的对偶面——当 `OffsetAnalysis` 判定一个访存装不进「每维一个连续区间」的模子时，`--triton-to-unstructure` 怎么把它标量化成 `scf.for` 逐元素循环。11 个机制覆盖：**m1 AxisInfo 四态格**（`unstructured⊑structured⊑scalarlike⊑scalar`，声明顺序即偏序，`OffsetAnalysis.h:L76-L81`，计数已核对源码=4）；**m2 combineInfo 逐维 std::min + scalarLike 布尔传播**（一维离散污染整维）；**m3 transfer functions**（`parseMulI`/`parseLoad` 产 unstructured，`parseMakeRange`/`parseSplat` 产 structured/scalarlike，判据=仿射性是否保住）；m4 parse 递归分派驱动；**m5 兜底判定多级闸门**（isStructured 早退放行 / forceScalarize·scalarLike·fromTensorArg 第二道闸 / 32 字节对齐闸 / 离散 mask `select` 解包分支，共 5 类强制离散入口）；**m6 部分标量化 codegen**（unstructured 维建 `scf.for`、structured 尾维保向量切片）；**m7 代价量化**（O(1) 结构化 vs O(∏unstructured 维 size) 标量化，16×8 张量：结构化 1 次、部分标量化 16 次、完全标量化 128 次）；**m8 scalarLike load 快捷路径 `splatAndLoadScenario`**（单点 load + splat 广播，O(1) 对比真 gather 的 O(N)）；m9 A5 快路径 `ascend.indirect_load`/`indirect_store`（SIMT 模板分流，本章仅标注不展开）；m10 MakeTensorPtr 完全离散时按 stride 重算线性偏移（省略标注，偏薄）；m11 循环/分支态传播。

4 张机制图（fig-ch14-m1-lattice、fig-ch14-m5-triggers、fig-ch14-m6-partial-scalarize、fig-ch14-m7-cost）+ 本章地图 chapter-map，独立盲审 1 轮 PASS（0 failure，m1/m5 图曾各有一处将不等价态/相反检查点画混的问题，已由 illustrator 定点修复并复核通过），map 站 1 轮 PASS。write↔review 3 轮收敛。最终 4 维评审汇总 verdict=**APPROVED**，issues 清单保留 13 条（2 条 fidelity 行号/措辞偏差、3 条 algorithm-pedagogy 含 1 条曾为 blocking 的"四类完整性论断遗漏第 5 条 discreteMask select 解包路径"——章节正文已补充说明该分支、2 条 figure-integration 曾为 blocking 已修复、1 条 formula-structure 非阻断、3 条 reader-comprehension 非阻断），均已随本轮交付一并归档存证。

## Why it matters

ch14 是 Part 3「结构化下降链」的分水岭对偶面：ch11-13 讲的是指针→三元组→memref→切片的「成功」路径，本章讲同一套分析框架判定「装不下」时怎么优雅退化——四态格比布尔二分多留的精度（为区分单点轴与广播轴）、多级闸门的不漏判不误伤设计（早退门筛掉安全的，其余每条访存被某道闸接住）、以及 scalarLike 快路径把"广播不是 gather"这件事在 codegen 层面彻底分开处理，都是"结构化优先、标量化兜底、按需优化"这一设计哲学的具体例证。也为 Part 3 后续昇腾优化 pass（blockify、HFusion/HIVM）埋下入口——本章标下的 `{DiscreteMemAccess}` 属性正是下游 pass 认领离散访存的标记。

## What to remember

- **本章心脏**：`matchAndRewrite`（`UnstructureConversionPass.cpp:L236-L306`）的多级闸门，尤其是易被忽视的第 5 条独立强制离散路径——`DiscreteMask` 属性（`discreteMaskAttrName`）命中时对 masked store/atomicRMW 的 `select` 解包（L284-L301），与早退门的 `is_discrete_mask`（`isDiscreteMask`）是同一条上游 `--discrete-mask-access-conversion` pass 打的两个不同属性名，容易和 `discreteAttrName`（跳过标记）混淆——全章反复提醒这几个"discrete"系名字之间的区别。
- **核心陷阱**：逐维 `scalarlike` 态与整张量布尔标志 `scalarLike`（`isScalarLike()`）是两个不同粒度、独立传播的量，`isStructured(dim)` 只认 `structured`/`scalar` 两态，不认 `scalarlike`——图（fig-ch14-m1-lattice）与文字都专门澄清，避免读者误以为三态等价。
- **代价量化落点**：16×8 张量结构化 1 次搬运、部分标量化（仅 dim0 循环）16 次、完全标量化 128 次；对齐闸的判据是 32 字节（昇腾内存搬运粒度）。
- **遗留 non-blocking 打磨项**（未来若有小修窗口）：§14.4 表格行号引用 L669-L671 应收紧为 L670-L671（L669 实际属前一分支）；§14.1"前三样承接自 PtrState"措辞对 scalarLike 字段级对应偏松（PtrState 无同名字段，只有 `isScalar()` 方法）；m3/m5/m7 三节"源码"层标签风格与其余机制不统一；m10 全程仅"省略"带过、未做最小说明；fig-ch14-m7-cost 柱标签"两维全离散"与 nested_loop 真实形状（单维 128）不符；§14.7 对齐闸反事实算例数字自相矛盾（尾维假设=4 但结论仍用 16×8=128）；fig-ch14-m1-lattice 图内批注框与脚注曾短暂矛盾（现已按盲审修复，问题已消解）；行内公式 `$`\mathrm{base}+\mathrm{stride}\cdot i`$` 被 lint 标复杂建议提升块级（非阻断）。
- **Bible 回写**：glossary +8 条（AxisInfo 四态格、`isStructured`/`isScalarLike`、combineInfo、matchAndRewrite 多级闸门、`{DiscreteMemAccess}`/discreteAttrName、DiscreteMask/is_discrete_mask、splatAndLoadScenario、ascend.indirect_load/store）；concepts +9 条（对应 m1-m9 机制摘要）；figures +6 条（4 机制图 + chapter-map，另 chapter-map 条目附带独立盲审通过说明）；interfaces **不新增**（skip_impl 无精简版）；arc-map **无变动**（`bible.py due ch14` 两清单皆空，本章无正式伏笔埋/回收，末节对下一章"blockify"的指向属顺序性前瞻，非需登记回收的具体承诺）。
- 诚实边界：host 无 NPU/CANN，交叉验证走 pin 精确源码（`OffsetAnalysis.cpp/.h` + `UnstructureConversionPass.cpp/.h` @2badfc89e）+ `unittest/Conversion/General/TritonToUnstructure/*.mlir` lit 夹具（如 `unstructure_mix.mlir`/`splat.mlir`/`nested_loop.mlir`），不伪造编译器 dump。下一站：ch15，从「把多个网格实例折成一条 blockify 循环」讲起，正式进入昇腾侧优化 pass。
