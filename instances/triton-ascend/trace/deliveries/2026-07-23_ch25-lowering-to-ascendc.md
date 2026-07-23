# ch25 交付：下降链收官——HFusion→HIVM→Standard，从融合张量 op 到 AscendC 库调用

- **Type**: delivery
- **Chapter**: ch25
- **Date**: 2026-07-23
- **Timestamp**: 2026-07-23T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist, **Lead（Review round3 逃逸后手动接管）**
- **User present**: False
- **Tags**: triton-ascend, part-5, deep, skip_impl, flagship, hivm-hfusion, HIVMToStandard, createLibCall, getOpLibraryCallName, reduceMemrefsToNestedFor, partial-conversion, AscendC, review-escape, lead-takeover

## What happened

Part 5「硬件 IR HIVM」第六站、**hivm-hfusion 子系统收官章（★flagship）**，承 ch23（HIVM 方言：被降的 op 与六级内存）+ ch24（HIVM 显式同步：降之前 IR 已插好 `set_flag`/`wait_flag`）。kind=deep，纯 C++ MLIR conversion pass 章（无 `implementation/` 目录，skip_impl）。核心 pass `ConvertHIVMToStandardPass`（`HIVMToStandard.cpp`，1933 行）：`runOnOperation` 立白名单（`func`/`scf`/`memref`/`arith` 合法）+ 黑名单（49 个 HIVM 硬件 op 全非法），逐设备函数跑 `applyPartialConversion`，把带同步的 HIVM 硬件 IR 一路降成对 AscendC 运行库的 `func.call`——降完 HIVM 方言在 IR 里彻底消失，这是全书结构化下降链的终点。

8 机制（m1-m8）：**m1** `createLibCall`/`replaceWithLibCall`——先 `lookupSymbol` 查重（同名库函数全局只声明一条），查不到才 `getOrInsert` 插一条外部声明，再 `emit` 一条 `func.call` 替换原 op；**m2** `getOpLibraryCallName` 库函数名 mangle——按 op 名/rank/元素类型/内存域/对齐（`isOpsAligned`）确定性拼名，是确定性函数（同输入必同输出）；**m3** `reduceMemrefsToNestedFor`——rank 超过库 `maxRank` 时按多出的轴拆嵌套 `scf.for` + `subview` 逐元素调库，调用次数=拆开各轴 dim 之积、深度=rank-maxRank，附基例/归纳步的停机性论证；**m4** pattern 三形态穷尽 49 个 op——形态 A 直接继承 `OpRewritePattern`（如 `MmadL1OpToLibraryCallPattern`），形态 B/C 都挂在同一个 `MultiDimOpToLibraryCallPattern` 之下（rank 门控 vs 按语义轴拆循环），`patterns.add<…>` 里只有这两支血统，不存在第四种；**m5** `ConvertHIVMToStandardPass` 主流程（partial conversion 收敛性论证：非法 op 只减不增故必然终止）；**m6/m7/m8**（supporting）同步参数透传、类型规范化（memref 统一动态 strided layout）、下降链收官在流水线中的位置（`buildOptimizeHIVMPipeline` 末站 + `convert-to-hivm-pipeline` 上游喂料）。贯穿真实全链夹具 `test/Dialect/HIVM/hivm-pipeline.mlir`（1 维 `memref<16xf16>` 的 load/vadd/store 三 op 函数，从 HIVM 到 `func.call @vadd_1d_half` 等全程跟随）。

**评审记录（如实归档）**：`review-report.json` verdict=APPROVED，7 条 issue（task 口径的 2 阻断项：`fig-m1-libcall` 图内维度 6 维与库名 `vadd_1d_half` 矛盾、"48 段规则"与全章其余 8 处"49 个 op"自相矛盾的数字笔误；5 条 non-blocking：§25.6 三形态类层级措辞、L318 裸文字章号、fig-m1 图上"hivm.vadd"缺"hir"、isOpsAligned 铺垫与代码传 nullopt 脱节、§25.1 预览未体现同步参数透传）。

**交付曲折（review-escape + Lead 接管，如实）**：多维评审进 **Review round3** 的 revise-fig-caption 阶段时 **API 崩溃（Connection closed）**，workflow 逃逸未到 Archive。Lead 手动接管补完：①`fig-m1-libcall` op 名统一 `hivm.vadd`→`hivm.hir.vadd`（图标题/左列标题/红框 op 行/replaceOp 底注/图注共 5 处三段式命名）+ gen 脚本 `LEFT_OP_L2` 维度 6→1 维 `memref<16xf16, #hivm.address_space<gm>>`（与 `vadd_1d_half` 自洽）+ **重渲染** + **独立盲审 PASS**（复核人非作图/修复者，SVG 实测 5×`hivm.hir.vadd`、0×裸 `hivm.vadd`）；②dossier IR 名纠错 `hivm.PointerCastOp`→`hivm.hir.pointer_cast`（三段式方言助记符）；③正文 5 处 issue 经核实**前轮已修**（L545 "49 段规则"、L547 形态 B/C 同一父类下两种 `matchAndRewrite` 写法、L144 isOpsAligned 只在 `VBrc` 广播类 op 落名说明、L336 表注点破 mmadL1 行 f16→float 语义、§25.1 结尾同步参数/`memref.cast` 留到 §25.7 免责声明）；④2 阻断项（48→49 数字、fig-m1 维度）前轮已修、Lead 核实在盘。

**归档逐条复核：7 issue 全清**——fig-m1 SVG 5×`hivm.hir.vadd` 0×裸名、L320「第 23 章」等章号引用均为 markdown 链接（`../../ch23-hivm-dialect/narrative/chapter.md`）、chapter.md 全文无残留"48"/"45+"。run-ledger：write↔review 3 轮、blind 1 轮（4 图 0 failure）、map 1 轮 PASS；`escalated` 字段记录逃逸相/子相/轮次/Lead 接管动作。16 门全绿。

## Why it matters

ch25 是全书下降链故事的收束点：ch09（MLIR/Linalg 原理）→ch10-14（TritonToLinalg 结构化下降）→ch15-19（ascend-opt 网格/双核/流水）→ch20-22（HFusion 融合 IR）→ch23-24（HIVM 硬件 IR 与显式同步）→本章（HIVM→Standard，硬件 op 变库调用）。读者第一次亲眼看到"张量级算子"最终变成什么——不是 PTX 汇编（对位基座 part-7 ch32-35 的 TTGIR→LLVM→PTX 五级台阶），而是一条 `func.call @<mangle 出的库函数名>`，再由闭源 CCE 出二进制。三形态穷尽论证（"不存在第四种"）呼应 m5 的 partial conversion 收敛性论证，是全书"结构性必然、非巧合"这条方法论主线在下降链终点的最后一次示范。

## What to remember

- **核心口径**：HIVM 硬件 op → `func.call` 走 `createLibCall`/`replaceWithLibCall`；库函数名由 `getOpLibraryCallName` 按 op名/rank/类型/内存域/对齐确定性 mangle；rank 超限靠 `reduceMemrefsToNestedFor` 拆嵌套 `scf.for`；49 个非法 op 只有两支 pattern 血统（直接 `OpRewritePattern` / 共享 `MultiDimOpToLibraryCallPattern`），无第四种形态。
- **评审 issue 现状**：7 条（2 blocking + 5 non-blocking）**全部修复在盘**，归档逐条复核确认（fig-m1 diagram + chapter.md 均已更新，figure-manifest 4 图 blind_review 全 PASS）。此前遗留的两条（L318 裸章号、fig-m1 图内 op 名"hivm.vadd"缺 hir）已由 Lead 接管一并修完——章号现为 markdown 链接、fig-m1 SVG 5×`hivm.hir.vadd` 0×裸名。**无遗留 open issue**。
- **恢复史（review-escape，如实）**：Review round3 revise-fig-caption 阶段 API 崩溃（Connection closed）→Lead 接管补完：fig-m1 op 名统一 hivm.vadd→hivm.hir.vadd + 维度 6→1 维 + 重渲染 + 独立盲审 PASS；dossier IR 名纠错 hivm.PointerCastOp→hivm.hir.pointer_cast；正文 5 issue 经核实前轮已修；2 阻断项（48→49 数字/fig-m1 维度）前轮已修 Lead 核实。run-ledger `escalated` 字段留档。
- **无新伏笔**：`bible.py due ch25` 两清单皆空；dossier `foreshadow_due` 应埋/应回收均为空。arc-map 无变化（f1-f5 此前已全部 resolved）。ch25 是 hivm-hfusion 子系统收官章，下一章 ch26 转向 backend-runtime 子系统（AscendBackend 契约），非直接机制延续，故无需埋新伏笔。
- **Bible 回写**：glossary **+6**（`ConvertHIVMToStandardPass`/`createLibCall / replaceWithLibCall`/`OpWithLibraryFunction / getOpLibraryCallName`/`isOpsAligned`/`MultiDimOpToLibraryCallPattern`/`reduceMemrefsToNestedFor`，现 268 键）；concepts **+5**（现 288）；figures **+4**（m1/m3/m4 三张机制图 + chapter-map 登记为 `fig-ch25-chapter-map` 防跨章撞 id——吸取 ch23 漏登记教训，本章 4 图全登记，现 152 条）；interfaces 不新增（无精简版，同 ch15-24 先例）。
- **skip_impl 交叉验证口径**：无精简版；靠 pin 源码 + 真实全链 lit 夹具 `hivm-pipeline.mlir`/`convert-to-hivm-op.mlir` 逐行核对行号（fig-m4 盲审记录里对 `HIVMToStandard.cpp` 逐行核实了 7 组行号，无一处对不上）。
