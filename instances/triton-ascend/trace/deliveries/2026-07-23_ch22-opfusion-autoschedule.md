# 算子融合与自动调度：FusionKind 分类与 Cube/Vector 分工的 tile 策略（deep+skip_impl）

- **Type**: delivery
- **Chapter**: ch22
- **Date**: 2026-07-23
- **Timestamp**: 2026-07-23T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist（Review 站逃逸后 Lead 手动接管修复）
- **User present**: False
- **Tags**: triton-ascend, part-5, deep, skip_impl, hivm-hfusion, opfusion, autoschedule, fusion-kind, shallowcv, review-escape

## What happened

Part 5「硬件 IR HIVM」第三站，hivm-hfusion 子系统承 ch21，deep+skip_impl（纯 C++ MLIR pass 章，`.cpp` 在 AscendNPU-IR/bishengir submodule 内，无精简版）。ch21 讲清了 `FusionKind` **是什么**（10 值融合意图枚举、`InferFuncFusionKind` 推断、贴 func 级），本章讲它**驱动什么**——一书两条脊梁：

**脊梁① OpFusion（`OpFusion.cpp`，按 FusionKind 定融合边界）**：入口读回融合块的 `FusionKind` 印章路由（§一）；核心 `isFusible`（`FusibleHelper.cpp:L557-582` 的 **9 分支 switch**）按 kind 给融合兼容表——ShallowCV 格 14×14 含 `kMatmul`（L673-712，14 case）、MixCV 格 **10×10 非 matmul 子块对称可融 + matmul 独立不对称规则**（patternB 仅 elementwise/zeroRank-elemwise，reduce/broadcast 被拒，L719-758）、SingleCube 恒 false（L576-578）（§二）；`fuseBlock`（`FusibleBlockAnalyzer.cpp:L177-221`）用**并查集 + 拓扑秩**主循环把算子合并成连通分量、逐轮追踪一定停且分组不重叠（§三）；`verifyRulesAndJoin`（L86-147）**五道关卡**逐边把关、换 kind 就换判定（§四）；`checkGroupRequirements`（L149-173）**出组约束——ShallowCV/MixCV 必须含 matmul**（`matmulCount>0`）否则踢出（§五）；`FusibleBlockOutliner`（L211-235）把融合块**外提成 device func 并回写 FusionKind 印章**（§六，与 ch08 outline-scope 同一 region→func 思路）。worked example `@testA` 9 算子 → 三连通分量，仅含 matmul 的 {3,5,7,9,11}(5 op) 保留成 `@testA_0`，两个纯 vector 分量因 matmulCount=0 被踢。

**脊梁② AutoSchedule（`AutoScheduleBase.cpp`，按 FusionKind 切 tile 分双核）**：`applySchedule`（L579-611）读回 `FusionKindAttr` **switch 选 Scheduler 子类**——PBR 家族 4 种（PureElemwise/AnyPB/LastAxisPBR/AnyPBR）共用 `AnyPBRScheduler`、`SingleCube`/`ShallowCV` 各有专属 scheduler、`ShallowVV` 走 no-op（0 实例化），去重后 **4 类** Scheduler；含 cube 的三种 kind（MixCV/SingleCube/ShallowCV）`blockDim` 减半（L1221-1231 `max(blockDim/2,1)`）（§七）；调度骨架 `pre→schedule→post` + tiling-key 分派（host tiling 函数 + `scf.index_switch` 多 tiling case）（§八）；样例 `ShallowCVScheduler`（`ShallowCVSchedule.cpp:L40-65`，夹具 `test-shallow-cv.mlir`）**二次拆分**——3 个 cube 段（`matmul_transpose_b` %1/%8/%15）留原核 + 3 条 vector 链外提，vector 算子 8（bcast×3+add×3+max×2），blockDim 40→20（§九）；对位基座与小结（§十）。5 图（4 机制图 + 本章地图）全 blind PASS，16 门禁全绿。

**交付曲折（Review-escape，如实）**：多维评审 round1 抓出 **3 处 blocking**、workflow 触发 review-escape 未跑到 Archive → Lead 手动接管：①**Scheduler 计数陷阱 15→4**（正文把 switch 分支/scheduler 家族误算成 15，去重实数 4，writer 定点改，与 fig-schedule-dispatch 对齐）；②**子核个数免责**（fig-shallowcv-split 把外提 vector/cube 子核个数当已验证事实硬计数，但精确点名需实跑 bishengir-opt、host 无工具链，illustrator 清硬计数、只画 IR 可数结构量 3 cube 段/3 vector 链 + 补免责行）；③**MixCV 10×10 结构**（fig-fusible-dispatch 残留 14×14/收紧一行错误措辞，illustrator 改成 10×10 非 matmul 对称 + matmul 独立不对称规则，逐 case 核 L719-758）。两图内容实质变更 → blind_review 重置 PENDING → **独立盲审（未参与作图者 Read PNG 核源码）PASS**；Lead 复核本章地图 PASS；writer 补插图引用。write↔review 2 轮、blind 2 轮（重置后 PASS）、map 1 轮。verdict APPROVED，5 issue 全 non-blocking。

## Why it matters

ch22 兑现 f5（ch21→ch22）：把 ch21 只点名的 `FusionKind` 从「一个枚举」坐实成「贯穿融合 + 调度两大 pass 的路由键」。它是全书讲清 hivm-hfusion **调度决策**的权威章——OpFusion 回答「哪些算子融进一个核」、AutoSchedule 回答「这个核怎么切 tile、cube/vector 怎么分工」，两条脊梁都以 FusionKind 印章为唯一路由依据，`FusibleBlockOutliner` 的「外提 + 回写印章」正是两脊梁的接缝。对位基座《Triton 源码解读》的 fusion/scheduling——但昇腾是异构双核（cube+vector），故多出 ShallowCV/MixCV 这类「cube 与 vector 互融」的融合类型与 blockDim 减半这类双核特有决策。

## What to remember

- **计数口径钉死（承 ch21 纪律）**：按 kind 分派的 **Scheduler 子类去重后是 4**（AnyPBR 家族共用 1 + SingleCube + ShallowCV + ShallowVV no-op 不实例化），不是 switch 分支数 15——这是本章 round1 blocking B1，正文严禁把「switch 分支数」当「scheduler 类数」。`isFusible` switch 9 分支、MixCV 10×10、ShallowCV 14×14、SingleCube 恒 false，逐 case 数实（FusibleHelper.cpp:L557-582/L673-712/L719-758），不靠 brief。
- **诚实边界（本章 blocking B2 的教训）**：外提后**精确的 vector/cube 子核个数**需实跑 `bishengir-opt`（host 无 CANN 工具链）才能逐一点名，图/正文一律**只画 IR 里可数的结构量**（3 个 cube 段 matmul_transpose_b + 3 条 vector 链），并挂免责行——别把「需真机才能确定的数」当已验证事实硬写。
- **交叉验证靠 pin-exact 源码 + lit 夹具**：skip_impl 无精简版；『融合/调度后长这样』取自项目自带 lit 夹具（`test/Dialect/HFusion/AutoSchedule/test-shallow-cv.mlir` 等）的 IR 期望 + pin `@2badfc89e` 逐段核对 `FusibleHelper.cpp`/`FusibleBlockAnalyzer.cpp`/`FusibleBlockOutliner.cpp`/`AutoScheduleBase.cpp`/`ShallowCVSchedule.cpp`，不伪造编译器 dump。
- **伏笔**：**回收 f5**（ch21→ch22：FusionKind 十种调度差异，status=resolved，resolved_in=ch22）；本章**未埋新伏笔**（dossier.foreshadow_due.plant 为空）。
- **Bible 回写**：glossary +6（OpFusion 脊梁① / AutoSchedule 脊梁② / Scheduler 调度师 / ShallowCV 浅配合 / outline 外提成核 / verifyRulesAndJoin 五道关卡，现 245 键；既有 FusionKind 词条已含 ch22 兑现口径）；concepts +10（现 262）；figures +5（fig-fusible-dispatch/fig-fuseblock-testA/fig-schedule-dispatch/fig-shallowcv-split + fig-ch22-chapter-map，均 blind PASS，现 133）；interfaces 不新增（skip_impl 无精简版）；arc-map f5 → resolved。
- **run-ledger**：`reviews/run-ledger.json` 记 Review round1 逃逸 + Lead 手动接管（正文 3 处 + 两图 + 独立盲审 + Lead 核地图 + 插图引）+ blind 重置 PENDING→PASS。
