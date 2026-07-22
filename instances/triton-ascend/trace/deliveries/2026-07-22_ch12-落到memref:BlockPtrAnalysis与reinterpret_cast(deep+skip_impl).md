# ch12 交付：落到 memref——BlockPtrAnalysis、reinterpret_cast 与 load/store→linalg

- **Type**: delivery
- **Chapter**: ch12
- **Date**: 2026-07-22
- **Timestamp**: 2026-07-22T20:10:00Z
- **Agents involved**: analyst, writer, illustrator, reviewer, archivist
- **User present**: False
- **Tags**: triton-ascend, part-3, deep, skip_impl, blockptranalysis, reinterpret_cast, memaccess, load-store-lowering, atomic-rmw, block-ptr, foreshadow-f1-resolved

## What happened

Part 3 第 4 章·deep+skip_impl（纯 C++ MLIR pass 章，承 ch11 的 PtrAnalysis 逆向工程，deps=ch11）：《落到 memref：BlockPtrAnalysis、reinterpret_cast 与 load/store→linalg》。深挖 TritonToLinalg 侧的**物化下半**——ch11 讲的是 TritonToStructured 侧把指针算术逆向成 (offset,sizes,strides) 结构化描述（PtrState），ch12 讲的是 TritonToLinalg 侧的 BlockPtrAnalysis 如何把同一套镜像算法算出的三元组**真正铸成** `memref.reinterpret_cast`，再把 `tt.load`/`tt.store` 落成 `memref.copy`+`bufferization.to_tensor` / `bufferization.materialize_in_destination`，`tt.atomic_rmw` 落成硬件原子算子。10 个机制全覆盖：m1 BlockData/MemAccType 状态词汇、m2 parse 镜像 ch11（链接回指不重推导）、**m3 createCastOp（本章心脏，三元组物化 worked example）**、m4 parseReinterpretCast 逆映射、m5 rewriteAddPtr 落地驱动（零 stride 修复+known 表解耦）、m6 MemAccType 决策与 gather 回退、m7 load→copy+to_tensor、m8 store→materialize_in_destination、m9 atomicRMW 落硬件原子算子（**纠正任务 brief『→linalg.generic』之误**，lit `atomic_rmw.mlir` 的 `CHECK-NOT: GenericAtomicRMW` 坐实）、**m10 make_tensor_ptr→reinterpret_cast（回收伏笔 f1，含 original_order 转置维序）**。

7 张机制图（fig-m1-blockdata-vs-ptrstate、fig-m1-memacc-lattice、fig-m3-triple-to-recast、fig-m5-rewriteaddptr-flow、fig-m6-struct-vs-gather、fig-m7-load-domain-relay）+ 本章地图 chapter-map，独立盲审首轮 PASS（0 failure）、map 站 1 轮 PASS。write↔review 2 轮收敛，`lint_trace_consistency` 通过（正文数值推演表与 explainer 素材一致、零漂移），`lint_dossier`/`lint_explainer` 仅 3 条预期性 warn（skip_impl 章 manual trace 未经运行验证，宿主无 CANN，已声明走 pin+lit 夹具交叉验证）。verdict=**APPROVED**：1 条省略清单未穷尽（accumulatePotentialOffsetOnBase 注释+orderSize 未用变量未列入省略声明，non-blocking）+ 1 份逐机制勾选表（10/10 核对，7 个 core 机制三层/trace/invariant/量化全齐，2 个 supporting 桥接段按 dossier 设计豁免）+ 8 条 reader-comprehension/formulas 类 non-blocking 建议（m7 缺一句 invariant 收尾、m9 缺三层小标题、`boundary_check`/`propagateWasBoolToInt8Attr` 术语先用后释、`hivm::StoreOp` 与 IR 里 `hivm.hir.store` 的绑定未点破等），**0 条 blocking**。

## Why it matters

ch12 是分水岭算法心脏的收尾：ch11 建立的『指针算术能被逆向还原成三元组』只是分析结果，ch12 才回答『这份分析结果怎么变成能跑的 IR』——`createCastOp` 是全书『结构化描述 → 物化内存引用』这条主张的**唯一构造点**，往后 P4/P5 的所有优化都建立在这条 `memref.reinterpret_cast` 之上。`MemAccType` 的『结构化 vs gather 回退』决策也把 ch10 立的『不认识就整体失败/保守』原则第一次量化到指令数（O(1) vs O(N)），给读者一个可推导的『为何要尽量判成 StrucMemAcc』理由。

本章也纠正了一处容易顺着 brief 传播下去的错误——`atomicRMW→linalg.generic` 只是文档旧注释，现行代码落硬件原子算子；若不纠正，会在读者心智里种下一个假的『昇腾把原子操作也走 Linalg 通用路径』的印象，与后续 HFusion/HIVM 章节的硬件专用化叙事相悖。

## What to remember

- **本章心脏**：`createCastOp`（`BlockPtrAnalysis.cpp:L322-L343`）= `inferBlockOffset` 多维 offset 塌缩总和 + `getResultMemrefType` 组 `StridedLayoutAttr` + size==1 维动态 stride 抬 `max(_,1)` → 发射 `memref.reinterpret_cast`。逆映射 `parseReinterpretCast`（L896-L915）只落第一维 offset、其余补零，往返自洽。
- **决策载体口径（勿混）**：ch11 `shouldLinearize` 是 PtrState/TritonToStructured 侧字段；本章 `MemAccType`（Undefined<StrucMemAcc<UnstrucMemAcc，merge=max）是 BlockData/TritonToLinalg 侧的**对应但不同**决策载体——回收伏笔 f1 时必须显式区分，不可把两条平行管线的字段等同。
- **atomicRMW 纠偏（跨章硬口径）**：现行 `AtomicRMWConverter` 不产 `linalg.generic`，按硬件支持度落 `hivm::StoreOp`（IR 打印 `hivm.hir.store`）/`hfusion::AtomicXchgOp`/`hfusion::AtomicRMWOp`；真正用 `linalg::GenericOp` 的是 `AtomicCASConverter`（CAS，非 RMW）。lit `atomic_rmw.mlir` 的 `CHECK-NOT: GenericAtomicRMW` 是直接证据。
- **伏笔 f1 已回收**：m3 兑现『三元组铸成 memref.reinterpret_cast』，m10（`rewriteMakeTensorPtrOp` + `original_order` 维序置换）兑现『block_ptr 的转置与维序』。arc-map.json 中 f1 状态改为 `resolved`。
- **遗留的 non-blocking 打磨项**（未来若有小修窗口可顺手做，不影响本章交付）：§12.7 补一句 `memref.copy`+`to_tensor` 的不变量收尾；§12.9 拆成三层小标题（直觉/机制/源码）对齐全章体例；§12.10 rewriteMakeTensorPtrOp 代码块省略声明补齐两处（accumulatePotentialOffsetOnBase 由来注释 + 未用的 `orderSize`）；`boundary_check`/`propagateWasBoolToInt8Attr` 首现处补极简括注；`hivm::StoreOp` 与 IR `hivm.hir.store` 的对应关系点破一次。
- Bible 回写：glossary +8 条（MemAccType/MemAccVal、boundary_check、rewriteAddPtrToUnstrucMemAcc、LoadConverter、StoreConverter、AtomicRMWConverter 纠偏、rewriteMakeTensorPtrOp 系列）；concepts +8 条（对应 m1/m3/m4/m6/m7/m8/m9/m10 的深挖版）；figures +8 条（7 机制图 + chapter-map）；interfaces **不新增**（skip_impl 无精简版，无接口可注册）；arc-map f1 → resolved。
- 诚实边界：host 无 NPU/CANN，交叉验证走 pin 精确源码（`BlockPtrAnalysis.cpp`/`.h` @2badfc89e ~2235 行 + `LoadStoreConverter.cpp`）+ `unittest/Conversion/**/*.mlir` lit 夹具（`legal_stride.mlir`/`atomic_rmw.mlir`/`parse_select.mlir`），不伪造编译器 dump。下一站：ch13《MaskAnalysis：边界语义》或 Part 3 后续章节。
