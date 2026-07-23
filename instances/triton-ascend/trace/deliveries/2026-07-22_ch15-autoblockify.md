# ch15 交付：AutoBlockify——把多个网格实例折成一条 blockify 循环

- **Type**: delivery
- **Chapter**: ch15
- **Date**: 2026-07-22
- **Timestamp**: 2026-07-22T23:30:00Z
- **Agents involved**: analyst, writer, illustrator, reviewer, archivist
- **User present**: False
- **Tags**: triton-ascend, part-4, deep, skip_impl, autoblockify, unrealizedconversioncast, ascend-opt, blockify-loop

## What happened

Part 4「异构双核」开篇，ascend-opt 子系统第一站，deep+skip_impl（纯 C++ MLIR pass 章，无 .py、宿主无 CANN 编不动，deps=ch10）：《AutoBlockify：把多个网格实例折成一条 blockify 循环》。承 ch10 一句带过的「`add_auto_blockify` 是 `ttir_to_linalg` 管线第 1 趟 pass」，本章系统展开。边界立得很清楚：不碰指针分析（ch11-14 triton-to-linalg 子系统的活），改**重塑网格粒度**；也明确不是 UB 容量 tiling（下游闭源 bishengir 的活，本章边界外）。

9 个机制全覆盖：**m1 autoBlockifySize 折叠粒度 + no-op 门**（`size==1` 默认直接返回，未开 `TRITON_ALL_BLOCKS_PARALLEL` 时编译期强制压回 1，本 pass 自己不算最优 size）；**m2 网格拍平 + blockifiedId 造载体**（preProcess 第一步——三维 `(idX,idY,idZ)` 按混合进制拍平成线性 `logicalBlockId`，`splat+range` 连号折出 `blockifiedId`，与 `ori` 合成的恒真 mask 一起打包进双输入 `UnrealizedConversionCastOp`；不变量：拍平/反解互为双射）是全章地基；**m3 UnrealizedConversionCast 作类型防火墙 + PropagateUnrealizedCastDown 逐 op 下推**（结果类型不变、载体沿 def-use 单调下推、处理即消解；白名单式穷举当前支持 op，非覆盖全部 Triton op）；**m4 checkBlockifiable 守门**（拒绝名单——denylist，非穷举——4 类硬拒绝 op + tensor-ptr；`scf.if` 打标签留给循环；递归靠去重集合保证终止，终止不等于可 blockify）；**m5 前导维批处理化**（`getExpandedType`/`rewrite*` 家族，size 恒拼在 shape 位置 0，原布局/stride 不动，对上 `Passes.td` 摘要句 `Expand highest dimension`）；**m6 blockify 循环**（不可批处理 region op 靠 `createBlockifyLoop` 折成 `scf.for`，上界 `min(max(blockNum-blockId,0),size)` 恰为本物理块落在合法区间内的逻辑实例数，三档 blockId 手算核对）；m7 尾块+mask 合成、m8 终态 cast 落地、m9 收益量化（调度块数单调不增 `ceil(G/size)≤G`，示例 6→2，size=1 退化回 m1 no-op 门）按 supporting 定位覆盖。

5 张机制图（fig-m2-flatten-carrier、fig-m3-cast-propagation、fig-m5-leading-dim、fig-m6-blockify-loop、fig-m9-before-after）+ 本章地图 chapter-map 共 6 图，独立盲审 1 轮 PASS（0 failure；fig-m3 曾在自查阶段发现杜撰符号 `rewriteArithAddI`，已改成真实 user 类型+函数名后过），map 站 1 轮 PASS。write↔review 3 轮收敛，`lint_trace_consistency` 全绿零漂移。

## Why it matters

ch15 是全书从「指针语义还原」（P3 结构化下降链）转向「昇腾侧编译器优化」（P4）的分水岭：昇腾达芬奇没有 GPU 的 warp，AutoBlockify 用编译器把多个逻辑 program 实例批处理进一个物理块的向量运算，达到与基座 GPU 靠 warp 合并访存同样的目的——是「同一优化目标、两种硬件模型下的两种实现」这条全书线索的又一个具体例证。机制上也很有代表性：`UnrealizedConversionCastOp` 双输入充当「类型防火墙」做渐进改写，是 MLIR 里一种值得记住的通用手法（造一个类型上说不通但局部合法的占位 cast，沿 def-use 逐段替换、单调收敛到无 cast 状态）。

## What to remember

- **本章心脏**：`UnrealizedConversionCastOp` 双输入载体（`(value, mask)`，结果类型仍是原标量类型）+ `PropagateUnrealizedCastDown::matchAndRewrite` 的逐 op 下推——两条不变量（载体恒 2 输入 / 处理即消解）保证下推单调收敛，最终 IR 里不再有双输入 cast。
- **两处 mask 语义分野，务必分清**：载体构造那处合成 mask 用的是 `arith.ori`（因为 `blockifiedId=logicalBlockId+range(0,size)` 恒 `>=0`，`ori` 出来的 mask 恒全 True，对合法折叠不做屏蔽）；尾块合并那处用的是 `arith.and`（与算子原有 mask 相与，真正起屏蔽作用）。全章反复强调这是两处不同机制、别混。
- **两处"名单"方向相反，也别混**：`PropagateUnrealizedCastDown::matchAndRewrite` 那张是「命中即支持」的白名单式分派表（未命中 `llvm_unreachable`）；`checkBlockifiable` 那张是「命中即拒」的拒绝名单/denylist（4 类硬拒绝 op + tensor-ptr）。两者都不等于「穷举」——评审对此专门核实无过度概括。
- **评审结论**：4 维评审 APPROVED，0 blocking / 6 non-blocking，全部集中在版式一致性与图注精确度：(1) m5 的不变量论证内容已在但未挂 `**不变量**` 标签，与 m2/m4/m6 版式不一致；(2) fig-m3 图注承诺「塞进 blockify 循环」分支但图上实际只画了两类真实分支（rewrite 批处理化 / cast↔cast 消解）；(3) 尾块/mask 一节里"本例的 uccMask 恒全 True"的表述容易被误读成本例特有，其实第2节已证明这是任何配置下都成立的结构性结果，且未点破尾部物理块内越界 lane 到底靠什么拦住；(4) `blockNum`/`blockId` 简写在 blockify 循环一节与尾块/收益两节字面上指代两个不同数值的量（`logicalBlockNum`/`logicalBlockId` vs 运行期 driver 截断的物理调度块数），未显式区分；(5) `region op` 一词在网格拍平一节首现无括注，且抢跑在 `checkBlockifiable` 一节（讲清 if/for 会被打标）之前；(6) cast 作载体一节第三次提到 blockify 循环（atomicCAS 现造一条）缺少像前两次那样的回指锚点链接。均为读者体验打磨、不涉及事实错误或缺主线机制，Lead 决定按 non-blocking 结案，留给下次存量回修批次一并处理。
- **Bible 回写**：glossary 新增 6 条（`logicalBlockId`/`blockifiedId`、`UnrealizedConversionCastOp` 作类型防火墙、`checkBlockifiable`、`PropagateUnrealizedCastDown`、blockify 循环、region op）+ 更新 1 条既有词条（`AutoBlockify / auto_blockify_size`，补记 ch15 系统展开）；concepts 新增 8 条（对应 m1-m9 机制摘要）；figures 新增 6 条（5 机制图 + chapter-map）；interfaces **不新增**（skip_impl 无精简版）；arc-map **无变动**（`bible.py due ch15` 两清单皆空，本章无正式伏笔埋/回收，末节对下一章"核亲和分析"的指向属顺序性前瞻，非需登记回收的具体承诺）。
- 诚实边界：host 无 NPU/CANN，交叉验证走 pin 精确源码（`AutoBlockify.cpp`/`RewriteOperation.cpp`/`Utils.cpp` @2badfc89e ~1080 行）+ `unittest/Conversion/General/AutoBlockify/auto_blockify.mlir` lit 夹具（`kernel`/`kernel2` 两用例），不伪造编译器 dump；收益量化明确标注为「按定义手算的量级对照，非真机 benchmark」。对位基座 ch25（GPU warp 内合并访存 vs 昇腾网格实例合并）。下一站：ch16《Cube 还是 Vector：给每个 op 判核亲和》，从「执行粒度」问题转向「执行载体」问题。
