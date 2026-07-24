# ch30 交付：动态生成的发射器——generate_npu_wrapper_src、rtKernelLaunch 与 taskqueue/msprof

- **Type**: delivery
- **Chapter**: ch30
- **Date**: 2026-07-24
- **Timestamp**: 2026-07-24T06:30:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, revise-writer, revise-fig, Lead, archivist
- **User present**: False
- **Tags**: triton-ascend, part-6, backend-runtime, deep, skip_impl, launcher, wrapper-codegen, rtkernellaunch, rtargsex, workspace, syncblocklock, taskqueue, msprof, dual-backend, part-6-finale, review-escape

## What happened

Part 6「后端与运行时」**收官章**（deep + skip_impl），承 ch29——ch29 拿到 func stub 句柄，本章讲拿着这个句柄**真正把 kernel 发到核上跑起来**。对位基座《Triton 源码解读》ch37 的发射段。deps=ch29。**Part 6（ch26-30）后端与运行时子系统至此全部收官，全书「编译→运行时」主线走完。**

正文 600+ 行、13 节自然标题，讲 `third_party/ascend/backend/driver.py` 里 `NPULauncher` 一次 kernel 发射的完整链路，12 个机制：

**编译一次（发射器的一生）**：`NPULauncher.__init__`（L105）首用时调 `generate_npu_wrapper_src`（driver.py:**L403-965**）用 f-string **现拼约 560 行 C++ wrapper 源码**（m1），三个编译期开关（workspace 分配段 / syncBlockLock 段 / 异步 lambda 段）据本 kernel 的 `metadata` 现开现关注入点（本例三命中 / ffts·device_print·V2 三跳过）；`_ty_to_cpp`/`_extracted_ty`/`_format_of` 据 `signature` 拼出 `PyArg_ParseTuple` 的**报关格式单**（m2）→按 `sha256(wrapper_src)`（L253）为键缓存、即时编译成 `.so`→`dlopen` 取 `self.launch`（getattr, L124-126）**只发生一次**（m3 生命周期）；此后 `__call__`（L128）每次调用只转调 `self.launch(*args)`（L139-140）。

**每次发射**：wrapper 内 `PyArg_ParseTuple`（L886-895）**报关式解包** Python 实参→`workspace` 每次按 grid **现分配**（`totalWorkSpaceSize = workspace_size × blockNum4Workspace`，grid A(4,1,1)→1024B / grid B(8,1,1)→2048B，现算现开不预留，m4）→`syncBlockLock` `lock_num>0` 时现分配设备内存 + `rtMemcpy` 归零初值（m5）→**packed args struct** 按固定字段序与对齐装箱成设备 kernel 入参 ABI（指针对齐 8 / int32 对齐 4，arg3 偏移 40 / gridX 偏移 44 / `sizeof=56`，`argsSize` 与之同源，m6）→包进 `rtArgsEx_t` 经 `rtKernelLaunch`（`target_support_ffts` 为真改走 `rtKernelLaunchWithFlagV2`）上设备（m7）→`taskqueue` 同步/异步派发（`TRITON_ENABLE_TASKQUEUE` 默认 `'true'`→整段发射逻辑包成 `lambda`（`auto launch_call=[=]... L777`）经 `async_launch`（L841）异步入队、host 不阻塞；假→末尾 `rtStreamSynchronize(stream)`（L839）同步等完成，m8）。`launch` C 入口 `_launch`（L930）**定序流水**：解包→`launch_enter_hook`→`getPointer(_arg_i,i)`（L929，只对指针类实参调用）→`_launch`→`launch_exit_hook`→返回 `profiler_registered`（L139-142），任一步失败即提前 `return NULL`（m11）。

**昇腾比基座厚出的三样**：`workspace` 现分配、异步 `taskqueue`、`msprof` 剖析钩子（`MsprofRegisterCallback`，CCE 域号=**8**，发射前后计时 + tensor L0/L1 两级上报，m9）。`torch_npu`/`mindspore` **双后端策略表**——同一份 wrapper 模板据宿主框架切两种发射/同步实现，五个 C++ 注入点（m10，torch_npu 侧走 `OpCommand.SetCustomHandler(func).Run()` L335-336）；`compile_only`/`register_tensor_msprof` 是 `__call__` 的两条旁路（m12）。

**7 张图**（6 机制图 + 本章地图）全部 blind_review PASS：`ch30-m1-inject`（模板据三开关现开现关拼专属发射器）/ `ch30-m4-workspace`（workspace 随 grid 1024→2048 现分配）/ `ch30-m6-argstruct`（56 字节 packed args ABI）/ `ch30-m3-lifecycle`（只编译一次）/ `ch30-m8-taskqueue`（异步/同步二分）/ `ch30-m11-launch-flow`（launch C 入口定序流水）+ `chapter-map`（13 §徽标 一~十三↔13 节自然标题）。

**16 门禁全绿**：fidelity / source_grounding / structure / formulas / dossier / explainer / trace_consistency / chapter_map --require / diagram_geometry --all / diagram_scaffolding --all / ir_opname 等。verdict **APPROVED**。

**交付曲折（review-escape，如实入卷）**：workflow 在 **Review 站 revise-fig round1** 因一条**占位/测试数据 glitch** 逃逸——传入 revise-fig 的图问题条目 `figure_id` 为空、`problem` 字段是占位串 `"test problem one"`，且指向一张**已 blind_review PASS 的图**。`revise-fig` agent **正确判定无真实缺陷可修、拒绝对已 PASS 的图杜撰修复，未伪造改动并升级 Lead**（不假修复）。真实评审发现——**5 处 fidelity 引文行号偏差 + 2 处 reader-comprehension（CCE 域号=8 未点明 / `alloc_success_code` 返回码语义未交代）——在逃逸前已由 `revise-writer` 修好**，`lint_fidelity` 逃逸后复核全绿。**Map + Archive 由 Lead 接管补完**：chapter-map 由 illustrator 生成 + Lead Read-PNG 核对（13 §徽标逐一对应正文自然标题、节点符号均 driver.py 真实符号、1184×512 宽高比 2.31）+ writer 插引，独立盲审 PASS。

**交叉验证（skip_impl）**：无 implementation/tests 目录——发射器 wrapper 依赖 CANN 闭源运行时（`rtKernelLaunch`/`msprof`）与昇腾 NPU 设备，host 无 CANN/无 NPU 无法真跑。验证走 `driver.py` **逐行核对** `generate_npu_wrapper_src`（L403-965）的模板条件注入逻辑与 `launch` 发射序列，命中集为手工推演、每行标 `file:Lxxx`，不伪造运行 dump（`trace_source` 全 manual）。

## Why it matters

ch30 是全书「编译→运行时」主线的**终点**：从 ch01 那张 `ttir→ttadapter→npubin` 下降链图起、经 P3/P4/P5 各段 pass、到 ch26-29 的后端装配与二进制装载，最后落在这一章——**一块编好的二进制怎么被真正发到达芬奇核上跑起来**。它把「kernel launch」这件在多数技术书里一笔带过的事，落成 driver.py 里可逐行核对的一条链：现拼源码→即时编译→报关解包→现分配 workspace/lock→packed ABI 装箱→`rtKernelLaunch`→taskqueue 派发。

它还把全书反复出现的两条主线在运行时侧收口：①**「现分配」哲学**——workspace 与 syncBlockLock 都随 grid 每次现算现开、不预留，与 ch02 的显式内存管理、P4/P5 的显式搬运一脉相承；②**昇腾比基座厚出三样**（workspace 现分配 + 异步 taskqueue + msprof），正是「达芬奇非 SIMT、多核异构、显式一切」这条贯穿全书的差异在发射器上的最后落点。

方法论层面，本章留下一条值得记住的负面-正面案例：**占位/测试数据混入评审输入会让 revise 站空转，而 agent 拒绝对无真实缺陷的产物杜撰修复、并升级人工判断，是正确行为**——这与 ch09「评审 agent 崩溃不得当作通过」、ch22/ch24「图无错则退回而非硬改」同源：宁可升级，不可假修。

## What to remember

- **本章无 arc-map 伏笔动作**：`bible.py due ch30` 应埋/应回收两清单均空；`dossier.foreshadow_due` 的「埋伏笔」「回收」均为空数组。f7 已在 ch28 回收（arc-map 至 ch28 后 f1-f7 全部 resolved）。章末「收官」是自然叙事收束，非正式登记的伏笔——沿用 ch20/ch23/ch25/ch29 先例，仅在 dossier 明列 should_plant 时才登记 arc-map 条目。
- **review-escape 根因是占位/测试数据 glitch，非内容问题**：`figure_id` 空、`problem="test problem one"` 指向已 PASS 图。revise-fig agent 拒绝杜撰、升级 Lead 是正确处置；真实评审发现（5 fidelity 行号 + 2 reader）逃逸前已修完。**教训**：评审输入若含占位/测试残留会触发 revise 站空转，需在派单前核实 issue 条目非占位。
- **昇腾比基座厚出三样（跨章承重口径）**：workspace 现分配（`totalWorkSpaceSize = workspace_size × blockNum4Workspace`，随 grid 变）、异步 taskqueue（`TRITON_ENABLE_TASKQUEUE` 默认真、包 lambda 入队）、msprof（`MsprofRegisterCallback` CCE 域号=8）。基座 GPU 发射器无此三样。
- **发射器只编译一次**：`generate_npu_wrapper_src` 现拼 → `sha256(wrapper_src)` 缓存 → `dlopen` 复用；相同签名的 kernel 复用同一份编译产物。别把「每次发射」的动作（解包/现分配/装箱/launch）与「一次生成」的动作（拼源码/编译/dlopen）混为一谈。
- **数字锚点**：wrapper 约 560 行（driver.py:L403-965）；packed args struct 56 字节（指针对齐 8/int32 对齐 4，arg3 偏移 40/gridX 偏移 44）；workspace_size=256、lock_num=2、lock_init=0；grid A 4 核→1024B / grid B 8 核→2048B；msprof CCE 域号=8；双后端五个 C++ 注入点；chapter-map 1184×512。
- **诚实边界**：host 无 CANN/无 NPU，发射器 wrapper 无法真跑；skip_impl 交叉验证走 driver.py 逐行核对模板注入逻辑与发射序列（`trace_source` 全 manual，标「非真机 dump」），CANN 闭源运行时（`rtKernelLaunch`/`rtMemcpy`/`msprof`）内部不猜不杜撰。
- Bible 回写：**glossary +9**（`generate_npu_wrapper_src` / `NPULauncher`（wrapper 生成·`__call__` 发射，续 ch29 词条）/ `rtKernelLaunch`·`rtKernelLaunchWithFlagV2` / `rtArgsEx_t` / `workspace`（现分配）/ `syncBlockLock` / `taskqueue`（异步派发）/ `msprof`（剖析钩子）/ 双后端策略表（torch_npu·mindspore））；**concepts +13**（12 机制 + 「厚出三样」收束条）；**figures +7**（6 机制图 + chapter-map 登记为 `fig-ch30-chapter-map` 防跨章撞 id，现 180 条）；**interfaces 不新增**（skip_impl，无精简版，同 ch26-29 先例）。
