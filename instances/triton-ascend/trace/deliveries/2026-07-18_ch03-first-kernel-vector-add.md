# ch03 交付：上手第一课——vector-add 的 GPU→NPU 最小改写

- **Type**: delivery
- **Chapter**: ch03
- **Date**: 2026-07-18
- **Timestamp**: 2026-07-18
- **Agents involved**: analyst, explainer, illustrator(ch03ill), ch03writer, reviewer, Lead, archivist
- **User present**: false
- **Tags**: triton-ascend, part-1, skip_impl, on-ramp, minimal-rewrite, torch-npu-registration, block-mask, logical-vs-physical-grid, npu-mask-boolean, test-as-truth, extension-hooks-preview, linter-bug-fix

## What happened

Part 1 **收尾** reader on-ramp 章，kind=**skip_impl/meta**（无只做减法的精简版——拿活体标本自带对拍测试作真相源）。verdict=**APPROVED**，全 linter green，3 图全部盲审 PASS。主题：把硬件 primer(ch02) 的 tiling/显式搬运落到一个**能跑通的最小核**上——拿 `tutorials/01 vector-add` 当活体标本，逐行看它相对基座 triton 同名核改了什么，为后续拆解建立坐标系。

**7 机制主线**：
1. **minimal-rewrite（两行最小改写）**：移植相关改动**只有两处**——顶部 `+import torch_npu`、张量 `device='cuda'→'npu'`；`@triton.jit add_kernel` 核体**逐字节不变**。（教程另删注释/`is_cuda` 断言/benchmark 属教程取舍、**非移植**改动，正文/图面区分。）兑现 ch01 埋的『同核异后端（核体 0 改动）』线索。
2. **torch-npu-registration**：`import torch_npu` 在 import 期把昇腾设备后端挂进 PyTorch 设备注册表，'npu' 才成合法 device——『同前端异后端』在 host 侧最外层落点。
3. **block-mask-arithmetic（算法机制）**：`grid=cdiv(n,BLOCK_SIZE)`、`offsets=pid*BLOCK_SIZE+tl.arange`、`mask=offsets<n_elements` 的满块/尾块算术，尾块 mask 拦越界 lane。核对脚本 `run_grid_mask.py` 复现满块/尾块边界。GPU/NPU 此段逐字一致（通用 Triton 范式、非昇腾特有）。
4. **logical-vs-physical-grid**：vector-add 用逻辑 grid=(cdiv(n,BLOCK_SIZE),) 先**跑通**；ch02 的 grid=(NUM_CORE,) 物理核绑定是**跑快**的推荐写法（AutoBlockify 自动收敛）。二者不矛盾——呼应 ch02 三级 tiling。
5. **npu-mask-boolean（i1/i8 语义）**：mask 语言层是 i1、落达芬奇按 i8 存储/搬运；ch03 点到、细节留后续硬件 IR 层。
6. **test-as-truth（测试即真相源）**：skip_impl 不写精简版，拿 tutorials/01 自带 torch 对拍测试作验证，host 无 NPU 时标『需真机』。
7. **extension-hooks-preview**：`tl.extra.cann.extension.compile_hint` 进阶钩子只作 preview 点名，展开留 P2 ch08。

**linter bug 修正经过（值得记）**：`lint_fidelity`/`lint_source_grounding` 的路径正则 `[\w/]+` **不含连字符**——昇腾教程文件名带连字符触发假 `narrative_grounding` **BLOCKING**（把合法源码路径当脚手架泄漏/未 grounding）。Lead 修正则允许连字符 + TDD 补例（exp-2026-07-19-01，commit **f7a5dec2**）→ 复跑清除假阳。

**Lead 派 ch03writer 补 4 处**（non-blocking reader-comp）：L21 移植两处**精度**表述（『移植相关改动只有两处』严守，教程删项另述）、kernels 一句 gloss、回指一致、截断处省略号。**Lead 派 ch03ill 修 fig-m1** 行号徽标（对齐真实改动行）。

回环轮数：write↔review 1 轮、blind 1 轮、map 1 轮（skip_impl 无 impl↔test）。

## Why it matters

Part 1 的**落地锚**：前两章（ch01 鸟瞰 + ch02 硬件 primer）都在建心智模型，ch03 把它砸到一份**能跑的最小核**上——读者第一次看到『改两行就切后端』的活体证据，也是后续所有拆解的坐标系原点。『逻辑 grid 跑通 vs 物理核跑快』这条张力从这里立起，一路牵到 P4 AutoBlockify。

## What to remember

- **诚实边界**：host **无 NPU/CANN**，vector-add 的**静态改写差异**(import/device/核体不变)照读、已核对；对拍测试与 i1/i8 运行时语义标『需真机』。
- **本章埋下**（→P2 ch08 scope/extension 章）：`compile_hint` 语言层**扩展钩子**——ch03 只 preview 点名，系统展开留 ch08。**前向线索、非 arc-map 正式伏笔**（`bible.py due ch03` 空）。
- **事实校准点**（勿再回退）：移植相关改动**严格只有两处**（import torch_npu + device='npu'），核体逐字节不变；教程删注释/is_cuda 断言/benchmark 是**教程取舍非移植**，勿混为一谈。block-mask 满块/尾块段 GPU/NPU **逐字一致**。
- **linter 已加固**（勿再复发）：fidelity/source_grounding 路径正则已允许连字符（f7a5dec2）——带连字符的昇腾源码/教程路径不再触发假 narrative_grounding BLOCKING。
- **章号 vs Part**：ch03 落 **part-1**，物理章号 ch03 < 已归档的 ch04（part-2），INDEX/state 按交付时间排序即可。
