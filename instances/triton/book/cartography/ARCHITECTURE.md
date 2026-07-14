# Triton 架构地图（源码解读书大纲综述）

> 源码基线：Triton v3.2.0 @ `9641643da6c52000c807b5eeed05edaec4402a67`，只读镜像在 `instances/triton/source/`。
> 本文件是 8 份子系统 digest 的综合，配套 `outline-final.json`（40 章 / 9 部分 / 6 篇原理 primer）。

## 一句话心智模型

**Triton 是一门 GPU DSL：你在 `@triton.jit` 里写的每一行 Python 不是在跑，而是在被追踪成 Triton IR；随后这份 IR 沿一条五级降级阶梯（`ttir → ttgir → llir → ptx → cubin`）逐层下降，每一层都是一次带理由的降级——而『带布局的张量到 per-thread 寄存器与访存指令』这条主线上，核心数据结构始终是 layout（把张量索引映到线程集合的函数）。**

全书就是沿这条阶梯逐层解读真实源码，并在每一站点名两件事：(1) 这一层的**双语栈接缝**（Python 校验/合法化 ↔ C++/MLIR 执行）；(2) 一块**新卡从哪里挂进来**（姊妹篇 triton-ascend 的配对锚点）。

---

## 子系统地形图（8 子系统 → 降级阶梯上的位置）

```
                      读者旅程（reader journey）= 降级阶梯从上到下
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  [Part 2] dsl-language      tl.*：追踪期把 Python 表达式翻成 tt.* IR        │  ← 前端语言层
   │        │  @builtin 注入 _builder → builder.create_*（双语栈第一现场）        │
   │        ▼                                                                     │
   │  [Part 3] jit-runtime       fn[grid](...) → JITFunction.run → 缓存 → 编译    │  ← 宿主运行时
   │        │  driver.active（后端发现/选举）、autotune、TRITON_INTERPRET 替身     │
   │        ▼                                                                     │
   │  [Part 4] compiler-driver   compile() 主循环；CodeGenerator：AST → TTIR      │  ← 编译前端
   │        │  constexpr↔tensor 两个世界；控制流下降到 scf（皇冠明珠）            │
   │        ▼                                                                     │
   │  [Part 5] ir-dialects       三层方言 tt / ttg / ttng + 布局(layout) 抽象     │  ← IR 与布局
   │        │  布局=函数 L → distributed/shared 编码 → LinearLayout 统一          │
   │        ▼                                                                     │
   │  [Part 6] analysis-transforms  静态分析(AxisInfo/Alloc/Membar) 驱动一串 pass │  ← 优化 pass
   │        │  Coalesce/AccelerateMatmul/RemoveLayout/软件流水线/Prefetch/WS      │
   │        ▼                                                                     │
   │  [Part 7] conversion-lowering  TTIR→TTGIR 贴布局；TTGIR→LLVM per-op 降级      │  ← 降级
   │        │  convert_layout 三路径 / 访存向量化 / dot→mma / 内联 PTX → cubin    │
   │        ▼                                                                     │
   │  [Part 8] backends-hw       约定即插件：CUDABackend / AMD HIP 落地           │  ← 硬件后端
   │           add_stages 注入 + load_dialects 挂 dialect + driver/launcher      │
   └──────────────────────────────────────────────────────────────────────────┘
        [Part 1] 起步（心智模型 + GPU 执行模型 primer + kernel 鸟瞰）—— 阶梯之前的地图
        [Part 9] tooling-ecosystem（proton/AOT/disasm/triton-opt/tutorials）—— 与阶梯正交的观察层
```

**贯穿全书的两条暗线：**

1. **双语栈接缝**：Python 层几乎只做『校验 + 合法化 + 把参数交给 `builder.create_*`』，真正的 op 定义 / verifier / pass 算法在 C++/MLIR（`_C.libtriton`）。从 Part 2 的 `@builtin` 到 Part 7 的 inline-PTX，每一层都有一个 hand-off；写作时必须在接缝处以『同一段 kernel 源码 ↔ dump 出的 IR 文本』搭桥，否则整条链看起来是魔法。
2. **无 GPU 可运行边界**：能脱机跑的绿洲——DSL 的纯 Python 逻辑（类型提升 `computation_type_impl`、constexpr、`standard/random`）、`TRITON_INTERPRET` numpy 替身、`make_ir`（AST→TTIR，只需 `libtriton.so`）、布局 `getElemsPerThread`/`toLinearLayout` 的 CPU 侧查询、`triton-opt`+FileCheck 的 IR 变换、`viewer` 喂 `example_*.json`、`link.py` 纯字符串生成。断裂处——`compile()` 的 ttgir 之后各级、`kernel.run` 发射、`ptxas`/`cuobjdump`、`do_bench`。素材生成据此分流（能跑的实测，不能跑的用真实抓取的 IR diff / PTX 片段佐证，绝不伪造运行结果）。

---

## 逐 Part 大纲叙述

### Part 1 · 起步：一门 DSL 与它的目标机器（ch01–ch03）
建立全书取景框。**ch01** 立心智模型（追踪成 IR、五级阶梯、双语栈、subtract-only 精简版约定、姊妹篇对位约定）。**ch02（primer）** 补齐 GPU 执行模型这一自包含地基——SIMT 的 grid/block/warp/lane、内存层级、访存合并、占用率——后续布局、访存、共享内存、后端占用率各章都回扣它。**ch03** 用最小 vector-add 低分辨率走通 `@jit → run → TTIR → TTGIR → LLVM → PTX → launch` 整条主线，是每一章放大进去的那张地图（flagship）。

### Part 2 · 领域语言 tl.*（ch04–ch09）
读者进入 Triton 的表面层，也是双语栈第一现场。**ch04** tl.* 两层结构与 `@builtin` 的 `_builder` 契约 + constexpr 编译期/运行期分野。**ch05** 三层类型系统(dtype/pointer/block) + cast + tensor 运算符重载。**ch06** 类型提升与隐式广播——最纯 Python、最反直觉、几乎可脱机单测的深水区。**ch07** 造块/形状变换/访存(load/store/mask/block-ptr/TMA)/原子——kernel 最高频、双语断点最密。**ch08** dot 块级矩阵乘（`min_dot_size` 后端钩子）+ 归约/扫描（combine_fn 经 `_generator` 回调变 IR region）。**ch09** 用 DSL 自举的 `standard`/`random`/`math`/extern libdevice + 编译期诊断提示——`extra/__init__.py` 的 pkgutil 后端发现是 ascend 挂载点。

### Part 3 · 宿主运行时（ch10–ch13）
一次 launch 的 host 侧全链。**ch10** JITFunction 元数据 + 缓存键（特化 `compute_spec_key` + `DependenciesFinder` 源码依赖哈希 + 全局变量守卫）。**ch11** `run()` 脊柱胶水：缓存查询→编译→发射，`driver.active` 三件套是跨到 driver 子系统的边界。**ch12** driver 抽象与后端发现（后端接入第一道脊柱）+ autotune + 磁盘缓存。**ch13** `TRITON_INTERPRET` numpy 替身执行——让全书的核在无 GPU 可跑可调。

### Part 4 · 编译前端（ch14–ch17）
`@jit` 函数如何变成 TTIR。**ch14** `compile()` 驱动主循环（内容寻址缓存 `triton_key`、`make_backend`、`add_stages`）+ ASTSource/IRSource + BaseBackend 契约 + CompiledKernel——后端六个钩子的定义处。**ch15（primer）** SSA 与结构化控制流：φ 节点 / 块参数 / loop-carried，为控制流下降铺台阶。**ch16** CodeGenerator 骨架 + constexpr↔tensor 两个世界 + 表达式/赋值/函数下降。**ch17** 控制流下降到 scf（if/for/while → 结构化 IR + φ）——全书前端皇冠明珠（flagship）。

### Part 5 · IR 与布局（ch18–ch23）
全书思想核心。**ch18** tt.* 硬件无关词汇表 + 方言黏合层（trait/验证/模块契约）。**ch19（primer）** 布局即函数——TritonGPU 张量的 encoding 就是把索引映到线程集合的函数 L，全书的钥匙。**ch20** distributed 布局（Blocked/Slice/Mma/DotOperand）；AMD 与 Nvidia MMA 编码并排共存是后端新增布局族的样板。**ch21** shared 编码与 swizzle（第二套心智模型）。**ch22（primer）** LinearLayout——用一族 GF(2) 线性函数统一所有布局，思想高潮（flagship）。**ch23** ttg.*/ttng.* 算子（convert_layout / 异步拷贝 / Hopper 硬件方言样板）。

### Part 6 · 优化 pass（ch24–ch30）
朴素 IR 如何变高性能，主线=静态分析量化驱动改写。**ch24** AxisInfo（连续性/整除性稀疏数据流，唯一适合 Python 玩具模型的单元）+ Coalesce（分析→改写最短闭环）。**ch25** 共享内存 Allocation/Alias/Membar。**ch26（primer）** Tensor Core 与 MMA 布局：tile 形状为什么长这样（重绘 PTX mma 文档图）。**ch27** AccelerateMatmul（tt.dot→MMA，配对脊柱重灾区）+ RemoveLayoutConversions 四阶段 + OptimizeDotOperands。**ch28（primer）** 软件流水线与模调度：num_stages 调度了什么。**ch29** 流水线落地：MatmulLoopPipeline 建模 + PipelineExpander 展开。**ch30** Prefetch + Warp Specialization（Hopper 进阶/选读）+ 杂项清理 pass。

### Part 7 · 降级（ch31–ch34）
五级阶梯后半程。**ch31** 五级台阶全景 + 第一跳 TTIR→TTGIR 贴布局。**ch32** 第二跳地基：类型塌缩 + ConvertLayoutOp 三条搬运路径（寄存器重排 / warp shuffle / 共享内存往返）。**ch33** 共享内存降级（`global_smem` 寻址 + swizzle）+ 全局访存向量化（AxisInfo 的收益兑现处）。**ch34** dot→mma/wgmma 指令选择 + 逐元素/fp8 + reduce/scan shuffle + 内联 PTX 构造器 + LLVM→PTX→cubin 出口，闭合阶梯。

### Part 8 · 硬件后端（ch35–ch37）
把所有后端接缝收成施工图。**ch35** CUDABackend 五段 stages 注入 + load_dialects 挂 dialect（把 Part 6 的 pass 串成真实序列，flagship）。**ch36** PTX→cubin（ptxas）→装载（driver.c）→launcher 代码生成→占用率。**ch37** AMD HIP 后端作为『同一抽象的第二种实现』逐面对照——这张对照表就是 triton-ascend 昇腾后端的施工图（flagship）。

### Part 9 · 工具生态（ch38–ch40）
与阶梯正交的观察/部署/学习层。**ch38** proton launch 钩子 + roofline viewer + do_bench 秒表。**ch39** AOT compile/link 部署 + disasm 读 SASS。**ch40** triton-opt 家族 + triton-tensor-layout 布局探针 + tutorials 01→09 阶梯。

---

## 与姊妹篇 triton-ascend 的配对脊柱（后端接缝章）

**triton-ascend = 本仓 fork + 昇腾 NPU 后端。** 它是本书的衍生姊妹篇：本书是基座（`pairs_with` 全部留空），对位方向由姊妹篇 outline 反向引用本书章号。姊妹篇的核心叙事——『一块新卡从哪里挂进来』——正是逐章对位本书标注了 `backend_seam: true` 的 17 章。四道接缝按 digest 归纳如下：

| 接缝 | 本书承载章 | ascend 在此做什么 |
|------|-----------|------------------|
| **① 后端发现 + 抽象契约** | ch12（driver 发现/选举）、ch14（BaseBackend 六钩子）、ch18（tt.* 中立边界） | 在 `backends/ascend/` 放一对 `compiler.py`+`driver.py` 各一个 concrete 子类，`is_active`/`supports_target` 决定何时被选中；零改动纳入发现机制 |
| **② stages 注入 + 第三方 dialect 挂载** | ch14、ch23（TritonNvidiaGPU 硬件方言样板）、ch29/ch30（pipeline/WS pass 序列）、ch31/ch35（add_stages）、ch40（RegisterTritonDialects） | 提供自己的 `add_stages`（最后一段产出昇腾二进制）+ `load_dialects` 挂 Ascend NPU dialect（仿 TritonNvidiaGPU 的定义/注册结构）；在 `RegisterTritonDialects.h` 等价物注册以能被 triton-opt 调试 |
| **③ 布局编码 + MMA 加速 pass** | ch20（AMD/Nvidia 编码并存样板）、ch26/ch27（AccelerateMatmul 硬绑 NVIDIA）、ch30（Prefetch 已接受第三方 MMA 编码）、ch32/ch34（TargetInfo + per-op lowering + NVGPU dialect） | 新增 `AscendXxxEncodingAttr`（同继承 DistributedEncoding）、实现其 `toLinearLayout` 即可复用 convert_layout/LinearLayout 中立桥；fork 面向 cube 单元的等价 matmul 加速 pass 与自己的 dot-operand 编码 |
| **④ AttrsDescriptor 特化 + driver/launcher/tracer 落地** | ch10/ch14（AttrsDescriptor）、ch37（HIP 的 pointer_range 扩展示范）、ch36（driver.c/launcher/occupancy）、ch38（proton `_select_backend` cuda→cupti/hip→roctracer） | 覆写 `_add_backend_properties` 挂昇腾专属特化属性；提供 `is_active`/`get_current_target`/`launcher_cls` 与 NPU 二进制装载；proton 需自有 tracer 分支 |

**对位样板章**：**ch37（AMD HIP 对照）** 是姊妹篇最直接的模板——AMD 已证明『6 方法骨架不变、Options/pass 序列/二进制格式/装载器血肉各异』，昇腾后端即照抄这套骨架、替换血肉。本书凡触及 `backend_seam` 章，写作时必须显式点名接缝、不可淡化——它们是与姊妹篇逐章对齐的锚点。

---

## 章数与粒度说明

40 章（含 6 篇 primer：GPU 执行模型 / SSA / 布局即函数 / LinearLayout / Tensor Core-MMA / 软件流水线）覆盖 8 份 digest 共 128 个测绘单元——按『讲透为准』把强耦合单元合章（如 AxisInfo+Coalesce、Allocation+Alias+Membar、pipeline 建模+展开、per-op lowering 收尾+出口），把独立难点（类型提升、控制流下降、swizzle、LinearLayout）单列。规模对齐 vLLM 书（37 章）的粒度感，落在 30–40 章参考区间上沿——因 Triton 是编译器全栈（DSL→前端→IR→优化→降级→后端），单元密度显著高于纯推理引擎，未硬凑亦未强压。`mode` 分布：code(deep) 16 / meta(skip_impl+primer) 24——MLIR C++ 层占比高，多数降级/布局/pass 章走 skip_impl（真源码内嵌 + IR diff 佐证，不产精简版），DSL 与纯 Python 运行时章走 deep（带 subtract-only 精简版）。

## 覆盖交叉核对纪要(2026-07-15,RUNBOOK §0.6 第 3 步)

核对员抓出 5 缺口,处置:
1. **python/src/ pybind11 绑定层** → 新开 ch18「双语桥:libtriton 的 pybind11 绑定层」(part-4 收口),后续章 +1 重编号——双语栈接缝的 C++ 半边不能只讲 Python 侧。
2. **make_ttir 的 TTIR 级 pass**(RewriteTensorPointer/Combine 等)→ 并入 ch32 降级台阶章(回扣 block pointer 的下场)。
3. **proton csrc C++ 采集引擎** → ch39 显式不深入(选读框给两条采集路径地图);理由:采集协议与编译主线正交。
4. **test/ lit 基建 + unittest/ gtest** → ch41 显式不入书(附「用 lit 复现 pass 测试」选读);理由:开发基建非编译产物主线。
5. **CMake 构建拓扑** → ch01 补「源码树与构建总览」一图(libtriton 聚合/TRITON_BACKENDS_TUPLE 注入);零散小工具(build_extern/experimental_descriptor/runtime/build.py/LLVMDIScope)相邻章一句话点名或不入书。
