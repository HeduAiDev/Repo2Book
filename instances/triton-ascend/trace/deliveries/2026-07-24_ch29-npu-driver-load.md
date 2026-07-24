# ch29 交付：NPU 运行时驱动与二进制装载——把编出的 blob 装上达芬奇

- **Type**: delivery
- **Chapter**: ch29
- **Date**: 2026-07-24
- **Timestamp**: 2026-07-24T03:30:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, Lead, archivist
- **User present**: False
- **Tags**: triton-ascend, part-6, backend-runtime, deep, npu-driver, jit-build, cann-runtime, registerkernel, magic-number, driver-contract

## What happened

Part 6「后端与运行时」第四站，承 ch28（编译期收官后转向运行时）；deps=ch28（本章讲 ch28 闭源边界产出的二进制 blob 怎么被装上设备）。对位基座《Triton 源码解读》ch37（`ptxas`→cubin→发射的装载段：`cuModuleLoad`/`cuModuleGetFunction`）。

本章 391 行，讲 triton-ascend 装载层的两层结构：**Python 驱动层** `third_party/ascend/backend/driver.py`（`NPUUtils` 单例首次实例化即以 `npu_utils.cpp` 源码 md5 为 key 就地 JIT 编出 `npu_utils.so`；`NPUDriver` 实现 triton `DriverBase` 契约）+ **C++ 扩展** `third_party/ascend/backend/npu_utils.cpp`（`registerKernel` 真正调 CANN `rt*` 运行时 API：`rtSetDevice`→`rtDevBinaryRegister`(拿 `devbinHandle`)→`rtFunctionRegister`(拿 func stub)）。8 个机制全覆盖：npu_utils.so 首次即时编译加载(m1)、跨语言装载调用链(m2)、registerKernel 的 rt* 调用序列(m3)、aiv/非 aiv magic 二分而非 aiv/aic/mix 三分(m4)、stubName/registered_names 同名去重(m5)、loadKernelBinary 参数解包与四元组返回(m6)、NPUDriver 契约实现(m7)、硬件规格探测与 AIV=AIC×2 双核比例(m8)。3 张机制图(fig-m1-jit-cache/fig-m2-crosslang-chain/fig-m3-rt-sequence) + 本章地图，独立盲审 1 轮 0 failure；write↔review 1 轮收敛；map 1 轮 PASS。

verdict **APPROVED**，0 blocking + 14 non-blocking（均 negotiable，留存量回修批次）：2 条代码块行号标注偏差(driver.py L47-L73 应为 L47-L75；npu_utils.cpp L322-L332 应为 L322-L334)；1 条跨章事实精度(全章 4 处 `cuModuleLoad` 与基座 ch37 实际展示的 `cuModuleLoadData` 不一致)；1 条 dossier must_keep 覆盖缺口(`get_aivector_core_num` 符号名正文 0 次出现)；2 条排版一致性(m2/m4 缺**源码**小标题分段、m6-m8 缺**直觉**小标题)；1 条 explainer 复杂度结论未搬进正文(unordered_map 均摊 O(1))；3 条图面行号偏差(fig-m3 两处 trunk box L67-68/L69-70 应为 L68-70/L71-72，fig-m2 单行 L92 应为 L92-93)；1 条叙事节奏(fig-m3 结论框提前揭晓 §29.5 的反直觉点)；1 条锦上添花建议(m4 可选配图)；4 条 reader-comprehension(`NPUDriver.self.utils` 与 `NPUUtils` 单例的绑定关系未显式点出/`rtFunctionRegister` 两个"名字"参数与尾随 0 未逐一解释/`get_backend_func` 黑盒未定性/§29.3 双重比喻累赘)。

无 implementation/tests 目录（该章无精简版产物，交叉验证走 pin 源码逐行核对，不伪造运行 dump）。

## Why it matters

ch29 是编译期→运行时叙事转折的第一站：ch28 讲透了「怎么编出二进制」，本章讲「这块二进制怎么被设备认得」。它把「NPU 二进制装载」这件在其他技术书里常被一笔带过的环节，落实成三步可核对的 `rt*` 调用序列，并且诚实标出了闭源边界——`rt*` 之外的 CANN 内部不猜不杜撰。

它还把 ch02 建立的达芬奇 cube:vector=1:2 硬件事实在运行时侧兑现了一次：`rtGetAiCoreCount` 只探 cube 核数，vector 核数直接 `×2` 算出，不再探测——这条线索从硬件规格（ch02）→编译期双核分工（P4/P5）→运行时装载探测（本章）贯穿全书。

下一章（发射器）承接本章拿到的 func stub 句柄，讲 `rtKernelLaunch` 真正把 kernel 发到核上跑起来——装载与发射的分界在本章末尾已经点清。

## What to remember

- **本章无 arc-map 伏笔动作**（`bible.py due ch29` 应埋/应回收两清单均空）。章末「下一章接着讲发射器」是自然的叙事衔接，非正式登记的伏笔——沿用 ch20/ch23/ch25 等章节的先例，仅在 dossier 明确列出 should_plant 时才登记 arc-map 条目。
- **跨章行号标注偏差待存量回修**：driver.py `__init__` 真实跨度是 L47-L75（非 L47-L73），npu_utils.cpp 的 `NpuUtilsMethods` 数组真实跨度是 L322-L334（非 L322-L332）；两图（fig-m2/fig-m3）另有 3 处 1-2 行的行号偏移，均已在 review-report.json 记录具体修法，留存量回修批次统一处理。
- **跨章事实精度待核实**：本章 4 处把 CUDA 对应接口写作 `cuModuleLoad`，但姊妹篇基座 ch37 实际展示的是 `cuModuleLoadData`（从内存 image 装载，语义不同于从文件名装载的 `cuModuleLoad`）——留存量回修批次统一改。
- **magic 二分不是三分**（本章反直觉点，已由 §29.5 讲清）：`kernel_mode=="aiv"` 才走 `RT_DEV_BINARY_MAGIC_ELF_AIVEC`，其余(aic/mix)统一走通用 `RT_DEV_BINARY_MAGIC_ELF`——不是 aiv/aic 各自专属一个魔数的对称三分。
- Bible 回写：glossary +8 条新词条（NPUUtils/`_build_npu_ext`/registerKernel/`rtDevBinaryRegister`+`rtFunctionRegister`/stubName+registered_names/NPULauncher/getArch+getAiCoreNum）+ 1 条既有词条（NPUDriver）补充 ch29 细节；concepts +8 条（对应 8 个机制）；figures +4 条（m1/m2/m3 三机制图 + chapter-map，登记为 `fig-ch29-chapter-map` 防跨章撞 id）；interfaces 不新增（无精简版，同 ch26-28 先例）。
