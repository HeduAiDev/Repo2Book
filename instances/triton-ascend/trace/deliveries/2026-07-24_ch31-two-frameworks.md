# ch31 交付：一套后端，两个框架——torch_npu / mindspore 策略注册表

- **Type**: delivery
- **Chapter**: ch31
- **Date**: 2026-07-24
- **Timestamp**: 2026-07-24T09:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, revise-writer, revise-fig, Lead, archivist
- **User present**: False
- **Tags**: triton-ascend, part-6, backend-runtime, deep, skip_impl, part-6-addendum, strategy-registry, two-level-registry, register-decorator, backend-policy, dual-backend, torch-npu, mindspore, set-literal, dossier-escape, review-escape, lead-takeover, ch29-correction

## What happened

Part 6「后端与运行时」**补遗章**（deep + skip_impl），deps=ch26（承 ch26 AscendBackend 契约装配站）。讲昇腾比基座 GPU **多出的一层**：一套后端服务 `torch_npu` / `mindspore` 两个宿主框架的差异收拢机制。**GPU 侧无此层**——单框架无需注册表去收拢框架差异，故本章**对位无**（基座《Triton 源码解读》无对应章）。

正文围绕 `backend_register.py` 的**两级策略注册表**展开：

**表怎么长成**：`BackendStrategyRegistry` 是一张 `category=框架（行）× method=能力（列）` 的两级字典——`|category| × |method| = 2 × 15 = 30` 格，每格放该框架对该能力的一份实现（m1）。`register(category, method)` 返回的**登记装饰器**贴在每个能力实现 `def` 上、在**导入期**执行：`category` 缺则新建空 dict，再 `if method in strategies[category]: raise`（L34）后存 `method→func` 并 `return func` 原样。三条 fail-fast——重复登记同格（`already registered`，L34）/ 查表缺 category（`not registered`，L41）/ 缺 method（L43）——绝不静默覆盖或返回 None。文件里赫然有两个 `def version_hash`（mindspore 版 L73-76 / torch_npu 版 L79-83），**看着像后者覆盖前者、其实不是 bug**：装饰器在每个 def 定义完的瞬间就把那个函数对象塞进注册表各存一份（trace 实证两格 id 相异），被第二个同名 def「覆盖」的只是模块级名字 `version_hash`（最终指向最后一个 def），注册表持有的两份实现互不影响（m4）。`_LazyBackendStrategyRegister` **懒单例名不副实**——名带 Lazy，实际 `@register` 在模块导入期就全部执行完（懒的只是这层包装对象、非注册动作），它真正兑现的是 `backend_strategy_registry` **全局唯一**：整个文件所有 `@register` 与运行时所有 `execute_func` 打的是同一张表。「一套后端」的物理落点就是这一个导出单例。

**表怎么用**：运行时对外入口 `get_backend_func`（`utils.py:L42-L53`）先把活动框架解析成 **`backend_policy`**——首行 `if backend_policy is None:` 是唯一解析闸门：env `TRITON_BACKEND` 优先 → 能否 `import torch_npu` 自动探测兜底 → 首次赋值后**进程内粘滞不变**（改 env 也不生效，trace `m2_case_cache_sticky_before=after=mindspore` 实证）；随后直落 `execute_func(backend_policy, method, *args)` **两级查表分派**（先守 category 再守 method，两道都过才 `return strategies[category][method](*args)`，L44，两次 O(1) 哈希）。**能力族**同一能力两框架各一套，产物可全量不同：`cxx_abi`（mindspore 版 `return 0` / torch_npu 版读真实 ABI）；`header_file`（mindspore 版 5 行 vs torch_npu 版 3 行 C++、逐行比对**零共享一行**，且藏一处 **set-literal 孤例**——`{enable_taskqueue}` 是**单元素集合**、`x in {y}` 恒等价 `x == y`，是全语料唯一这样写的地方，本章**内嵌该 `# SOURCE` 片段并诚实点破其恒真语义、不当 bug 改**）。**框架 import 全写在被装饰函数体内**（`backend_register.py` 顶层只 `import os/typing`），故 host 可导入真跑「框架无关」的能力、只登记查表不调用需 import 框架的能力。

**3 张图**全部通过盲审/Lead 核：`fig-ch31-registry-grid`（2 框架 × 15 能力 = 30 格两级字典 + 三条 fail-fast 出口）/ `fig-ch31-dispatch-flow`（get_backend_func 解析 backend_policy → 两级查表 → 同一 header_file 两框架 5 vs 3 行零共享）+ chapter-map（登记为 `fig-ch31-chapter-map` 防跨章撞 id，5 §徽标 一~五↔5 节自然标题）。

**16 门禁全绿**：source_grounding / structure / formulas / dossier / explainer / trace_consistency / chapter_map --require / diagram_geometry --all / diagram_scaffolding --all / anchors / punct / ir_opname 等。verdict **APPROVED**。

**交付曲折（两度逃逸 + Lead 接管，如实入卷）**：① **Dossier-verify 站**对抗性自核抓出 `cxx_abi` 能力的 **`delete` 计划与 `must_keep` 清单矛盾**（一处标 delete、另一处列为必保留符号），dossier 内部不自洽 → 返回 BLOCKED 升级 Lead → **Lead 手工修 dossier 消解矛盾后 `skip_dossier` 重跑**工作流。② **Review round3** 一个评审 agent **API 崩溃**（Connection closed）→ workflow 逃逸未到 Archive → **Lead 接管**余下评审与收口。真实评审发现在收口前均已修：header_file set-literal（内嵌 `# SOURCE` + 诚实点破 `{enable_taskqueue}` 恒真）、`fig-ch31-registry-grid` **箭头落点纠错**（命中/重复登记两箭头改落进 `cxx_abi` 行 × mindspore 列的绿色 `return 0` 格、不再落最底 `nonexistent_cap` 行）+ **独立盲审 PASS**、§39 枚举序、m3/m6 补「直觉/不变量」标签、三处相似名（`BackendStrategyRegistry` / `_LazyBackendStrategyRegister` / `backend_strategy_registry`）打通辨析、cxx_abi ABI 括注补齐。**Map + Archive 由 Lead 接管补完**（chapter-map 生成 + Lead Read-PNG 核 5 §徽标逐一对应正文自然标题、节点符号均 `backend_register.py` 真实符号、834×450 + writer 插引）。

**ch29 勘误联动**：ch29 曾把 `get_backend_func` 当黑盒留白（当时记为 reader-comprehension non-blocking）；本章讲清其解析/粘滞/两级查表机制后，回补 ch29 的定性——**Lead 统一提交时含 ch29 勘误**（本次归档不 commit）。

**交叉验证（skip_impl）**：无 implementation/tests 目录。但 `backend_register.py` 顶层仅 `import os/typing`、框架依赖全写在被装饰函数体内，故真实模块**可在 host 直接导入并驱动**：driver = `explainer/traces/run_registry.py` **host 真跑注册表**——建表（`2 × 15 = 30` 格）+ 查表分派（命中 `mindspore/cxx_abi → 0`、缺 category / 缺 method / 重复登记三 fail-fast、`header_file` 两框架产物差异、`version_hash` 双同名 def 对象身份核验），原始输出 `explainer/traces/registry_trace.json`、`trace_source=run`。仅「框架无关」能力被真调用；需 `import torch/mindspore` 的能力（`version_hash`/`type_convert`）只登记查表、不在 host 调用。m2 的 `backend_policy` 解析逻辑因 `utils.py` 携 triton 重依赖无法 host 导入，故在 driver 中**逐字复刻** `utils.py:L42-L52` 解析分支，但分派用的 registry 是真实单例、`execute_func` 查表与 `utils.py:L53` 字节一致。

## Why it matters

ch31 补上 Part 6 的一块拼图：全书主线（ch01→ch30）讲的是「一份 Triton kernel 怎么编译、装载、发到达芬奇核上跑」，但**昇腾工程现实里从来不是一个宿主框架**——同一套后端要同时服务 `torch_npu`（PyTorch 昇腾适配）与 `mindspore`（华为自研框架）两条上层生态。本章把「一套后端服务两框架」这句愿景落成 `backend_register.py` 里可 host 真跑的一张两级表：**用注册表把 O(能力数 × 框架数) 份差异实现收进一张 30 格的字典**，消费点 `get_backend_func('...')` 一律不动——新增框架 = 加一整行，新增能力 = 每行加一列。这正是 ch30「双后端策略表」在发射器 wrapper 上那个下游用法的**上游机制本身**。

它也补齐了两条方法论承重：① **fork 路线的代价与红利**——GPU 侧单框架没有这一层，昇腾多出它，是「一份代码库服务多生态」的结构性成本，也是注册表模式化解它的红利；② **读真实源码的孤例教学**——`header_file` 那处 `{enable_taskqueue}` 单元素集合恒真写法，是真实工程代码里的「怪味道」，本章选择**内嵌原样源码并诚实点破恒真语义、既不美化也不当 bug 改**，延续全书「以真实源码为准、不杜撰不篡改」的底线。

方法论层面还留下两条：**dossier 对抗性自核在 Dossier-verify 站拦住 `delete↔must_keep` 矛盾**（避免下游 implementer/writer 各吃到不一致口径），与**评审 agent API 崩溃当作 BLOCKED 逃逸、由 Lead 接管人工收口、绝不当作通过**（同 ch09/ch24/ch25 先例）——都是「宁可升级，不可假通过」的再次印证。

## What to remember

- **本章无 arc-map 伏笔动作**：`bible.py due ch31` 应埋/应回收两清单均空；`f1-f7` 至 ch28 已全部 resolved。deps=ch26 但非伏笔回收——补遗章承接的是机制上下文（AscendBackend 契约装配）而非登记过的伏笔条目。
- **两级注册表口径（跨章承重）**：`category=框架（行）× method=能力（列）`，`2 × 15 = 30` 格。分派 = 先选行（框架）再选列（能力）两次 O(1) 哈希命中唯一一格。三条 fail-fast：重复登记（L34）/ 缺 category（L41）/ 缺 method（L43）。ch30 的「双后端策略表」是它的下游用法。
- **三处相似名别混**：`BackendStrategyRegistry`（注册表**类**）/ `_LazyBackendStrategyRegister`（**懒单例包装**，名不副实、@register 导入期即执行完、真正兑现全局唯一）/ `backend_strategy_registry`（导出的**全局唯一实例**）。本章专门打通辨析。
- **两个同名 `def version_hash` 不是 bug**：装饰器在 def 定义完瞬间就把函数对象塞进注册表各存一份（trace 两格 id 相异），被重绑的只是模块级名字（指向最后一个 def），注册表两份实现互不影响。
- **backend_policy 粘滞**：`get_backend_func` 首行 `if backend_policy is None:` 是唯一解析闸门；env 优先 → import torch_npu 自动探测 → 首次赋值后进程内单调不变（改 env 不生效，trace 实证）。
- **set-literal 孤例**：`header_file` 实现里 `x in {enable_taskqueue}` 是单元素集合、恒等价 `x == enable_taskqueue`，全语料唯一。本章内嵌 `# SOURCE` 并诚实点破恒真语义，不当 bug 改——记住这个「读真实源码孤例」的处理范式。
- **诚实边界（skip_impl）**：无精简版。交叉验证靠 `backend_register.py` 逐行核 + `run_registry.py` **host 真跑**注册表建表/查表（`trace_source=run`），因顶层只 import os/typing、框架 import 在函数体内才可 host 跑；需 import torch/mindspore 的能力只登记查表不调用；m2 解析逻辑逐字复刻 `utils.py:L42-L52`（utils.py 带 triton 重依赖导不进），分派仍用真实单例。
- **两度逃逸 + Lead 接管**：① Dossier-verify 抓 cxx_abi `delete↔must_keep` 矛盾 → Lead 修 dossier `skip_dossier` 重跑；② Review round3 评审 agent API 崩 → Lead 接管收口 + 补 Map/Archive。真实发现已修（header_file 内嵌+点破 / fig 箭头落点纠错+独立盲审 / §39 枚举序 / m3·m6 标签 / 三相似名 / ABI 括注）。
- **ch29 勘误联动**：ch29 把 `get_backend_func` 当黑盒的留白由本章回补定性；**Lead 统一提交时含 ch29 勘误**（本次归档不 commit）。
- Bible 回写：**glossary +10**（`BackendStrategyRegistry` / `register 装饰器` / `execute_func` / `get_backend_func` / `backend_policy` / `_LazyBackendStrategyRegister` / `backend_strategy_registry` / `两级注册表` / `能力族` / `set-literal 孤例`）；**concepts +10**；**figures +3**（`fig-ch31-registry-grid` / `fig-ch31-dispatch-flow` + chapter-map 登记为 `fig-ch31-chapter-map` 防跨章撞 id，现 183 条）；**interfaces 不新增**（skip_impl，无精简版，同 ch26-30 先例）。
