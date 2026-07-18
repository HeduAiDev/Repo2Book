# ch01 交付：鸟瞰——一个 fork 了 Triton、却把整条下降链换成昇腾 NPU 路的后端

- **Type**: delivery
- **Chapter**: ch01
- **Date**: 2026-07-18
- **Timestamp**: 2026-07-18
- **Agents involved**: analyst, explainer, illustrator, ch01writer, reviewer, Lead, archivist
- **User present**: false
- **Tags**: triton-ascend, flagship, birdseye, part-1, skip_impl, fork-vs-plugin, three-stage-lowering, davinci-dual-core, dossier-escape, book-map, first-archive

## What happened

triton-ascend 全书**第一章、也是全书第一次归档**（bible init 为空）。kind=skip_impl/meta（flagship 开篇鸟瞰，不逐行解读源码、三支柱各挑最小源码锚点内嵌）。verdict=**APPROVED**，全 linter green，8 图（含全书 book-map，roadmap 不计）全部盲审 PASS。

**三支柱主线**：
1. **fork，不是插件**——上游 Triton v3.2.0 整树在内、昇腾增量原位放 `third_party/ascend/`。血统证据在 `compiler.py:L34` 的 `from triton._C.libtriton import ir, passes, ascend` 与 `tutorials/01-vector-add.py:L1-L3` 三行叠加版权头，**不在** `compiler.py` 自身版权头（顶部是 Huawei-only 单版权）。能力理由：换整条下降链只有 fork 做得到，插件的『注册表顶替』碰不到 add_stages/OpBuilder/方言层。
2. **三段结构化下降链 ttir→ttadapter→npubin**——`add_stages`（compiler.py:L939）只登记三段线性节点，对照基座 GPU 路五段（ttir/ttgir/llir/ptx/cubin）。第二段 `ttadapter`（triton_adapter）抛弃 SIMT 指针模型、把 tensor-of-pointers 逆向还原成结构化 Linalg memref；分叉精确落在此、根因是 NPU 非 SIMT。npubin 默认 A2_A3、交闭源 bishengir-compile，其内部最终落 AscendC 库调用。
3. **达芬奇 cube/vector 双核**——目标硬件双核异构 + UB/L1/L0A/L0B/L0C/GM 显式内存层级，落成 `ascend_ir.cc` 的 `CoreType` 与 `AddressSpace` 两枚举。cube:vector = 1:2 由 ch02 建（不从枚举读出）。

外加全书 **book-map**：7 Part / 33 章沿下降链展开，ch02、ch09 标『原理』先修徽标。

**Dossier escape 经过（值得记）**：Dossier 站 `dossier-verify` 对抗性自核**抓出 Lead 发车 focus 的事实错**——(a) `compiler.py` 版权头是 Huawei-only 而非双版权（双版权在 tutorials/01），fork 血统实际在 upstream import；(b) NPUOptions 行号 L704→L705；(c) npubin 默认走 A2_A3 分支；(d) cube:vector=1:2 应归 ch02、非本章。Lead 据此修 dossier + `skip_dossier` 复跑。教训：**Lead 发车 focus 本身也过对抗性自核，善。**

**Reviewer 判 9 项全 non-blocking**（flagship 开篇计），Lead 派 ch01writer 落地 7 处强项：AscendC 最小 gloss（图注词接正文，细节留 ch25）、memref 技术定义前移到首现处、OpBuilder 落点（留 ch04 双 builder）、load_dialects 双现区分（ascend_ir.cc L492 注册 annotation/hivm/scope vs triton_ascend.cc L361 注册 TritonAscendDialect，分属不同 pybind 子模块）、TCoreType 四值组合值指路（P4 核亲和）、m04 pass 链量化+顺序必要性 invariant、m06 直觉类比段。

## Why it matters

这是**全书译名与归档格式的基线**：后续 32 章的术语（三段下降链/结构化 memref/达芬奇双核/UB-GM 显式搬运/CoreType/AddressSpace…）都对齐本章 glossary。三支柱是全书心智模型——"同前端异后端 / fork+原位增量 / 抛弃指针换结构化 / 目标硬件双核"，后面每一章的差异追根溯源都在 ch01 立的这几点。dossier escape 证明对抗性自核连 Lead 的发车 focus 都能纠错，是流程可靠性的正面样本。

## What to remember

- **诚实边界**：host 无 NPU/CANN/bishengir，运行时轨迹一律标『需真机』；skip_impl 只内嵌真源码锚点作自包含，能出的编译期行号照读、已逐条 grep 核对。
- **埋下的全书前向线索**（非 arc-map 正式伏笔，首章不硬造；全书伏笔自顶向下从 outline 依赖图注入）：三段下降链→P3/P5 展开；达芬奇双核数量比 1:2→ch02 原理篇量化；结构化 Linalg/memref 数学根基→ch09，分水岭机制→ch10 triton_adapter，addptr→memref 逆向→ch11 PtrAnalysis；AscendC 终点→ch25；双 builder / OpBuilder→ch04；核亲和（该上哪个核）→ch16；显式内存层级落 buffer 语言→ch05；force_simt_only 旁路→ch20。
- **事实校准点**（勿再回退）：compiler.py 版权头 Huawei-only；fork 血统在 upstream import；NPUOptions 在 L705；npubin 默认 A2_A3；cube:vector=1:2 归 ch02 不归 ch01。
