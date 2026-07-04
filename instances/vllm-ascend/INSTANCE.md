# 实例：vllm-ascend（vLLM-Ascend 源码解读：昇腾 NPU 后端如何接入 vLLM）

> 本文件 = 本实例的「源码版本 + 当前状态 + 专属规则」。通用方法论/工厂运转见仓库根 `CLAUDE.md`；实例配置见 `instances/vllm-ascend/repo2book.json`。
> 中文、高级读者。解读对象 = vllm-ascend（昇腾 NPU 的 vLLM 后端插件）。

## 源码版本与「配对依赖」（本书的关键前提）
- **锁定 vllm-ascend `v0.21.0rc1`**（commit `80610e44`，`instances/vllm-ascend/source/` 工作树即此版）。规范路径前缀 `vllm_ascend/…`。
- **配套并依赖 vLLM `v0.21.0`**（README 明示："CI commitment for vLLM main branch and vLLM v0.21.0 tag"）。这个基座**已经在本仓库** `instances/vllm/source/`（vllm 实例已锁 v0.21.0）——解读 vllm-ascend 时凡涉及它接入/改写的 vLLM 接口，直接对照该目录，无需另克隆。
- 因此本书是 vLLM 书的**姊妹篇**：vLLM 书讲引擎本体（CUDA 线），本书讲「同一个 v0.21.0 引擎如何被搬到昇腾 NPU 上」。

## 它是什么（一句话 解读）
vllm-ascend 是 vLLM 的 **out-of-tree 平台插件**：不改 vLLM 源码，而是经 **setuptools entry points** 把自己注册进去，再用 **monkey-patch** 替换昇腾上跑不动/跑不快的实现。
- 入口（`setup.py` entry points → `vllm_ascend/__init__.py`）：
  - `vllm.platform_plugins`: `ascend = vllm_ascend:register` → 返回 `vllm_ascend.platform.NPUPlatform`（`PlatformEnum.OOT`）。
  - `vllm.general_plugins`: `register_connector` / `register_model_loader` / `register_service_profiling` / `register_model`（在 engine-core 子进程里生效）。
- **两段式 patch**（`vllm_ascend/patch/`）：`platform/`（25 个，平台初始化期打，改 distributed / kv_cache_coordinator / multiproc_executor / mla_prefill_backend / mamba 等 vLLM 内部）+ `worker/`（22 个，worker 期打）。

## 子系统地形（~108k LoC Python + csrc/ AscendC 算子）
按体量与解读价值排序（详见 `book/cartography/ARCHITECTURE.md` 种子）：
`ops`(19k，昇腾自定义/融合算子) · `distributed`(14k，NPU 通信/并行) · `attention`(12.8k) · `worker`(12.6k，NPUWorker/ModelRunner) · `quantization`(6k) · `core`(3k，调度/KV) · `spec_decode`(2.7k) · `compilation`(2.5k，torchair/图模式) · `models`(2.2k) · `sample`(1.9k)；外加 `platform.py`（NPUPlatform 总入口）与 `patch/`（接入机制）。

## 实例专属硬规则
- 解读以 **vllm_ascend/** 为主线；对照基座写 `vllm/...`（指 `instances/vllm/source` @ v0.21.0），二者都用规范路径，**绝不**出现 `instances/*/source/`。
- 昇腾相关代码 host 无法跑（无 NPU/CANN）：精简版只验证可读控制流，行为以源码为准，不强求在本机运行 NPU 算子。
- 章节用 `ch`-前缀 slug，置于 `instances/vllm-ascend/artifacts/`。

## 当前状态（本 fork 完成）
- ✅ 锁定 v0.21.0rc1（80610e44）、blobless clone 进 source/。
- ✅ 摸清「配对依赖 vLLM v0.21.0 + OOT 插件 + 两段 patch」的接入骨架，写入本文件 + `cartography/ARCHITECTURE.md` 种子。
- ⏭ 下一步：补完整大纲（按子系统 + 接入机制分 Part），把顶层 `repo2book.json` 的 `active_instance` 切到 `vllm-ascend`，再逐章发车。

## Part VIII — 算法原理篇（primer，v4 新增）

- **规划**：`book/cartography/outline-final.json` 新增 Part VIII「算法原理篇：论文里的 DeepSeek」，6 章 `ch31`–`ch36`，全部 `mode: "primer"`：
  - `ch31-primer-mla`（MLA：低秩 KV 压缩/解耦 RoPE/权重吸收，deps `ch20`）
  - `ch32-primer-sparse-attention`（NSA→DSA/Lightning Indexer 谱系，deps `ch21`+`ch31`）
  - `ch33-primer-speculative-sampling`（拒绝采样定理+MTP+DSpark 前瞻，deps `ch29`）
  - `ch34-primer-eplb`（EPLB 均衡算法本体，deps `ch09`）
  - `ch35-primer-quantization`（量化数学：GPTQ/AWQ/SmoothQuant，deps `ch27`）
  - `ch36-primer-v4-csa-hca`（DeepSeek-V4 CSA/HCA 两级压缩混合注意力，deps `ch31`+`ch32`）
  - 发车顺序（见 RUNBOOK 发车阶段）：串行线 `ch31→ch32→ch36`（记号/概念递进），并行线 `ch33`/`ch34`/`ch35` 互不依赖。
- **硬规则 2 豁免范围**：CLAUDE.md HARD RULE 2「只做减法不做加法」的豁免**仅限 `kind=primer`** 的章——这 6 章的落地代码段仍是忠实参考实现（非杜撰），但正文主线是论文推导而非源码逐段精简，成对启用 `lint_paper_grounding` 门禁（`# PAPER` 全覆盖 + 正文出处可溯源）；其余 30 章（`ch01`–`ch30`，`mode: "code"`）不受影响，`lint_fidelity` 照常跑。
- **论文包位置**：`instances/vllm-ascend/book/papers/<slug>/`（`paper.md` 为主，部分章有辅助论文如 `ch32` 的 `paper-dsa.md`、`ch33` 的 `paper-mtp.md`、`ch35` 的 `paper-awq.md`/`paper-smoothquant.md`），`meta.json` 记来源；总索引 `book/cartography/papers-map.json`。
- **2026-07-04 gap 盘点**（`book-gap-audit` workflow 首跑）：全书 30 章码章体检出 6 处「悬崖」——正文引用了论文级机制但未展开推导，对应 6 章 primer 消解：
  | 悬崖 | 首现处 | 消解章 |
  |---|---|---|
  | 解耦 RoPE（为何不能吸收进 W_UK） | ch20 MLA on NPU | ch31 |
  | DSA/Lightning Indexer 谱系（NSA→V3.2 演进） | ch21 attention backend | ch32 |
  | 拒绝采样定理 + MTP 草稿机制 | ch29 | ch33 |
  | EPLB 重排算法本体（只讲工程接入未讲均衡算法） | ch09 | ch34 |
  | 量化数学（GPTQ/AWQ/SmoothQuant 推导） | ch27 昇腾量化框架 | ch35 |
  | DeepSeek-V4 CSA/HCA 两级压缩注意力 | ch21/ch22/ch30 一带而过 | ch36 |
  - 指路框补丁（6 章全部 APPROVED 后）：在 ch20/ch21/ch29/ch09/ch27 各定点插入一句「本章默认你已了解 X；其数学推导见第 NN 章」回指对应 primer 章（ch36 的 V4 CSA/HCA 暂无需单独补丁，随 ch21/ch22/ch30 的既有指路一并覆盖）；随后重跑 `book-gap-audit` 验证 6 处悬崖降级为「已建立/有指路」，报告存 `book/audits/`。
