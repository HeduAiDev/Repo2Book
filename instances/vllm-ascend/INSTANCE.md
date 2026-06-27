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
