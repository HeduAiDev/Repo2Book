# vllm-ascend 架构地图（种子）

> 锁定版本: vllm-ascend **v0.21.0rc1**（commit `80610e44`，源码工作树即此版） · 配套依赖: vLLM **v0.21.0**（`instances/vllm/source`）· 本文件为 fork 的初版 解读种子，待 cartography 流程补全为全量地图 + 大纲。

## 心智模型：一个「不改 vLLM、却接管整条执行路径」的 OOT 插件
vLLM v0.21.0 是引擎本体（设备无关骨架 + CUDA 实现）。vllm-ascend 不 fork、不改 vLLM，而是在**安装期**经 setuptools entry points 把自己挂进去，在**运行期**经平台抽象 + monkey-patch 把昇腾 NPU 的实现替换进每一层。读这本书 = 沿 vLLM 的执行主线，看每一站「昇腾版」如何顶替 CUDA 版。

## 三个接入支柱（本书的脊柱）
1. **Entry points（安装期注册）** — `setup.py` → `vllm_ascend/__init__.py`：
   - `vllm.platform_plugins: ascend = vllm_ascend:register` → `NPUPlatform`（`vllm_ascend/platform.py`，`PlatformEnum.OOT`）。vLLM 启动时探测到 NPU 即选中它，由它回答"用哪个 attention/worker/communicator/编译后端"。
   - `vllm.general_plugins`: `register_connector`/`register_model_loader`/`register_service_profiling`/`register_model` —— 在 engine-core 子进程内注册 KV 连接器、模型加载器、profiling、模型。
2. **Platform 抽象（运行期分发）** — `NPUPlatform` 实现 vLLM `Platform` 契约：给出 attention backend、device communicator、worker 类、自定义编译 pass key 等，把"设备相关的选择"全收口到一处。
3. **Patch（运行期改写 vLLM 内部）** — `vllm_ascend/patch/`，两段式：`platform/`（25 文件，平台初始化期）+ `worker/`（22 文件，worker 期），替换 vLLM 里昇腾跑不动/需特化的点：`patch_distributed` / `patch_kv_cache_coordinator` / `patch_kv_cache_interface` / `patch_multiproc_executor` / `patch_mla_prefill_backend` / `patch_mamba_*` 等。

## 子系统地形（按体量 / 解读价值）
| 子系统 | 路径 | 量级 | 解读要点 |
|---|---|---|---|
| 自定义算子 | `vllm_ascend/ops/` + `csrc/` | ~19k | AscendC/融合算子、对 vLLM 算子的 NPU 替换 |
| 分布式 | `vllm_ascend/distributed/` | ~14k | NPU 通信器、并行、KV 传输 |
| 注意力 | `vllm_ascend/attention/` | ~12.8k | 昇腾 attention 后端 / MLA prefill |
| Worker | `vllm_ascend/worker/` | ~12.6k | NPUWorker / NPUModelRunner（对 vLLM gpu_worker/runner 的对位实现）|
| 量化 | `vllm_ascend/quantization/` | ~6k | 昇腾量化方案 |
| 调度/KV | `vllm_ascend/core/` | ~3k | 调度与 KV cache 的 NPU 特化 |
| 投机解码 | `vllm_ascend/spec_decode/` | ~2.7k | 对位 vLLM v1 spec-decode |
| 编译 | `vllm_ascend/compilation/` | ~2.5k | torchair / 图模式 |
| 模型 | `vllm_ascend/models/` | ~2.2k | 昇腾特化模型实现 |
| 采样 | `vllm_ascend/sample/` | ~1.9k | 采样算子 NPU 版 |
| 平台/接入 | `platform.py` · `patch/` · `__init__.py` | — | 上面「三支柱」 |

## 建议的大纲骨架（待 cartography 细化）
- **Part 0 接入机制**：entry points → NPUPlatform → patch 两段式（本书的"为什么能不改 vLLM 就接管"）。
- **Part 1 设备与进程**：platform / device / 通信器 / multiproc_executor patch。
- **Part 2 执行主线**：NPUWorker / NPUModelRunner / forward context（对照 vLLM gpu_worker/runner）。
- **Part 3 注意力与 KV**：attention 后端 / MLA prefill / kv_cache 协调器 patch。
- **Part 4 算子与编译**：ops + csrc（AscendC）/ torchair 图模式。
- **Part 5 量化 / 投机 / 采样 / 模型**：各子系统的 NPU 对位实现。
> 每章主线 = vllm-ascend 源码，对照基座 vLLM v0.21.0（`instances/vllm/source`）说明它顶替/扩展了哪一站。
