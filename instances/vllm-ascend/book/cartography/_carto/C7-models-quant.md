# C7 — 模型 / 量化 / 投机解码 / 采样 / LoRA（昇腾 NPU 对位）

对象：vllm-ascend v0.21.0rc1，源码 `instances/vllm-ascend/source/`，前缀 `vllm_ascend/…`。
基座对照：vLLM v0.21.0 `instances/vllm/source/`，前缀 `vllm/…`。
vllm-ascend = vLLM 的 OOT 平台插件（昇腾 NPU 顶替 CUDA）。

## 子系统规模（实测 LOC）

| 目录 | LOC | 备注 |
|---|---|---|
| `quantization/` | 6083 | 最大；3 个 config 入口 + 13 个 method scheme + registry/adapter |
| `spec_decode/` | 2737 | 1 个 2043 行巨无霸 `llm_base_proposer.py` + 8 个薄 proposer 壳 |
| `models/` | 2244 | 只有 deepseek_v4 / deepseek_v4_mtp 两个昇腾特化模型 + attention layer |
| `model_loader/` | 2205 | netloader（弹性网络加载）+ rfork（fork 加速冷启动），与主线推理弱相关 |
| `sample/` | 1865 | rejection_sampler 1518 + sampler 302 + penalties 45 |
| `lora/` | 567 | lora_ops（NPU 自定义算子壳）+ punica_npu + 类替换 utils |
| `profiler/` | 106 | torch_npu profiler 薄封装 |
| `model_executor/` | 263 | 只有一个 offloader/prefetch |

## 核心发现：四种"顶替 CUDA"的接入范式

这个杂烩子系统里所有目录其实在重复**同一个 OOT 插件套路**——找到 vLLM 的扩展点，注册/子类化/换实现把昇腾顶进去。共四种范式，是本子系统最大的教学主线：

1. **注册表 + 工厂（quantization）**：三个 config 类用 `@register_quantization_config(...)` 注进 vLLM 全局量化注册表（`modelslim_config.py:401` `AscendModelSlimConfig`、`compressed_tensors_config.py:52`、`fp8_config.py:56`；compressed_tensors 甚至先把 vLLM 原版从 `QUANTIZATION_METHODS` 删掉再替换，见 `compressed_tensors_config.py:41`）。`get_quant_method()`（`modelslim_config.py:512`）按 layer 类型 + quant_type 字符串，经 `methods/registry.py` 的 `_SCHEME_REGISTRY` 取 scheme，再用 `method_adapters.py` 的三个 wrapper（`AscendLinearMethod`/`AscendKVCacheMethod`/`AscendFusedMoEMethod`）把 scheme 包成 vLLM 的 `LinearMethodBase`/`BaseKVCacheMethod`/`FusedMoEMethodBase`。**三层解耦**（config 解析 → registry 选 scheme → adapter 适配 vLLM 接口）是全仓最干净的设计，13 个 scheme（W4A4/W4A8/W8A8/MXFP4/MXFP8/FP8/KV-C8…）全挂在 `base.py` 的三个 ABC（`AscendLinearScheme`/`AscendAttentionScheme`/`AscendMoEScheme`）下。

2. **工厂分发（spec_decode）**：`spec_decode/__init__.py:33` `get_spec_decode_method()` 一个 if-elif 把 method 字符串映射到 8 个 `Ascend*Proposer`。多数 proposer 是薄壳——`AscendNgramProposerNPU`（35 行）直接继承 vLLM `NgramProposerGPU` 把 GPU 当 NPU 用、`draft_proposer.py`/`eagle_proposer.py` 各 17/19 行。真正的重量级在 `llm_base_proposer.py`（2043 行，继承 vLLM `SpecDecodeBaseProposer`），重写 prepare_inputs/propose，用了 ACLGraph、昇腾 Triton kernel（`ops/triton/spec_decode`）、MLA、各种昇腾并行组。

3. **子类化覆写（sample）**：`AscendSampler(Sampler)`（`sampler.py:45`）、`AscendRejectionSampler(RejectionSampler)`（`rejection_sampler.py:34`）继承 vLLM 基类，只覆写几处。两条 NPU 主线：(a) `random_sample` 用 `exponential_().div_().argmax()` 的 Gumbel 技巧避开 `torch.multinomial` 的 CPU-NPU 同步（`sampler.py:19-42`），并用 `npu_stream_switch`/独立 stream 做异步指数随机；(b) penalties 走昇腾 Triton（`ops/triton/reject_sample`），Triton 不可用时优雅回退 vLLM 默认实现（`HAS_TRITON` 分支随处可见）。

4. **运行时类替换（LoRA）**：`PunicaWrapperNPU(PunicaWrapperBase)`（`punica_npu.py:14`）在 `__init__` 里按 device/rank 二选一绑定 lora ops——大 rank(≥128)/310P 用 vLLM torch_ops，否则用 `lora/lora_ops.py` 的昇腾自定义算子 `torch.ops._C_ascend.{bgmv,sgmv}_*`（纯薄壳调 C++ 算子）。`lora/utils.py:70` `refresh_all_lora_classes()` 把 4 个 `Ascend*LinearWithLoRA` 类**追加进 vLLM 全局 `_all_lora_classes` 元组**，让 vLLM 的 layer 替换机制能识别昇腾的 `AscendQKVParallelLinear`。

模型注册同理：`models/__init__.py` 用 `ModelRegistry.register_model(...)` 注 DeepseekV4/MTP；自定义 loader 用 `@register_model_loader("netloader")`（`model_loader/netloader/netloader.py:38`）、rfork 同。**全是"注册到 vLLM 扩展点"的变体。**

## 量化细节锚点

- ModelSlim 路线：读 checkpoint 里 `quant_model_description.json`（`MODELSLIM_CONFIG_FILENAME`），逐层字符串（如 `"W8A8_DYNAMIC"`）决定 scheme。`get_linear_quant_type`（`modelslim_config.py:297`）处理 packed/fused 模块（qkv_proj、gate_up_proj、experts）的"分片必须同 quant 类型"校验。
- `packed_modules_model_mapping`（`modelslim_config.py:53`）和 `QUANT_MODEL_SUBSTR_MAPPINGS`（:271）是一大坨模型名→模块名映射表，给读者讲"为什么量化要懂模型结构"很好的实例，但不必逐条讲。
- MXFP 系（`quant_parser.py` `QuantTypeMapping`）：W8A8_MXFP8/W4A4_MXFP4/W4A8_MXFP 的 act/weight/scale dtype 三元组，依赖 `device/mxfp_compat` 的 `FLOAT4_E2M1FN_X2_DTYPE`/`FLOAT8_E8M0FNU_DTYPE`——昇腾硬件 microscaling 格式，是 NPU 特化最硬核处。
- `method_adapters.py:140` `AscendLinearMethod.apply` 里一大段 `tp_rank` 选择（oproj_tp/mlp_tp/flashcomm2/dsa_cp）耦合昇腾并行特性，讲时可裁剪只留主干。

## 教学价值排序

- **高**：quantization 三层接入范式（registry+adapter+ABC scheme）、sample 的 multinomial 规避 + Triton 回退、LoRA 的全局类替换 trick——都是 OOT 插件"如何无侵入顶替 CUDA"的范式样本。
- **中**：spec_decode 工厂分发 + 薄壳继承模式（巨无霸 llm_base_proposer 只挑骨架讲，全讲会失焦）；deepseek_v4 作为昇腾特化模型唯一样本，可与 vLLM ch25 对位讲"同一个 DeepSeek-V4 在 NPU 上改了哪些 layer/op"。
- **低（建议略或一笔带过）**：model_loader 的 netloader/rfork（弹性加载/fork 冷启动，是部署优化，偏离推理主线，且 2.2k 行很专门）、profiler（106 行封装）、model_executor offloader。

## 建议章节（2–4 章；按主题不按目录）

### Ch-A《昇腾量化框架：把 NPU 量化方案接进 vLLM》
- focus：`@register_quantization_config` 三入口 → `get_quant_method` 分发 → `methods/registry._SCHEME_REGISTRY` 选 scheme → `method_adapters` 三 wrapper 适配 vLLM `LinearMethodBase`/`FusedMoEMethodBase`/`BaseKVCacheMethod`。以 W8A8_DYNAMIC 或 W4A8 走通一条 create_weights→apply 全链，点出 MXFP microscaling dtype 是 NPU 硬特化。ModelSlim `quant_model_description.json` 逐层解析。
- key_source_paths：`vllm_ascend/quantization/{modelslim_config.py(401,512,297),method_adapters.py,quant_parser.py}`、`vllm_ascend/quantization/methods/{registry.py,base.py,w8a8_dynamic.py,w4a8.py}`
- pairs_with：`vllm/model_executor/layers/quantization/`（base_config.QuantizationConfig、register_quantization_config、LinearMethodBase）+ vLLM 书 ch—（量化对应章，若无则对 ch22 权重加载）
- teach_value：高。OOT 注册表+适配器范式最干净的样本。
- est_size：中-大（quant 是 6k LOC 最大块）
- deps：vLLM 量化框架基础、FusedMoE/Linear layer 概念（前置子系统章）

### Ch-B《采样与投机解码的 NPU 对位：规避同步、Triton 回退、proposer 工厂》
- focus：(1) `AscendSampler`/`AscendRejectionSampler` 子类化，`random_sample` 用 Gumbel-exponential 避 `torch.multinomial` 的 CPU-NPU 同步 + 独立 stream 异步随机；(2) penalties/top-k-top-p 走昇腾 Triton，`HAS_TRITON` 优雅回退；(3) `get_spec_decode_method` 工厂 + 8 个薄壳 proposer 继承模式，挑 `llm_base_proposer` 骨架（prepare_inputs/propose + ACLGraph）讲，不逐行。
- key_source_paths：`vllm_ascend/sample/{sampler.py,rejection_sampler.py,penalties.py}`、`vllm_ascend/spec_decode/{__init__.py,llm_base_proposer.py(挑骨架),ngram_proposer_npu.py}`
- pairs_with：`vllm/v1/sample/{sampler.py,rejection_sampler.py}` + `vllm/v1/spec_decode/` → vLLM 书 ch27《The Sampling Pipeline》+ ch28《Speculative Decoding: Proposers and Rejection Sampling》
- teach_value：高（采样部分）/ 中（spec_decode）。
- est_size：中
- deps：vLLM 采样/rejection sampling 机制（ch27/ch28）、NPU stream/Triton 背景

### Ch-C（可选，建议并入或轻量）《模型与 LoRA 的昇腾接入：注册、类替换、自定义算子》
- focus：`ModelRegistry.register_model` 注 DeepseekV4/MTP（昇腾唯一特化模型，挑改动 layer 与 vLLM ch25 对位）；LoRA 的 `PunicaWrapperNPU` device 分支选 ops + `refresh_all_lora_classes` 追加进 `_all_lora_classes` 全局元组的类替换 trick + `torch.ops._C_ascend` 自定义算子薄壳。可顺带一笔 model_loader 的 `@register_model_loader` 扩展点（netloader/rfork 不深讲）。
- key_source_paths：`vllm_ascend/models/{__init__.py,deepseek_v4.py(挑差异)}`、`vllm_ascend/lora/{punica_npu.py,lora_ops.py,utils.py}`、`vllm_ascend/model_loader/netloader/netloader.py:38`（仅扩展点）
- pairs_with：`vllm/model_executor/models/deepseek_v2`、`vllm/lora/punica_wrapper/`、`vllm/model_executor/model_loader/` → vLLM 书 ch22《Model Definitions and Weight Loading》+ ch25《DeepSeek-V4》
- teach_value：中。LoRA 全局类替换 trick 是亮点；model_loader 建议略。
- est_size：中-小
- deps：vLLM 模型定义/LoRA punica（ch22/ch25）

> 建议：A、B 必出；C 视篇幅并入 B 末或独立。共 **2–3 章**。netloader/rfork/profiler/model_executor 建议不单独成章。
