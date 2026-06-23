# ch24 精简版 implementation notes — 注意力后端抽象与元数据

只做减法的忠实子集，与 vLLM 同名/同结构/同控制流。所有删除标 `# SUBTRACTED:`，每个 def/class 标
`# SOURCE: vllm/...:Lxxx`。两个真实 CUDA 算子（`reshape_and_cache_flash` /
`flash_attn_varlen_func`）在 host 无法运行，据 dossier 记录的可观察语义给出等价 CPU 实现（标
`# SUBTRACTED` 说明真实是 CUDA kernel），让精简版能在 host 跑出与 vLLM 一致的 paged 读写数值。

## 模块划分（镜像真实 vLLM 文件）

| 精简版文件 | 对应真实 vLLM 文件 |
| --- | --- |
| `backend.py` | `vllm/v1/attention/backend.py` |
| `registry.py` | `vllm/v1/attention/backends/registry.py` |
| `selector.py` | `vllm/v1/attention/selector.py` |
| `platform_cuda.py` | `vllm/platforms/cuda.py`（选后端部分） |
| `flash_attn.py` | `vllm/v1/attention/backends/flash_attn.py` + `vllm/_custom_ops.py`（`reshape_and_cache_flash`）+ `vllm/v1/attention/backends/utils.py`（`set/get_kv_cache_layout`） |
| `attention_layer.py` | `vllm/model_executor/layers/attention/attention.py` + `vllm/forward_context.py`（替身） |

## 1:1 Source Map（精简版符号 ↔ vllm/...:Lxxx ↔ 改动 ↔ 原因）

| 精简版符号 | 真实位置 | 改动 | 原因 |
| --- | --- | --- | --- |
| `AttentionBackend`（六抽象 staticmethod + 能力探针 + `validate_configuration`） | `vllm/v1/attention/backend.py:L55` | 保留六抽象方法/`validate_configuration`/`get_kv_cache_*`；删 `get_preferred_block_size`、若干同构能力探针的少量项 | 抽象核心必讲；删项为同构旁支，不破坏选后端判定 |
| `CommonAttentionMetadata` | `vllm/v1/attention/backend.py:L352` | 保留核心 8 字段（含 `block_table_tensor`/`slot_mapping`/`causal`）；删 DCP/encoder/已弃用字段与方法 | f14 接口字段必讲；旁支字段服务 cross-attn/DCP/向后兼容 |
| `AttentionMetadataBuilder.build` / `update_block_table` | `vllm/v1/attention/backend.py:L582` / `L601` | 保留 `build` 中心入口与 `update_block_table`；删 cudagraph/drafting/cascade 默认壳 | `build` 是 Common→专属 metadata 唯一翻译入口 |
| `AttentionImpl.forward` / `do_kv_cache_update` | `vllm/v1/attention/backend.py:L788` / `L910` | 保留抽象 `forward`/`do_kv_cache_update`；删 fused/rope 钩子、MLA 变体类 | 算注意力与写 KV 的落点；MLA 是独立专题 |
| `AttentionBackendEnum` + `get_class` + `register_backend` | `vllm/v1/attention/backends/registry.py:L34` / `L111` / `L211` | 枚举只留 FLASH_ATTN（值改指本章模块以便 host 懒加载）+ 两代表项 + CUSTOM；删 ~20 同构项与 Mamba 注册表 | 懒加载/覆盖机制核心；删项为同构条目 |
| `get_attn_backend` / `AttentionSelectorConfig` / `_cached_get_attn_backend` | `vllm/v1/attention/selector.py:L53` / `L22` / `L106` | 把全局 `VllmConfig` 读取改为可选参数 `backend`/`block_size`；删 kv_cache_dtype assert | host 无全局 VllmConfig；f18 选后端公开入口 |
| `get_attn_backend_cls` / `get_valid_backends` / `_get_backend_priorities` | `vllm/platforms/cuda.py:L282` / `L248` / `L79` | 删 MLA 分支、FLEX/TURBOQUANT 兜底项、诊断日志与 block-size warning；`DeviceCapability` 给最小可比较实现 | 显式/自动两路选后端逻辑必讲；删项不改选择结果 |
| `FlashAttentionBackend.get_kv_cache_shape` / `get_kv_cache_stride_order` | `vllm/v1/attention/backends/flash_attn.py:L137` / `L148` | 删 `include_num_layers_dimension` 的 6 维分支、FP8 dtype 辅助 | KV cache shape=(2,…)/stride(NHD/HND) 约定必讲 |
| `FlashAttentionMetadataBuilder.build` | `vllm/v1/attention/backends/flash_attn.py:L388` | 解构 Common 共享字段 + 装配 FA metadata；删 AOT scheduler/DCP/cascade/cudagraph 中段分支 | 展示『共享字段直接搬 + 特有字段新增』翻译模式 |
| `FlashAttentionImpl.forward` / `do_kv_cache_update` | `vllm/v1/attention/backends/flash_attn.py:L682` / `L851` | 走 `not use_cascade` + `dcp_world_size==1` 主分支；删 encoder/FP8/DCP/cascade 分支 | `kv_cache.unbind(0)` 拆 K/V、`flash_attn_varlen_func` 照 block_table 读、`reshape_and_cache_flash` 照 slot_mapping 写——f14 读写两半 |
| `reshape_and_cache_flash` | `vllm/_custom_ops.py:L2713` | 真实仅转发 `torch.ops._C_cache_ops`；host 给 CPU 等价（slot→block_idx/off 散写） | PagedAttention 写半边；host 无 CUDA kernel |
| `flash_attn_varlen_func` | `vllm/vllm_flash_attn`（kernel） | host 给 CPU 等价（照 block_table 读 paged KV + causal GQA softmax） | PagedAttention 读半边；host 无 CUDA kernel |
| `Attention.__init__` / `forward` | `vllm/model_executor/layers/attention/attention.py:L298` / `L409` | 走 `use_direct_call=True` 直调；删量化/mm_prefix/告警；以 prefix 注册进 forward_context | f18 后端选择那一头 + 分发 |
| `get_attention_context` / `unified_kv_cache_update` / `unified_attention_with_output` | `vllm/model_executor/layers/attention/attention.py:L620` / `L662` / `L705` | 删 `direct_register_custom_op` 注册壳/`*_fake`/`maybe_transfer_kv_layer`；`forward_context` 用最小替身 | f18 取数那一头（按 layer_name 取 kv_cache/metadata/slot_mapping）+ 写/算两算子 |

## 验收

- `python3 -m pytest tests/` → 18 passed（纯 host，不 import vllm）。
- 覆盖：KV cache shape/stride 约定、`validate_configuration` 探针聚合、懒加载+覆盖、Hopper 自动选
  FLASH_ATTN、Common→FA metadata 翻译、PagedAttention 写（按 slot_mapping）、读（按 block_table，
  单请求对照稠密 causal 注意力 `allclose`）、f18 端到端按 layer_name 分发。
- 真实 CUDA 路径（实际 kernel）须进容器 `scripts/vllm_docker.sh` 跑；本精简版只在 host 验证可观察语义。
