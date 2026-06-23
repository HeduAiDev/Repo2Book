# ch22 精简版实现笔记（只做减法）

## 范围
确立 vLLM v1 模型契约，以 Llama 为最简参考：`(vllm_config, prefix)` 构造约定、
QKV/Row/MergedColumnParallelLinear 的 TP 切分、Attention 统一封装入口、按
`packed_modules_mapping`/`stacked_params_mapping` 的权重装载，以及三段式
`initialize_model → load_weights → process_weights_after_loading`。

## 文件
- `llama.py` —— 正文主线：`LlamaForCausalLM/LlamaModel/LlamaDecoderLayer/LlamaAttention/LlamaMLP`，
  与 `vllm/model_executor/models/llama.py` 同名同结构同控制流，只删 PP/Eagle/LoRA注入/GGUF/量化/sliding-window 分支。
- `linear.py` —— TP 线性层：`LinearBase/ColumnParallelLinear/MergedColumnParallelLinear/QKVParallelLinear/RowParallelLinear`，
  与 `vllm/model_executor/layers/linear.py` 同名，只删量化/GGUF/bitsandbytes/weight_loader_v2/Phi-3 已-fuse 分支。
- `_runtime.py` —— **脚手架，非正文重点**：让模型定义与线性层脱离 CUDA 在 host 跑起来的最小运行时
  （TP 上下文、norm/act/rope/embedding 的 forward_native、Attention 的 eager SDPA 替身、权重装载基础设施、最小 VllmConfig）。

## 减法判据
把真实 vLLM 删掉所有 `# SUBTRACTED` 分支 ≈ 得到本精简版。最关键的一处不可避免替换：
真实 `Attention.forward` 经 `unified_kv_cache_update + unified_attention_with_output` 两个自定义算子
（经 `forward_context` 取 `attn_metadata`/分页 `kv_cache`，再 `self.impl.forward` 调具体 backend）。
host 无 CUDA/分页 KV cache/metadata，精简版用 **eager full-causal SDPA**（含 GQA 的 KV 头复制）等价
复现 decoder-only 注意力的可观察输出，使模型定义能跑通并数值对照。该替换全程 `# SUBTRACTED` 标注。

## 1:1 Source Map（精简版 ↔ 真实 vllm ↔ 改动 ↔ 原因）

| 精简版符号 | 真实 vllm | 改动 | 原因 |
|---|---|---|---|
| `llama.LlamaForCausalLM` | `vllm/model_executor/models/llama.py:501` | 删 SupportsLoRA/PP/Eagle 基类与 PP `lm_head=PPMissingLayer` 分支 | LoRA/PP/投机解码特性不在本章；单 PP-stage 恒走 is_last_rank 主路径 |
| `llama.LlamaModel.load_weights` | `llama.py:436` | 删 cos/sin_cached skip、kv-scale 重映射、GPTQ bias 跳过、PP missing 判定 | 非量化 BF16 safetensors 主线下这些分支恒不触发；保留 stacked_params_mapping for/else 双路 |
| `llama.LlamaModel.__init__` | `llama.py:350` | 删 `@support_torch_compile`、`EagleModelMixin`、PP `PPMissingLayer`/`make_empty_intermediate_tensors` | torch.compile 是 custom-ops 章主题；Eagle/PP 不在本章；单卡 start=0 end=num_layers |
| `llama.LlamaAttention.__init__` | `llama.py:124` | 删 sliding-window/layer_types/Eagle3、`EncoderOnlyAttention` 分支 | Llama 标准配置无 sliding window；decoder-only 恒 `attn_cls=Attention` |
| `linear.ColumnParallelLinear.weight_loader` | `linear.py:534` | 删 GGUF/materialize、bitsandbytes/is_sharded_weight | 非量化 quant_config=None 恒走 `output_dim` narrow 主路径 |
| `linear.QKVParallelLinear.weight_loader` | `linear.py:1187` | 删 GGUF/BlockQuantScale/packed_dim/bitsandbytes 与 `loaded_shard_id is None`(磁盘已 fuse) 拆分递归 | 只讲非量化主路径；保留 q 用 tp_rank、k/v 用 tp_rank//num_kv_head_replicas 的关键对比 |
| `linear.MergedColumnParallelLinear.weight_loader` | `linear.py:694` | 同上删量化/GGUF/Phi-3 已 fuse 分支 | 保留 shard_id(0=gate,1=up) 的 offset/size + tp_rank narrow 主路径 |
| `linear.RowParallelLinear.forward` | `linear.py:1543` | 删 input 非并行的 split 分支（恒 input_is_parallel） | o_proj/down_proj 的 input 来自列并行已切好的输出；保留 bias-rank0-only + all_reduce |
| `_runtime.Attention.forward` | `attention/attention.py:409` | 用 eager SDPA 替 unified_* 自定义算子两步 | host 无 CUDA/分页 KV cache；保留 prefix→static_forward_context 注册与 reshape 主线 |
| `_runtime.initialize_model` / `process_weights_after_loading` / `load_model` | `model_loader/utils.py:40,99`、`base_loader.py:43`、`default_loader.py:376` | 删 set_current_vllm_config 上下文、量化 kernel 重排、torchao/EP/PP 编排 | 非量化单卡三段式主线：建空壳→流式装载→Attention 后处理→eval |

## 自检
- `python3 -m pytest tests/` → 17 passed（TP 切分/GQA 复制/qkv·gate_up fuse/三段式装载/forward 形状/causal/residual 协议）。
- `python3 scripts/lint_fidelity.py instances/vllm/artifacts/ch22-model-definitions` → 全部通过，无 BLOCKING。
