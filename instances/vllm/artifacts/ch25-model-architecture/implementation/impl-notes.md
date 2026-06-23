# ch25 implementation notes — DeepSeek-V4 as deltas over Llama（只做减法）

本章是「读一个完整大模型」的 capstone。精简版的目标（与 dossier 一致）：让读者看懂
DeepSeek-V4 相对 Llama 基线(ch22)的四类 delta 的**骨架、控制流与算子边界**，而非在 host 复现
GPU 数值。V4 前向本质依赖 GPU-only 自定义算子（`torch.ops.vllm.deepseek_v4_attention` /
`deepseek_v4_fp8_einsum` / `deepseek_v4_mega_moe_experts` / `mhc_pre` / `mhc_post`）+ SM100 +
DeepGEMM + tilelang，因此**这些算子保留调用边界，内核实现下放对应专章**（注意力内核 ch24、
FusedMoE/EP ch26、投机解码 ch28）。

## 文件

- `deepseek_v4.py` —— 模型主体：`DeepseekV4MLP / DeepseekV4FP8Config / DeepseekV4MegaMoEExperts /
  DeepseekV4MoE / DeepseekV4Attention / DeepseekV4DecoderLayer / DeepseekV4Model / hc_head /
  DeepseekV4ForCausalLM`（+ `make_deepseek_v4_expert_params_mapping`）。
- `deepseek_v4_attention.py` —— MLA 执行层：`DeepseekV4MultiHeadLatentAttentionWrapper`
  （`forward / attn_gemm_parallel_execute / attention_impl`）。
- `deepseek_v4_mtp.py` —— MTP draft：`SharedHead / DeepSeekV4MultiTokenPredictorLayer /
  DeepSeekV4MultiTokenPredictor / DeepSeekV4MTP`。
- `_runtime.py` —— host 脚手架（TP/EP 上下文、norm/act/linear/embedding/rope/quant 占位），
  把依赖 CUDA / DeepGEMM / 分布式的实体替换成 import-time 可用的 native 占位；正文不讲它。

## 1:1 Source Map（精简版 ↔ 真实 vllm ↔ 改动 ↔ 原因）

| 精简版符号 | 真实 vLLM（pin f3fef123） | 改动 | 原因 |
|---|---|---|---|
| `DeepseekV4MLP` | `models/deepseek_v4.py:70` | 原样保留 | shared_experts 复用，结构同构 LlamaMLP，是 MoE 内 dense 路径 |
| `DeepseekV4FP8Config.expert_dtype/is_scale_e8m0` | `models/deepseek_v4.py:121-209` | 删 `override_quantization_method/is_mxfp4_quant` 等分发样板 | dossier 批准删配置样板；保留惰性解析核心表达「量化 delta」 |
| `DeepseekV4MegaMoEExperts.forward` | `models/deepseek_v4.py:392-623` | 删 `_run_mega_moe/get_symm_buffer/_check_runtime_supported`、`finalize_weights` 内部 DeepGEMM 布局变换 | dossier 批准删 DeepGEMM 内核细节（下放 ch26）；保留 `torch.ops.vllm.deepseek_v4_mega_moe_experts` 算子边界 + w13/w2 量化权重形状 |
| `DeepseekV4MoE.forward/_forward_fused_moe` | `models/deepseek_v4.py:707-918` | 删 use_mega_moe×scoring/expert_dtype 兼容校验早退 | dossier 批准删配置样板；保留 gate→fused_topk_bias→experts+=shared 主数据流与双后端选择 |
| `DeepseekV4Attention.__init__` | `models/deepseek_v4.py:921-1086` | 删 yarn rope 标志拼装、`compress_ratio==4` 建 indexer 分支、`DeepseekV4MLAModules` 容器 | dossier 批准删稀疏/yarn（下放 ch24）；MLA 低秩投影 fused_wqa_wkv/wq_b/wo_a/wo_b/q_norm/kv_norm/attn_sink 全保留 |
| `DeepseekV4DecoderLayer.hc_pre/hc_post/forward` | `models/deepseek_v4.py:1089-1216` | 删 lazy `import vllm...mhc`（tilelang 注册）| dossier：mhc 内核 GPU-only；保留 hc_pre→attn_norm→attn→hc_post / hc_pre→ffn_norm→ffn→hc_post 控制流（取代 add-norm 的 delta 焦点）|
| `DeepseekV4Model.forward/load_weights` | `models/deepseek_v4.py:1219-1447` | 删 PP/intermediate_tensors 分支、aux_stream/topk_buffer GPU 实例化、stacked_mapping 长尾 | dossier 批准删 PP/装载长尾；保留 repeat(hc_mult) 多流展开 + `_mtp_hidden_buffer` 暂存 + hc_head 压回 + 三类装载特例（e8m0fnu→uint8 view / expert_mapping 多副本 / attn_sink TP 切）|
| `hc_head` | `models/deepseek_v4.py:1450-1466` | 原样保留（纯 PyTorch）| 「混合残差」收尾真身，可逐行读懂，被主模型与 MTP 复用；测试做了逐项数值核对 |
| `DeepseekV4ForCausalLM` | `models/deepseek_v4.py:1507-1568` | 删 `_make_deepseek_v4_weights_mapper` regex 长尾 | dossier 批准删装载长尾；保留 forward/compute_logits/get_mtp_target_hidden_states/load_weights 接口边界 |
| `DeepseekV4MultiHeadLatentAttentionWrapper` | `layers/deepseek_v4_attention.py:106-412` | 改 `PluggableLayer`→`nn.Module`、`DeepseekV4MLAModules`→平铺 kwargs；删 compressor/indexer 实现、QuantFP8/einsum recipe GPU 分支 | dossier 批准删稀疏/FP8 内核（下放 ch24）；保留 forward 输出端两段低秩投影 + `attn_gemm_parallel_execute` 多流编排 + `attention_impl` 低秩 split+fused RMSNorm 前处理 |
| `DeepSeekV4MultiTokenPredictorLayer/.../DeepSeekV4MTP` | `models/deepseek_v4_mtp.py:61-275` | 删 `load_weights`（_rewrite_spec_layer_name/逐层校验 450+ 行）| dossier 批准删装载长尾；保留 enorm/hnorm + e_proj/h_proj 融合两路信号 + 复用 DeepseekV4DecoderLayer + compute_logits 补 hc_head |

## 验收判据

把真实 `deepseek_v4.py / deepseek_v4_attention.py / deepseek_v4_mtp.py` 删掉所有标 `# SUBTRACTED`
的分支（稀疏 indexer/compressor、DeepGEMM/MegaMoE 内核内部、装载长尾、PP/yarn/配置分发样板、
GPU-only stream/event/量化算子内核），应当 ≈ 得到本精简版。`must_keep` 的 27 个符号全部原样保留
（`lint_fidelity` 校验通过，无 BLOCKING）。

## 测试

`tests/test_deepseek_v4_deltas.py`（19 个，host，不 import vllm）覆盖四类 delta 的可观察结构与
纯 PyTorch 段数值：MLA 低秩 fused_wqa_wkv/wq_b/wo_a/wo_b 形状与 attn_sink；MoE gate+shared+双后端
选择 + fused_topk_bias top-k/renorm + MegaMoE uint8 量化权重；hc 多流超连接结构 + hc_head 逐项数值核对
+ 多流压回单流；主干 _mtp_hidden_buffer 桥 + get_mtp_target_hidden_states；量化 delta（expert_dtype
惰性默认 fp4、e8m0fnu→uint8 字节装载）；MTP 融合两路信号 + compute_logits 补 hc_head。
